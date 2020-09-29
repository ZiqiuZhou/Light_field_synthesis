import argparse
import os
import os.path
import warnings
from math import floor

import cv2
import numpy as np
import torch
from scipy.interpolate import *

from init_param import param, inputView, get_folder_content

warnings.filterwarnings("ignore")
import h5py


def make_dir(inputPath):
    if not os.path.exists(inputPath):
        os.makedirs(inputPath)


def crop_img(input, pad):
    return input[pad:- pad, pad: - pad]

def resize_img(img, height, width):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def get_patches(input, patchSize, stride):
    [height, width, depth] = input.shape # 372 * 372 * 12

    numPatches = (floor((width - patchSize) / stride) + 1) * (floor((height - patchSize) / stride) + 1) # patches数量400
    patches = np.zeros((patchSize, patchSize, depth, numPatches)) # 60 * 60 * 12 * 400

    count = -1
    #[iX, iY]为每个patch的top_left坐标
    for iX in np.arange(0, width - patchSize + 1, stride):
        for iY in np.arange(0, height - patchSize + 1, stride):
            count = count + 1
            patches[:, :, :, count] = input[iY: iY + patchSize, iX: iX + patchSize, :]
    return patches


def get_num_patches(): # 每个视角图片产生400个patches
    height = param.cropHeight
    width = param.cropWidth
    patchSize = param.patchSize
    stride = param.stride

    numPatchesX = floor((width - patchSize) / stride) + 1
    numPatchesY = floor((height - patchSize) / stride) + 1
    numPatches = numPatchesY * numPatchesX
    return numPatches


def get_img_pos(ind):
    pos = ind / (param.origAngRes - 1)
    return pos


def defocus_response(input): # mean
    """
    input是 Eq.5的等号左边
    :param input: size: 512 * 512 * 4
    :return: size: 512 * 512
    """
    curMean = np.nanmean(input, 2)
    curMean[np.isnan(curMean)] = 0
    output = curMean
    return output


def corresp_response(input): # standard deviation
    """
    input是 Eq.5的等号左边
    :param input: size: 512 * 512 * 4
    :return: size: 512 * 512
    """
    import warnings
    warnings.filterwarnings("ignore")
    inputVar = np.nanvar(input, 2, ddof=1)
    inputVar[np.isnan(inputVar)] = 0
    output = np.sqrt(inputVar)
    return output


def pad_with_one(input, finalLength):
    """
    :param input: (60, 60, depth), type: tuple
    :param finalLength: 4
    :return: list: [60, 60, depth, 1]
    """
    output = list(input) + np.ones((1, finalLength - len(input)), dtype=np.uint8).flatten().tolist()
    return output


def save_hdf(f, datasetName, input, inDims, startLoc, createFlag, arraySize=1):
    """
    :param f: file = h5py.File('TrainingData/Training/training.h5')
    :param datasetName: 数据集的name，为‘IN', 'FT', 'GT'和'RP'
    :param input: size: 60 * 60 * depth
    :param inDims: pad_with_one(input.shape, 4): [60, 60, depth, 1]
    :param startLoc: [0, 0, 0, writeOrder[j]],
    :param createFlag:
    :param arraySize: numTotalPatches: 400 * 4 * 100
    :return:
    """
    if createFlag:
        # 创建dataset数据集, (*inDims[0:- 1], arraySize)是数据集的shape: (60, 60, depth, 264 * 4 * 100)
        dset = f.create_dataset(datasetName, (*inDims[0:- 1], arraySize), dtype='f', chunks=tuple(inDims))
    else:
        dset = f.get(datasetName)

    sliceIdx = []
    for i in range(len(inDims)): # 4
        # _handle_simple in h5py/slection.py does not handle length=1 slices properly
        if inDims[i] == 1:
            idx = startLoc[i]
        else:
            idx = slice(startLoc[i], startLoc[i] + inDims[i])
        sliceIdx.append(idx)
    while input.shape[-1] == 1:
        input = input[..., 0]

    dset.write_direct(input.astype('float32'), dest_sel=tuple(sliceIdx))
    startLoc[-1] = startLoc[-1] + inDims[-1]
    return startLoc


def warp_images(disparity, input, delY, delX):
    [h, w, _, numImages] = disparity.shape
    X = np.arange(0, w, dtype='float')
    Y = np.arange(0, h, dtype='float')
    XX, YY = np.meshgrid(X, Y)
    points = np.zeros((h * w, 2))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    c = input.shape[2]
    output = np.zeros((h, w, c, numImages), 'float')

    for j in range(0, numImages):
        for i in range(0, c):
            curX = XX + delX[j] * disparity[:, :, 0, j]
            curY = YY + delY[j] * disparity[:, :, 0, j]
            output[:, :, i, j] = griddata(points, input[:, :, i, j].flatten(), (curX, curY), method='cubic',
                                          fill_value=np.nan)

    return output


def warp_all_images(images, depth, refPos):
    images = images.cpu().numpy()
    depth = depth.cpu().numpy()
    if torch.is_tensor(refPos):
        refPos = refPos.cpu().numpy()
    [h, w, c, numImages] = images.shape
    numInputViews = len(inputView.Y) # 4

    warpedImages = np.zeros((h, w, c, numImages), 'float')
    for i in range(0, numInputViews):
        deltaY = inputView.Y[i] - refPos[0]
        deltaX = inputView.X[i] - refPos[1]
        warpedImages[:, :, i * 3 + 1: (i + 1) * 3, :] = warp_images(depth, images[:, :, i * 3 + 1:(i + 1) * 3, :],
                                                                    deltaY, deltaX)

    return warpedImages


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def prepare_depth_features(inputLF, deltaY, deltaX, disparity):
    depthResolution = param.depthResolution
    deltaDisparity = 21
    # convert the input rgb light field to grayscale
    (height, width, _, angHeight, angWidth) = inputLF.shape
    grayLF = np.zeros((height, width, angHeight, angWidth))
    for i in range(0, angHeight):
        for j in range(0, angWidth):
            grayLF[:, :, i, j] = rgb2gray(inputLF[:, :, :, i, j])

    defocusStack = np.zeros((height, width, depthResolution))
    correspStack = np.zeros((height, width, depthResolution))
    featuresStack = np.zeros((height, width, 90))
    delta = 2 * deltaDisparity / (depthResolution - 1)
    indDepth = 0

    for curDepth in np.arange(- deltaDisparity, deltaDisparity + delta, delta):
        shearedLF = np.zeros((height, width, angHeight * angWidth))
        X = np.arange(0, width)
        Y = np.arange(0, height)

        # backward warping all the input images using each depth level (see Eq. 5)

        indView = 0
        for iax in range(0, angWidth):
            for iay in range(0, angHeight):
                curY = Y + curDepth * deltaY[indView]
                curX = X + curDepth * deltaX[indView]
                ip = interp2d(X, Y, grayLF[:, :, iay, iax], kind='cubic', fill_value=np.nan)
                shearedLF[:, :, indView] = ip(curX, curY)
                indView = indView + 1
        # computing the final mean and variance features for depth level using Eq. 6

        defocusStack[:, :, indDepth] = defocus_response(shearedLF)
        correspStack[:, :, indDepth] = corresp_response(shearedLF)

        if (indDepth + 1) % 10 == 0:
            print('\b\b\b%d%%' % ((indDepth + 1) / depthResolution * 100), end='', flush=True)

        indDepth = indDepth + 1

    featuresStack[:, :, 0: 45] = defocusStack.astype('float32')
    featuresStack[:, :, 45: 90] = correspStack.astype('float32')
    return featuresStack


def isnan(x):
    return x != x


def prepare_color_features(depth, images, refPos):
    images = crop_img(images, param.depthBorder)
    warpedImages = warp_all_images(images, depth, refPos)
    warpedImages = torch.from_numpy(warpedImages)
    if param.useGPU:
        warpedImages = warpedImages.cuda()
    warpedImages = warpedImages.float()
    indNan = isnan(warpedImages)
    warpedImages[indNan] = 0
    [h, w, _, _] = depth.shape
    if not torch.is_tensor(refPos):
        refPos = torch.from_numpy(refPos)
        if param.useGPU:
            refPos = refPos.cuda().float()
    refPos = refPos.view(2, 1, 1, -1)
    colorFeatures = torch.cat((depth, warpedImages, (refPos[0, :, :, :] - 1.5).repeat(h, w, 1, 1),
                               (refPos[1, :, :, :] - 1.5).repeat(h, w, 1, 1)), 2)
    return colorFeatures, indNan


def im2double(im):
    info = np.iinfo(im.dtype)  # Get the data type of the input image，取得该类型变量在本机器上可取值的最大值
    return im.astype(np.float) / info.max  # Divide all values by the largest possible value in the datatype


def read_illum_images(scenePath):
    """
    读取某一个场景图，生成9 * 9个视差图并截取中间8 * 8个
    """
    numImgsX = 8
    numImgsY = 8
    h = param.height
    w = param.width
    fullLF = np.zeros((h, w, 3, numImgsY, numImgsX), dtype=np.float) # 512 * 512 * 3 * 9 * 9
    deltaDisparity = []
    for input in os.listdir(scenePath): # 81 images
        path = os.path.join(scenePath, input)

        if input == 'range.txt':
            line = open(path).readlines()[0].split(' ')
            deltaDisparity.append(float(line[0])) # 存放视差range
            deltaDisparity.append(float(line[1]))
            continue

        inputImg = cv2.imread(path, -cv2.IMREAD_ANYDEPTH) # 512 * 512 * 3
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB)  # BGR to RGB
        inputImg = im2double(inputImg)  # normalize
        inputImg = resize_img(inputImg, 250, 400)# 裁剪至256 * 256
        if(int(input[-6]) != 0):
            index = int(input[-6] + input[-5])
        else:
            index = int(input[-5])
        curIdx_Y, curIdx_X = np.unravel_index(index, [numImgsY, numImgsX], 'C')
        fullLF[:, :, :, curIdx_Y, curIdx_X] = inputImg
        h = inputImg.shape[0] // numImgsY
        w = inputImg.shape[1]
        """
        if h == 375 and w == 540:
            fullLF = np.pad(fullLF, ((0, 1), (0, 1), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if h == 375 and w == 541:
            fullLF = np.pad(fullLF, ((0, 1), (0, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        """
    #fullLF = fullLF[:, :, :, 1:9, 1:9] # 挑选9 * 9 中的中间8 * 8 个, 512 * 512 * 3 * 9 * 9
    inputLF = fullLF[:, :, :, 0:8:7, 0:8:7] # 挑选8 * 8 中的4个角, 512 * 512 * 3 * 2 * 2
    return fullLF, inputLF, deltaDisparity


def compute_training_examples(curFullLF, curInputLF, deltaDisparity):
    """
    进入第一个CNN前的输入，用于计算disparity
    :param curFullLF: 512 * 512 * 3 * 9 * 9
    :param curInputLF: 512 * 512 * 3 * 2 * 2
    :param deltaDisparity:
    :return:
    """
    cropSize = param.cropSizeTraining
    numRefs = param.numRefs # 4
    patchSize = param.patchSize
    stride = param.stride
    origAngRes = param.origAngRes

    # preparing input images
    height, width, _, _, _ = curInputLF.shape # 512 * 512 * 3 * 2 * 2
    inImgs = curInputLF.reshape((height, width, -1))

    inImgs = crop_img(inImgs, cropSize) # 372 * 372 * 12
    pInImgs = get_patches(inImgs, patchSize, stride) # 大小: 60 * 60 * 12 * 400
    pInImgs = np.tile(pInImgs, (1, 1, 1, numRefs)) # 60 * 60 * 12 * 1600
    numPatches = get_num_patches()
    # selecting random four references(novel view positions) from the original 8 * 8 grid
    numSeq = np.random.permutation(origAngRes ** 2)
    refInds = numSeq[0:numRefs]

    # initializing the arrays
    pInFeat = np.zeros((patchSize, patchSize, param.numDepthFeatureChannels, numPatches * numRefs)) # input features: 60 * 60 * 200 * 1600
    pRef = np.zeros((patchSize, patchSize, 3, numPatches * numRefs)) # 4个references, 每个ref切割成numPatches个小patch, size: 60 * 60 * 3 * 1600
    refPos = np.zeros((2, numPatches * numRefs)) # reference所在[Y, X]位置， 4个位置ref,每个ref切出numRefs := 400个patches

    for ri in range(0, numRefs):
        print('Working on random reference %d of %d:' % (ri + 1, numRefs), end='   ')
        curRefPos = type('', (), {})()
        curRefInd = type('', (), {})()
        [curRefInd.Y, curRefInd.X] = np.unravel_index(refInds[ri], [origAngRes, origAngRes], 'F') # 返回索引refInds[ri]在二维8 * 8数组中的二维索引
        curRefPos.Y = get_img_pos(curRefInd.Y)
        curRefPos.X = get_img_pos(curRefInd.X)
        wInds = np.arange(ri * numPatches, (ri + 1) * numPatches)
        # preparing reference
        ref = curFullLF[:, :, :, curRefInd.Y, curRefInd.X] # 在8 * 8 grid中挑选出一个reference view
        ref = crop_img(ref, cropSize)
        pRef[:, :, :, wInds] = get_patches(ref, patchSize, stride)
        # preparing features
        deltaViewY = inputView.Y - curRefPos.Y # 输入4个顶点与reference视角的视角差
        deltaViewX = inputView.X - curRefPos.X
        inFeat = prepare_depth_features(curInputLF, deltaViewY, deltaViewX, deltaDisparity) # inFeat: 512 * 512 * 200
        inFeat = crop_img(inFeat, cropSize)
        pInFeat[:, :, :, wInds] = get_patches(inFeat, patchSize, stride)
        # preparing ref positions
        refPos[0, wInds] = np.tile(curRefPos.Y, (1, numPatches))
        refPos[1, wInds] = np.tile(curRefPos.X, (1, numPatches))
        print('\b\b\b\bDone', flush=True)
    return pInImgs, pInFeat, pRef, refPos


def compute_test_examples(curFullLF, curInputLF, deltaDisparity):
    # preparing input images
    [height, width, _, _, _] = curInputLF.shape
    inImgs = curInputLF.reshape((height, width, -1))

    curRefPos = type('', (), {})()
    curRefInd = type('', (), {})()
    curRefInd.Y = 4
    curRefInd.X = 4
    curRefPos.Y = get_img_pos(curRefInd.Y)
    curRefPos.X = get_img_pos(curRefInd.X)

    print('Working on reference (5, 5):', end='   ')

    # preparing reference
    ref = curFullLF[:, :, :, curRefInd.Y, curRefInd.X]

    # preparing features
    deltaViewY = inputView.Y - curRefPos.Y
    deltaViewX = inputView.X - curRefPos.X
    inFeat = prepare_depth_features(curInputLF, deltaViewY, deltaViewX, deltaDisparity)

    # preparing ref positions
    refPos = np.array([[curRefPos.Y], [curRefPos.X]])

    print('\b\b\b\bDone', flush=True)
    return inImgs, inFeat, ref, refPos


def write_training_examples(inImgs, inFeat, ref, refPos, outputDir, writeOrder, startInd, createFlag, arraySize):
    """
    写入某个场景的patches
    :param inImgs: 60 * 60 * 12 * 1600, 4个顶点patches
    :param inFeat: 60 * 60 * 200 * 1600 input features
    :param ref: 60 * 60 * 3 * 1600, 4个随机选取的references, 每个ref切割400个patches
    :param refPos: 2 * 1600, reference所在[Y, X]位置， 4个位置ref,每个ref切出numRefs := 400个patches
    :param outputDir: param.trainingData——  'TrainingData/Training/'
    :param writeOrder: np.random.permutation(400 * 4 * 100), 写入次序
    :param startInd: 此轮写入的场景index, 每个场景包含numPatches * param.numRefs个patches， startInd代表其中第一个patch位置
    :param createFlag: True
    :param arraySize: numTotalPatches: 400 * 4 * 100
    :return:
    """
    fileName = outputDir + '/training.h5'
    numElements = refPos.shape[1] # 1600

    file = h5py.File(fileName, "a", libver='latest')

    for k in range(0, numElements):
        j = k + startInd

        curInImgs = inImgs[:, :, :, k] # 60 * 60 * 12
        curInFeat = inFeat[:, :, :, k] # 60 * 60 * 200
        curRef = ref[:, :, :, k] # 60 * 60 * 3
        curRefPos = refPos[:, k] # 2,
        save_hdf(file, 'IN', curInImgs.astype('float32'), pad_with_one(curInImgs.shape, 4),
                 [0, 0, 0, writeOrder[j]], createFlag, arraySize)
        save_hdf(file, 'FT', curInFeat.astype('float32'), pad_with_one(curInFeat.shape, 4),
                 [0, 0, 0, writeOrder[j]],
                 createFlag, arraySize)
        save_hdf(file, 'GT', curRef.astype('float32'), pad_with_one(curRef.shape, 4), [0, 0, 0, writeOrder[j]],
                 createFlag, arraySize)
        save_hdf(file, 'RP', curRefPos.astype('float32'), pad_with_one(curRefPos.shape, 2), [0, writeOrder[j]],
                 createFlag, arraySize)
        print("\b\b\b\b%3d%%" % (k / numElements * 100), end='', flush=True)
        createFlag = False
    print("\b\b\b\bDone")
    file.close()
    return createFlag


def write_test_examples(inImgs, inFeat, ref, refPos, outputDir):
    fileName = outputDir + '.h5'
    file = h5py.File(fileName, "a", libver='latest')
    save_hdf(file, 'IN', inImgs.astype('float32'), pad_with_one(inImgs.shape, 4), [0, 0, 0, 0], True)
    save_hdf(file, 'FT', inFeat.astype('float32'), pad_with_one(inFeat.shape, 4), [0, 0, 0, 0], True)
    save_hdf(file, 'GT', ref.astype('float32'), pad_with_one(ref.shape, 4), [0, 0, 0, 0], True)
    save_hdf(file, 'RP', refPos.astype('float32'), refPos.shape, [0, 0], True)
    file.close()
    print("Done")


def prepare_training_data():
    sceneFolder = param.trainingScenes # 'TrainingData/Training/'
    outputFolder = param.trainingData # 'TrainingData/Training/'
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder) # numScences = 100, 100个不同场景图片
    numPatches = get_num_patches()
    numTotalPatches = numPatches * param.numRefs * numScenes # 400 * 4 * 20
    writeOrder = np.random.permutation(numTotalPatches)
    firstBatch = True
    make_dir(outputFolder)

    #shuffle_index = np.random.permutation(numScenes)
    for ns in range(0, numScenes): # numScenes: 20
        #idx = shuffle_index[ns]
        print('**********************************')
        print('Working on the "%s" dataset (%d of %d)' % (sceneNames[ns], ns + 1, numScenes), flush=True)

        print('Loading input light field ...', end=' ')
        curFullLF, curInputLF, deltaDisparity = read_illum_images(scenePaths[ns]) # 读取原始光场图，返回8 * 8 视角图和4个inputs
        print('Done')
        print('**********************************')

        print('Preparing training examples')
        print('------------------------------')
        [pInImgs, pInFeat, pRef, refPos] = compute_training_examples(curFullLF, curInputLF, deltaDisparity)

        print('Writing training examples...', end='    ', flush=True)
        firstBatch = write_training_examples(pInImgs, pInFeat, pRef, refPos, outputFolder, writeOrder,
                                             ns * numPatches * param.numRefs, firstBatch, numTotalPatches)


def prepare_test_data():
    sceneFolder = param.testScenes # 'TrainingData/Test/'
    outputFolder = param.testData # 'TrainingData/Test/'
    [sceneNames, scenePaths, numScenes] = get_folder_content(sceneFolder)

    for ns in range(0, numScenes):
        curOutputName = outputFolder + '/' + sceneNames[ns]

        print('**********************************')
        print('Working on the "%s" dataset (%d of %d)' % (sceneNames[ns], ns, numScenes), flush=True)

        print('Loading input light field ...', end=' ')
        curFullLF, curInputLF, deltaDisparity = read_illum_images(scenePaths[ns])
        print('Done')
        print('**********************************')

        print('Preparing test examples')
        print('------------------------------')
        [pInImgs, pInFeat, pRef, refPos] = compute_test_examples(curFullLF, curInputLF, deltaDisparity)

        print('Writing test examples...', end='', flush=True)
        write_test_examples(pInImgs, pInFeat, pRef, refPos, curOutputName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Light Field Images')
    parser.add_argument('--dataset', default='both', type=str, choices=['test', 'train', 'both'],
                        help='choose dataset to process')
    opt = parser.parse_args()
    dataset = opt.dataset
    if dataset == 'test':
        prepare_test_data()
    elif dataset == 'train':
        prepare_training_data()
    elif dataset == 'both':
        prepare_test_data()
        prepare_training_data()
