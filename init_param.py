import os

import numpy as np
import torch


def get_folder_content(folderPath):
    contentNames = [] # store directory name for each scene
    contentPaths = [] # store directory path for each scene
    numContents = 0
    dir = os.path.expanduser(folderPath)
    for target in sorted(os.listdir(dir)): # directory name
        d = os.path.join(dir, target) # file path
        contentNames.append(target)
        contentPaths.append(d)
        numContents += 1
    return contentNames, contentPaths, numContents


param = type('', (), {})() # using type() to create class object

param.depthResolution = 45  # number of depth levels (see Eq. 5)
param.numDepthFeatureChannels = param.depthResolution * 2
#param.deltaDisparity = 21  # the range of disparities (see Eq. 5)
param.origAngRes = 8  # original angular resolution
param.depthBorder = 6  # the number of pixels that are reduce due to the convolution process in the depth network. This border can be avoided if the networks are padded appropriately.
param.colorBorder = 6  # same as above, for the color network.
param.testNet = 'TrainingData'

### here, we set the desired novel views and indexes of the input views.
novelView = type('', (), {})()
inputView = type('', (), {})()

[novelView.X, novelView.Y] = np.mgrid[0:1.1:1 / 7., 0:1.1:1 / 7.] # 新视角，range:[0, 1], size: 8 * 8
[inputView.X, inputView.Y] = np.mgrid[0:2., 0:2.] # 输入顶点视角，range:{0, 1}, size: 2 * 2
novelView.Y = novelView.Y.conj().transpose()
novelView.X = novelView.X.conj().transpose()
novelView.Y = novelView.Y.flatten()
novelView.X = novelView.X.flatten()
inputView.Y = inputView.Y.flatten()
inputView.X = inputView.X.flatten()

################################################################################

param.useGPU = torch.cuda.is_available()

################## Training Parameters #################################

param.height = 250
param.width = 400

param.patchSize = 60
param.stride = 16
param.numRefs = 4  # number of reference images selected randomly on each light field
param.cropSizeTraining = 70  # we crop the boundaries to avoid artifacts in the training
param.batchSize = 10  # we found batch size of 10 is faster and gives the same results

param.cropHeight = param.height - 2 * param.cropSizeTraining
param.cropWidth = param.width - 2 * param.cropSizeTraining

param.trainingScenes = 'TrainingData/Training/'
param.trainingData = 'TrainingData/Training/'
_, param.trainingNames, _ = get_folder_content(param.trainingData)

param.testScenes = 'TrainingData/Test/'
param.testData = 'TrainingData/Test/'
_, param.testNames, _ = get_folder_content(param.testData)

param.trainNet = 'TrainingData'

param.isContinue = False
param.startIter = 0
param.totalIter = 20000

param.testNetIter = 1000
param.printInfoIter = 1

### ADAM parameters
param.alpha = 0.0001
param.beta1 = 0.9
param.beta2 = 0.999
param.eps = 1e-8
