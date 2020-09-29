from init_param import get_folder_content
from PIL import Image
import matplotlib.animation as animation
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_Animation(path1):
    frames1 = []  # for storing the generated images
    frames2 = []
    fig1 = plt.figure()
    for input1 in (os.listdir(path1)): # 64 images
        if input1[-4:] == '.png':
            image_GT = Image.open(os.path.join(path1, input1))
            inputImg = np.array(image_GT)
            frames1.append([plt.imshow(inputImg, animated=True)])

    ani1 = animation.ArtistAnimation(fig1, frames1, interval=50, blit=True,
                                        repeat_delay=None)
    plt.show()

if __name__ == "__main__":
    GT_Folder = 'Test_GT/'
    Syn_Folder = 'Test_Synthesis/'

    [GTNames, GTPaths, numGT] = get_folder_content(GT_Folder)
    [SynNames, SynPaths, numSyn] = get_folder_content(Syn_Folder)
    image_idx = input("Please type a number from 1 - 7:") # image index to be presented
    image_idx = int(image_idx)
    select = input("Please type GT or Syn to select the animation of groundtruth or synthesized lightfields:")

    if select == "GT":
        generate_Animation(GTPaths[image_idx - 1])
    if select == "Syn":
        generate_Animation(SynPaths[image_idx - 1])