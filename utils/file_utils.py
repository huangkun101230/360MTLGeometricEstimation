import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import matplotlib.pyplot as plot

import numpy as np
import cv2

def save_depth_tensor_as_float(directory, filename, tensor):
    if not os.path.exists(directory):
        print("Given directory does not exist. Creating...")
        os.mkdir(directory)
    tensor = tensor.detach().cpu().numpy()[0]
    filepath_exr = os.path.join(directory, filename + ".png")
    plot.imsave(filepath_exr, tensor)

def save_norm_tensor_as_float(directory, filename, tensor):
    if not os.path.exists(directory):
        print("Given directory does not exist. Creating...")
        os.mkdir(directory)
    tensor = tensor.detach().cpu()
    array = tensor.squeeze(0).numpy()
    array = array.transpose(1, 2, 0)
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    filepath_exr = os.path.join(directory, filename + ".png")
    cv2.imwrite(filepath_exr, array)