import cv2
import numpy as np
from skimage.feature import hog
from DataProcessingModel import DataProcessing as DP
import os
folder_daisy_path=r"Data/daisy"
X,Y=DP.proData(folder_daisy_path,1,10)
print(len(X[6]))