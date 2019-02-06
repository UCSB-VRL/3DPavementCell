import numpy as np
import os
from skimage.morphology import label
import SimpleITK as sitk
from celldataset import IndexTracker
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
from scipy.signal import medfilt
from skimage.measure import label
from celldataset import IndexTracker
from frangi3d import frangi
def main():
    folder = '/home/tom/Modified-3D-UNet-Pytorch/hist_match_interp'
    files = os.listdir(folder)
    for file in files:
        img = sitk.ReadImage(os.path.join(folder,file))
        img = sitk.GetArrayFromImage(img)
        img = img.astype('float32')
        img = img/255
        img_filter = denoise_tv_chambolle(img) 
        img_filter = (img_filter*255).astype('uint8')
        img_filter = sitk.GetImageFromArray(img_filter)
        sitk.WriteImage(img_filter,'/home/tom/Modified-3D-UNet-Pytorch/hist_match_median/'+file)

main()
