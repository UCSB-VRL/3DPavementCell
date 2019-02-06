import numpy as np
import skimage
from skimage.transform import rescale,resize
from skimage.filters import gaussian
import SimpleITK as sitk
from matplotlib import pyplot as plt

img = sitk.ReadImage('/home/tom/celldata/backgroundSubtracted_T01.tif')
img = sitk.GetArrayFromImage(img)
img = img/(2**8-1)
img = resize(img,(5*img.shape[0],img.shape[1],img.shape[2]))
img = (img*255).astype('uint8')
img = sitk.GetImageFromArray(img)
sitk.WriteImage(img,'/home/tom/interpolate/'+'first.tif')


