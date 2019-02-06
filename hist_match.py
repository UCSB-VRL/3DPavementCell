import numpy as np
from celldataset import cell_training,cell_testing
import SimpleITK as sitk
from skimage.transform import resize
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image.
    Code adapted from
    http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def main():
    cell = cell_training('/home/tom/Modified-3D-UNet-Pytorch/PNAS/')
    cell_test = cell_testing('/home/tom/celldata/data4_sub')
    train =[]
    for i in range(cell.__len__()):
        temp = cell.__getitem__(i)['data']
        #print type (train)
        train.append(temp)
    train = np.concatenate(train)
    for i,sample in enumerate(cell_test):
        file_name = sample['name']
        img_size =  sample['image_size']
        #img_size = int(img_size)
        file_name = str(file_name[0:len(file_name)-1])
        img = sample['data']
        data = img[0,:,:,:]
        data = data[0:img_size[0]*5]
        #data = resize(data,(img_size[0],512,512))
        data = hist_match(data,train)
        data = (255*data).astype('uint8')
        data = sitk.GetImageFromArray(data)
        sitk.WriteImage(data,'/home/tom/data1_match/dataset4'+file_name+'matched.tif')


main()



