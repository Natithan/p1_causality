from __future__ import division
from PIL import Image
import math
import os
IMG_NAME = 'test_for_mturk'
TARGET_DIR = '/mnt/g/My Drive/1 PhD/Sceps/Supplementary material/'
IMAGE_TO_SLICE =f'{TARGET_DIR}/{IMG_NAME}.png'

def long_slice(image_path, out_name, outdir, slice_size):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0
    slices = int(math.ceil(height/slice_size))

    count = 1
    for slice in range(slices):
        #if we are at the end, set the lower bound to be the bottom of the image
        if count == slices:
            lower = height
        else:
            lower = int(count * slice_size)

        bbox = (left, upper, width, lower)
        working_slice = img.crop(bbox)
        upper += slice_size
        #save the slice
        working_slice.save(os.path.join(outdir, "slice_" + out_name + "_" + str(count)+".png"))
        count +=1

if __name__ == '__main__':

    long_slice(IMAGE_TO_SLICE, #image filename
               IMG_NAME, #slice names
                os.getcwd(), #output dir
                700 #height in px
               )