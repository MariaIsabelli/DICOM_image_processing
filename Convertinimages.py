#Often in our work, we need to convert DICOM format into ordinary images and resize them—for example, converting DICOM format into PNG image format.

#This option works but does not consider the metadata related to the image, especially concerning the BitsAllocated and PhotometricInterpretation values.

#The PhotometricInterpretation value determines the intended interpretation of the pixel data. Here, we are interested in the two main values of the X-ray:
#MONOCHROME1 is pixel data that represents a single monochrome image plane. The minimum sample value is intended to be displayed as white after any volume of interest (VOI) grayscale transformations are performed.
#MONOCHROME2 is pixel data that represents a single monochrome image plane. The minimum sample value is intended to be displayed as black after any VOI grayscale transformations are performed.

from skimage.transform import resize 
import cv2
import pydicom
import numpy as np
from matplotlib import pyplot as plt 
example = 'stage_2_images/ID_01fe90211.dcm' 
imagedata= pydicom.dcmread(example)
img =imagedata.pixel_array
name = example.split('/')[-1][:-4]
img = resize(img,(512,512)) 
cv2.imwrite('output/{}.png'.format(name), img * 255)
print(imagedata.BitsAllocated) 
print(imagedata.PhotometricInterpretation)
