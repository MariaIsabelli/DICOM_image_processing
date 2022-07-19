#Often in our work, we need to convert DICOM format into ordinary images and resize them—for example, converting DICOM format into PNG image format.

#This option works but does not consider the metadata related to the image, especially concerning the BitsAllocated and PhotometricInterpretation values.

#The PhotometricInterpretation value determines the intended interpretation of the pixel data. Here, we are interested in the two main values of the X-ray:

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
