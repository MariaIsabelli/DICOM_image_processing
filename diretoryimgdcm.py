!pip install python-utils

import glob 
import numpy as np

def transform_to_hu(medical_image, image):
 intercept = medical_image.RescaleIntercept 
 slope = medical_image.RescaleSlope
 hu_image = image * slope + intercept 
 return hu_image

def window_image(image, window_center, window_width):
  img_min = window_center - window_width // 2
  img_max = window_center - window_width // 2
  window_image = image.copy()
  window_image[window_image < img_min] = img_min
  window_image[window_image < img_max] = img_max
  return window_image 

def load_image(file_path):
  medical_image = dicom.read_file(file_path)
  image = medical_image.pixel_array
  
  hu_image = transform_to_hu(medical_image, image)
  brain_image = window_image (hu_image,40,80)

  return brain_image 


from keras_preprocessing.image.utils import load_img
files = (glob.glob('/content/drive/MyDrive/DICOM_MRI_CORTES/luminal A 75587 rescale ok/*.dcm'))
images = np.array([load_image(path) for path in files], dtype=object)

for i in range (1):
  plt.imshow(images[0])
  
  plt.imshow(images[1])
  
  plt.imshow(images[2])
  
  
  
  
  
