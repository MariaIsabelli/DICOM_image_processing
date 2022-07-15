!pip install pydicom 

import numpy as np
import pydicom
from PIL import Image
import os

def get_names(path):
  names = []
  for root, dirnames, filenames in os.walk(path):
    for filename in filenames:
      _, ext = os.path.splitext(filename)
      if ext in ['.dcm']:
        names.append(filename)
        return names
      
      def convert__dcm_jpg(name):
  
  im = pydicom.dcmread('/content/drive/MyDrive/DICOM_MRI_CORTES - Copia-20220628T192641Z-001/DICOM_MRI_CORTES - Copia/OK LUMINAL A 1095464/' + name)
  im = im.pixel_array.astype(float)
  rescaled_image = (np.maximum(im,0)/im.max())*255
  final_image = np.uint8(rescaled_image)
  final_image = Image.fromarray(final_image)
  return final_image

print(get_names('nome do diretorio de img dicom'))

names = get_names('/content/drive/MyDrive/DICOM_MRI_CORTES - Copia-20220628T192641Z-001/DICOM_MRI_CORTES - Copia/OK LUMINAL A 1095464')
for name in names:
  image = convert__dcm_jpg(name)
  image.save(name+ '.jpg')
