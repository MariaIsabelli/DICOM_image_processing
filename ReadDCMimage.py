!pip install pydicom

import matplotlib.pyplot as plt
import pydicom as dicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
import numpy as np

import tensorflow as tf 
print(tf.__version__)

import keras as K
print(K.__version__)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  

fpath = 'caminho da img .dcm'
ds = dicom.dcmread(fpath)
plt.imshow(ds.pixel_array)
ds = dcmread(fpath)

print()
print(f"File path........: {fpath}")
print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print(f"Nome do Paciente..: {display_name}")
print(f"Registro do Paciente.......: {ds.PatientID}")
print(f"Tipo de Exame.........: {ds.Modality}")
print(f"Data.......: {ds.StudyDate}")
print(f"Tamanho da Imagem.......: {ds.Rows} x {ds.Columns}")
print(f"Espaçamento entre Pixel....: {ds.PixelSpacing}")


print(f"Local do Corte...: {ds.get('SliceLocation', '(missing)')}")


plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()

fpath ='caminho da img.dcm'
fpath2 = 'caminho da img.dcm'
fpath3 = 'caminho da img .dcm'
fpath4 = 'caminho da img.dcm'

ds = dicom.dcmread(fpath)
ds2 = dicom.dcmread(fpath2)
ds3 = dicom.dcmread(fpath3)
ds4 = dicom.dcmread(fpath4)

for i in range (1):
  axes = plt.subplots(nrows= 1, ncols= 1, figsize=(3, 5))
  plt.imshow(ds.pixel_array)
  axes2 = plt.subplots(nrows= 1, ncols= 1, figsize=(3, 5))
  plt.imshow(ds2.pixel_array)
  axes3 = plt.subplots(nrows= 1, ncols= 1, figsize=(3, 5))
  plt.imshow(ds3.pixel_array)
  axes4 = plt.subplots(nrows= 1, ncols= 1, figsize=(3, 5))
  plt.imshow(ds4.pixel_array) 
