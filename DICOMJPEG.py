!pip install pydicom


import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

from PIL import Image
from skimage import io
import cv2

import tensorflow as tf

import albumentations 

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 15

from pathlib import Path

INPUT_PATH = Path("/kaggle/input/siim-isic-melanoma-classification")

#Reading data

train_metadata = pd.read_csv(INPUT_PATH/"train.csv");print(f"Train shape: {train_metadata.shape}")
train_metadata.head()

# Select a random image for analysis

image_name = np.random.choice(train_metadata.image_name)
image_name

# DICOM format

dicom_arr = convert_color_space(ds.pixel_array, "YBR_FULL_422", "RGB")
dicom_img = Image.fromarray(dicom_arr)

# JPEG format

jpeg_img = Image.open(f"{INPUT_PATH}/jpeg/train/{image_name}.jpg")
jpeg_arr = np.asarray(jpeg_img)

# TFRecord format

dataset = tf.data.TFRecordDataset(tf.io.gfile.glob('/kaggle/input/siim-isic-melanoma-classification/tfrecords/train*.tfrec'))
dataset = dataset.map(mapping_func)

tfrecord_arr = filter_tfrecords(dataset, image_name)
tfrecord_img = Image.fromarray(tfrecord_arr)

print(f"DICOM array shape: {dicom_arr.shape}")
print(f"JPEG array shape: {jpeg_arr.shape}")
print(f"TFRecord array shape: {tfrecord_arr.shape}")

# DICOM vs JPEG

is_same_image(dicom_arr, jpeg_arr)

mse(dicom_arr, jpeg_arr) #Mean Squared Error

# Write DICOM image to JPEG at 75% quality

dicom_img.save(f"dicom2jpeg_75q_{image_name}.jpg", "JPEG", quality=75)

# Read the image

dicom2jpeg_75q_img = Image.open(f"dicom2jpeg_75q_{image_name}.jpg")
dicom2jpeg_75q_arr = np.asarray(dicom2jpeg_75q_img)

is_same_image(jpeg_arr, dicom2jpeg_75q_arr)

mse(jpeg_arr, dicom2jpeg_75q_arr)

#That's interesting! Let's check this for randomly selected 100 images.

image_names = np.random.choice(train_metadata.image_name, size=100, replace=False)
comparisons = Parallel(n_jobs=8, backend='threading')(delayed(
    compare)(image_name) for image_name in tqdm(image_names, total=len(image_names)))

comparisons_df = pd.DataFrame(comparisons)
comparisons_df.head()

print(np.sum(~comparisons_df["jpeg_vs_dicom2jpeg_75q_is_same"]))

print(np.sum(comparisons_df["jpeg_vs_dicom2jpeg_75q_mse"]))

#Source of TFRecord

resize_and_crop = albumentations.Compose([
    albumentations.SmallestMaxSize(max_size=1024, interpolation=cv2.INTER_LINEAR),
    albumentations.CenterCrop(1024, 1024)
])
resized_cropped_dicom_arr = resize_and_crop(image=dicom_arr)["image"]

is_same_image(tfrecord_arr, resized_cropped_dicom_arr)

mse(tfrecord_arr, resized_cropped_dicom_arr)

 #DICOM to PNG/JPEG
  
  # Write DICOM image to JPEG at 100% quality
dicom_img.save(f"dicom2jpeg_100q_{image_name}.jpg", "JPEG", quality=100)

# Read the image
dicom2jpeg_100q_img = Image.open(f"dicom2jpeg_100q_{image_name}.jpg")
dicom2jpeg_100q_arr = np.asarray(dicom2jpeg_100q_img)

is_same_image(dicom_arr, dicom2jpeg_100q_arr)

mse(dicom_arr, dicom2jpeg_100q_arr)

# Write DICOM image to PNG at 100% quality
dicom_img.save(f"dicom2png_100q_{image_name}.png", "PNG", quality=100)

# Read the image
dicom2png_100q_img = Image.open(f"dicom2png_100q_{image_name}.png")
dicom2png_100q_arr = np.asarray(dicom2png_100q_img)

is_same_image(dicom_arr, dicom2png_100q_arr)

mse(dicom_arr, dicom2png_100q_arr)

#Conclusions
#JPEG images are at 75% quality compared to images in DICOM files
#There is a minimal loss of information while converting DICOM image to JPEG with 100 quality.
#There is zero loss of information while converting DICOM image to PNG with 100 quality.
