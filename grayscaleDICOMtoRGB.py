import numpy as np
import pydicom
from PIL import Image, ImageDraw

# your file paths
INPUT_DICOM_PATH = ''
MASK_PATH = ''
OUTPUT_DICOM_PATH = ''

def apply_ct_window(img, window):
    # window = (window width, window level)
    R = (img-window[1]+0.5*window[0])/window[0]
    R[R<0] = 0
    R[R>1] = 1
    return R
  
ds = pydicom.dcmread(INPUT_DICOM_PATH)
img = ds.pixel_array # dtype = uint16
img = img.astype(float)
img = img*ds.RescaleSlope + ds.RescaleIntercept

display_img = apply_ct_window(img, [400,50])

# for this particular example
top, left, bottom, right = [211,99,291,158]
thickness = 4
img_bbox = Image.fromarray((255*display_img).astype('uint8'))
img_bbox = img_bbox.convert('RGB')
draw = ImageDraw.Draw(img_bbox)
for i in range(thickness):
    draw.rectangle(
        [left + i, top + i, right - i, bottom - i],
        outline=(255,0,0)
    )
del draw
img_bbox = np.asarray(img_bbox)

# modify DICOM tags
ds.PhotometricInterpretation = 'RGB'
ds.SamplesPerPixel = 3
ds.BitsAllocated = 8
ds.BitsStored = 8
ds.HighBit = 7
ds.add_new(0x00280006, 'US', 0)
ds.is_little_endian = True
ds.fix_meta_info()

# save pixel data and dicom file
ds.PixelData = img_bbox.tobytes()
ds.save_as(OUTPUT_DICOM_PATH)
