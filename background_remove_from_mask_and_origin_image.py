import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize

images = load_img("input_images/bike-mask.png")  #Mask Image Input
# image = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
RESCALE = 255
out_img = img_to_array(images, dtype=np.float32)
out_img /= RESCALE

THRESHOLD = 0.8

out_img[out_img > THRESHOLD] = 1
out_img[out_img <= THRESHOLD] = 0

shape = out_img.shape
a_layer_init = np.ones(shape = (shape[0], shape[1], 1))
mul_layer = np.expand_dims(out_img[:, :, 0], axis = 2)
a_layer = mul_layer*a_layer_init
rgba_out = np.append(out_img, a_layer, axis = 2)

rgba_out_mask = Image.fromarray((rgba_out*RESCALE).astype('uint8'), 'RGBA')

input = load_img("input_images/bike-origin.jpeg") #Original Image Input
inp_img = img_to_array(input, dtype=np.float32)
inp_img /= RESCALE

a_layer = np.ones(shape = (shape[0], shape[1], 1))
rgba_inp = np.append(inp_img, a_layer, axis = 2)

rem_back = (rgba_inp * rgba_out)
rem_back_scaled = Image.fromarray((rem_back*RESCALE).astype('uint8') , 'RGBA')

converted_image = rem_back_scaled.convert("RGBA")

converted_image.show()
rem_back_scaled.save("results/bike.png") #Result Image Save

background_input = load_img("R.jpg")
background_inp_img = img_to_array(background_input, dtype = np.float32)
RESCALE = 255
background_inp_img /= RESCALE

background_height = background_inp_img.shape[0]
background_width = background_inp_img.shape[1]
print(background_height, background_width)

resized_rem_back = resize(rem_back, (background_height, background_width))
output_chbg = np.zeros((background_height, background_width, 3))
output_chbg[:,:,0] = background_inp_img[:,:,0]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,0]
output_chbg[:,:,1] = background_inp_img[:,:,1]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,1]
output_chbg[:,:,2] = background_inp_img[:,:,2]*(1-resized_rem_back[:,:,3])+resized_rem_back[:,:,2]

output_chbg_scaled = Image.fromarray((output_chbg*RESCALE).astype('uint8'), 'RGB')
output_chbg_scaled.show()
output_chbg_scaled.save("background_changed_result/bike_background_chaged.png")