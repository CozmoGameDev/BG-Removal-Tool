from typing import Final as Const

import tensorflow.lite as tflite
from keras import Model

import numpy as np

import cv2
from cv2 import Mat

# Load Image Path
image_path: Const[str] = "image.png"

# Load the DeepLabV3 model from the .tflite file
model_path: Const[str] = "deeplabv3_1_default_1.tflite"
model: Const[Model] = tflite.Interpreter(model_path=model_path)
model.allocate_tensors()

# Get input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

# Function to remove background from an image
def remove_background(image_path: str):
    # Load the image
    image: Mat = cv2.imread(image_path)

    # Preprocess the image
    input_shape = input_details[0]['shape'][1:3]
    input_image = cv2.resize(image, input_shape)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = (input_image.astype(np.float32) / 127.5) - 1.0

    # Set the input tensor
    model.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    model.invoke()

    # Get the output tensor
    mask: Mat = model.get_tensor(output_details[0]['index'])[0]

    # Post-process the mask
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    # Resize the mask to match the input image size
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert mask to the same data type as the image
    mask = mask.astype(np.uint8)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

# Test the function with an image
result: Const[str] = remove_background(image_path)

# Display the result
cv2.imshow("Original Image", cv2.imread(image_path))
cv2.imshow("Masked Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()