from typing import Final as Const

import tensorflow.lite as tflite
from keras import Model

import numpy as np

import cv2
from cv2 import Mat

# Load the DeepLabV3 model from the .tflite file
model_path: Const[str] = "deeplabv3_1_default_1.tflite"
model: Const[Model] = tflite.Interpreter(model_path=model_path)
model.allocate_tensors()

# Get input and output details
input_details = model.get_input_details()
output_details = model.get_output_details()

GREEN: Const[tuple] = (0, 255, 0)

# Function to remove background from an image
def remove_background(image: Mat):
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
    masked_image[np.where((mask == 0))] = GREEN

    return masked_image

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video capture
    ret, frame = video_capture.read()

    # Apply background removal function to the frame
    result = remove_background(frame)

    # Display the original frame and the masked image
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Masked Image", result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()