import numpy as np
import cv2
import os
from PIL import Image
from image_processing import tensor_to_image, process_image

def censor_eyes(image_path, output_path, censor_width, censor_height):
    """Censor the eyes in an image by applying a black rectangle."""
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale for eye detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Censor all detected eyes with a single black rectangle
    if len(eyes) > 0:
        x_min = min([x for (x, y, w, h) in eyes])
        y_min = min([y for (x, y, w, h) in eyes])
        x_max = max([x + w for (x, y, w, h) in eyes])
        y_max = max([y + h for (x, y, w, h) in eyes])
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
    # Save the censored image
    cv2.imwrite(output_path, image)
    print(f"Censored image saved to {output_path}")

def gaussian_blur_whole_face(image_path, output_path, sigma):
    """"Apply Gaussian blur to the whole face in an image."""
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Apply Gaussian blur to each detected face
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_region, (0, 0), sigma)
        image[y:y+h, x:x+w] = blurred_face

    # Save the blurred image
    cv2.imwrite(output_path, image)
    print(f"Blurred image saved to {output_path}")

def mosaic_whole_face(image_path, output_path, mosaic_size):
    """Apply mosaic effect to the whole face in an image."""
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Apply mosaic effect to each detected face
    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        small_face = cv2.resize(face_region, (mosaic_size, mosaic_size))
        mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = mosaic_face

    # Save the mosaiced image
    cv2.imwrite(output_path, image)
    print(f"Mosaiced image saved to {output_path}")


#Print image:
def print_image_path(image_path):
    """Print the image using PIL."""
    image = Image.open(image_path)
    image.show()