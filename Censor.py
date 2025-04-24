import numpy as np
import cv2
import os
from PIL import Image
from image_processing import tensor_to_image, process_image

def censor_eyes(image_path, output_path):
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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the censored image
    cv2.imwrite(output_path, image)

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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the blurred image
    cv2.imwrite(output_path, image)

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

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the mosaiced image
    cv2.imwrite(output_path, image)


#Print image:
def print_image_path(image_path):
    """Print the image using PIL."""
    image = Image.open(image_path)
    image.show()

if __name__ == "__main__":
    # Example usage
    image_path = './dataset/MF0901_1100_00F.jpg'  # Path to the input image
    output_path_censor = 'output/censored_eyes.jpg'  # Path to save the censored image
    output_path_blur = 'output/blurred_face.jpg'  # Path to save the blurred image
    output_path_mosaic = 'output/mosaiced_face.jpg'  # Path to save the mosaiced image

    # Censor eyes in the image
    censor_eyes(image_path, output_path_censor, 50, 50)

    # Apply Gaussian blur to the whole face in the image
    gaussian_blur_whole_face(image_path, output_path_blur, sigma=15)

    # Apply mosaic effect to the whole face in the image
    mosaic_whole_face(image_path, output_path_mosaic, mosaic_size=10)

    # Print the censored image
    print_image_path(output_path_censor)