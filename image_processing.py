from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os
import rembg
import numpy as np

def tensor_to_image(image_vector):
    # Reshape the flattened tensor (4096,) back to (64, 64)
    image_vector = image_vector * 0.5 + 0.5
    image_tensor = image_vector.view(128, 128)

    # Convert to NumPy array
    image_array = image_tensor.numpy()

    # Matplotlib expects pixel values in the range [0,1] or [0,255]
    plt.imshow(image_array, cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()





# Function to load and process an image
def process_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale

    # Define the transformation (resize, convert to tensor, and normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 64x64 pixels
        transforms.ToTensor(),        # Convert image to tensor (values between 0 and 1)
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize: (x-0.5)/0.5 gives range [-1, 1]
    ])

    # Apply the transformation
    image_tensor = transform(image)

    # Flatten the image (64x64 becomes a vector of size 4096)
    return image_tensor[0].view(-1)


def remove_background(image_path):
    input_image = Image.open(image_path)
    input_array = np.array(input_image)

    output_array = rembg.remove(input_array)
    output_image = Image.fromarray(output_array)
    output_path = image_path.replace('training_faces', 'training_faces_no_bg')  # Replace only the first occurrence
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_image.save(output_path)



# Example usage
# image_path = 'test1.png'
# image_vector = process_image(image_path)

# tensor_to_image(image_vector)
# print(image_vector)  # This should be torch.Size([4096]) for 64x64 images

