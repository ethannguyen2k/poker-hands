import os
from PIL import Image, ImageOps

# Folders for source images and output
source_folder = "D:/GitHub_Web/poker-hands/templates"
output_folder = "D:/GitHub_Web/poker-hands/templates/output"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# List all image files in the source folder
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Process each image
for image_file in image_files:
    # Load the image
    image_path = os.path.join(source_folder, image_file)
    image = Image.open(image_path)

    # Invert the colors
    inverted_image = ImageOps.invert(image.convert("RGB"))

    # Save the inverted image to the output folder
    output_path = os.path.join(output_folder, image_file)
    inverted_image.save(output_path)

    print(f"Processed and saved: {image_file}")

print("All images processed successfully.")
