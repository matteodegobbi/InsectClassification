import os
from PIL import Image

def remove_corrupted_images(directory):
    # Loop through all files and subfolders in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            try:
                # Attempt to open the image
                with Image.open(file_path) as img:
                    img.verify()  # Verify if it's an image
            except (IOError, SyntaxError) as e:
                # If the image is corrupted, an error will be raised
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)  # Delete the corrupted file

if __name__ == "__main__":
    folder_path = 'image_dataset/'  # Set your folder path here
    remove_corrupted_images(folder_path)

