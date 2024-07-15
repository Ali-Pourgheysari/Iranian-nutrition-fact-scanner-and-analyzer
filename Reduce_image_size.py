from PIL import Image, ImageOps
import os

imgs_path = './Data/Detection test'
imgs_path_resized = './Data/temp'

width = 640

for img_name in os.listdir(imgs_path):
    img_path = os.path.join(imgs_path, img_name)
    
    # Open the image
    img = Image.open(img_path)
    
    # Apply EXIF transpose to correct orientation
    img = ImageOps.exif_transpose(img)
    
    # Resize the image
    img_resized = img.resize((width, width))
    
    # Save the resized image
    img_resized.save(os.path.join(imgs_path_resized, img_name))
