import easyocr
from easyocr.detection import get_detector, get_textbox
import torch
import cv2
import os
import random
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from Analyse import isMatched, Analyse, Find_certificate_number
from ultralytics import YOLO

label_detection_model = YOLO("Models/Label_detection.pt") 

# Load the custom CRAFT model
craft_save_pth = torch.load('Models/CRAFT_clr_amp_11000.pth', map_location='cpu')
craft_model = craft_save_pth["craft"]

# Save the models separately for the detector and recognizer
torch.save(craft_model, "Models/CRAFT_detector.pth")

# Initialize EasyOCR reader without loading default models
reader = easyocr.Reader(
    lang_list=["fa"],
    detector=False,
    user_network_directory='./Models',
    model_storage_directory='./Models',
    recog_network='Ocr_best'
)

# Correctly assign custom models to reader
reader.get_detector = get_detector
reader.get_textbox = get_textbox
reader.detector = reader.initDetector("Models/CRAFT_detector.pth")

# Specify the image path
img_path = "./Data/temp/test"
output_path = "Data/temp/"

# Get list of images
img_lst = os.listdir(img_path)
random.shuffle(img_lst)

# Process each image
for i in img_lst:
    img_file = os.path.join(img_path, i)
    label_detection_model = YOLO("Models/Label_detection.pt")

    # Perform object detection
    yolo_results = label_detection_model(img_file)
    
    for yolo_result in yolo_results:

        # Get the bounding box and label of the detected object
        bbox = yolo_result.boxes.xyxy[0].cpu().numpy()

        # read and crop the image based on the detected object
        image = cv2.imread(img_file)
        image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        result = reader.readtext(image)
        
        energy_mid_y = -1
        
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 5)

            if text[0] < 'z' and text[0] > 'A' or text[0] >= '0' and text[0] <= '9':
                cv2.putText(image, text, (int(top_left[0]), int(top_left[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                # put persian text on image
                reshaped_text = arabic_reshaper.reshape(text) # correct its shape
                bidi_text = get_display(reshaped_text) # correct its direction
                img_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype('./Data/Fonts/BYekan.ttf', 30)
                draw.text((int(top_left[0]), int(top_left[1]) - 15), bidi_text, font=font, fill=(255, 0, 0))
                image = np.array(img_pil)

                if isMatched(text, 'انرژی'):
                    energy_mid_y = (top_left[1] + bottom_right[1]) / 2
                
                file_name = i.replace('.jpg', '')
                Find_certificate_number(bbox, text, image.shape[1], image.shape[0], output_path, file_name)
                
        if energy_mid_y != -1:
            file_name = i.replace('.jpg', '')
            Analyse(result, energy_mid_y, output_path, file_name)
        
        # Write the annotated image to the output directory
        output_file = os.path.join(output_path, i)
        cv2.imwrite(output_file, image)

        # # write the result to a text file
        # with open('Data/temp/' + i + '.txt', 'w', encoding="utf-8") as f:
        #     for (bbox, text, prob) in result:
        #         (top_left, top_right, bottom_right, bottom_left) = bbox
        #         f.write(f'{top_left[0]},{top_left[1]},{top_right[0]},{top_right[1]},{bottom_right[0]},{bottom_right[1]},{bottom_left[0]},{bottom_left[1]},{text}\n')
