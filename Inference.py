#code to load custom craft model (from )
import easyocr
from easyocr.detection import get_detector, get_textbox
import torch
import cv2

save_pth= torch.load('CRAFT_1000.pth' , map_location='cpu')
model = save_pth["craft"]
torch.save(model , "CRAFT_detector.pth") 

reader = easyocr.Reader(
    lang_list=["fa"],
    detector=False
)
reader.get_detector, reader.get_textbox = get_detector, get_textbox
reader.detector = reader.initDetector("CRAFT_detector.pth")

import os

img_path = "Data/Cropped/"

img_lst = os.listdir(img_path)
import random
random.shuffle(img_lst)

for i in img_lst:
    result = reader.readtext(img_path + i)
    # print(result)

    image = cv2.imread(img_path + i)

    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        image = cv2.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 5)
    
    # write 
    cv2.imwrite(f'Data/temp/{i}', image)
