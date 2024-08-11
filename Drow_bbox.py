import json
import re
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display


# read json file
with open('./Data/label_detail_min.json',  encoding="utf8") as json_file:
    data = json.load(json_file)

# extract data
for i in range(0, len(data)):
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    title = ""
    lst = []

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = int(bboxes[j]['x'] * original_width / 100), int(bboxes[j]['y'] * original_height / 100)
        x3, y3 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)

        transcript = transcriptions[j]
        # for item in transcript:
        #     if item in char_dic.keys():
        #         transcript = transcript.replace(item, char_dic[item])
             
        lst.append([x1, y1, x3, y3, transcript])

    # show image
    img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
    for x1, y1, x3, y3, transcript in lst:
        cv2.rectangle(img, (x1, y1), (x3, y3), (0, 255, 0), 1)

        if transcript[0] < 'z' and transcript[0] > 'A' or transcript[0] >= '0' and transcript[0] <= '9':
            cv2.putText(img, transcript, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            # put persian text on image
            reshaped_text = arabic_reshaper.reshape(transcript) # correct its shape
            bidi_text = get_display(reshaped_text) # correct its direction
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('./Data/Fonts/BYekan.ttf', 30)
            draw.text((x1, y1 - 15), bidi_text, font=font, fill=(255, 0, 0))
            img = np.array(img_pil)
        
    # write img
    cv2.imwrite(f'./Data/temp/{re.split("-", data[i]["ocr"])[-1]}', img)
