import json
import re
import cv2

# read json file
with open('./Data/label_detail_min.json',  encoding="utf8") as json_file:
    data = json.load(json_file)

# extract data
for i in range(0, len(data)):
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    file_name = re.split('/ |-', data[i]['ocr'])[-1].replace('.jpg', '.txt')
    title = ""
    lst = []

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = int(bboxes[j]['x'] * original_width / 100), int(bboxes[j]['y'] * original_height / 100)
        x2, y2 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y']) * original_height / 100)
        x3, y3 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)
        x4, y4 = int((bboxes[j]['x']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)

        transcript = transcriptions[j]
        
        for k, label in enumerate(data[i]['label']):
            if 'Title' in label['labels'] and title == "":
                title = transcriptions[k]
                x1_title, y1_title = int(label['x'] * original_width / 100), int(label['y'] * original_height / 100)
                x3_title, y3_title = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)
                break
        
        if title != "" and k != j:
            if x1 >= x1_title and y1 >= y1_title and x3 <= x3_title and y3 <= y3_title:
                continue                
        
        lst.append([x1, y1, x2, y2, x3, y3, x4, y4, transcript])

    # show image
    img = cv2.imread('./Data/Cropped/' + re.split('/ |-', data[i]['ocr'])[-1])
    for x1, y1, x2, y2, x3, y3, x4, y4, transcript in lst:
        cv2.rectangle(img, (x1, y1), (x3, y3), (0, 255, 0), 1)
    # set window's width
    cv2.imshow('image', img)
    cv2.waitKey(0)

        # # Create a file and write to it
        # with open(f'./Data/CRAFT/gt_{file_name}', 'a', encoding="utf8") as file:
        #     file.write(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')

        