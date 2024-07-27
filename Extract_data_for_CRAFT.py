import json
import re

# read json file
with open('./Data/label_detail_min.json',  encoding="utf8") as json_file:
    data = json.load(json_file)

# extract data
for i in range(0, len(data)):
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    file_name = re.split('/ |-', data[i]['ocr'])[-1].replace('.jpg', '.txt')

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = int(bboxes[j]['x'] * original_width / 100), int(bboxes[j]['y'] * original_height / 100)
        x2, y2 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y']) * original_height / 100)
        x3, y3 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)
        x4, y4 = int((bboxes[j]['x']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)

        transcript = transcriptions[j]
        
        # Create a file and write to it
        with open(f'./Data/CRAFT/gt_{file_name}', 'a', encoding="utf8") as file:
            file.write(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')

        