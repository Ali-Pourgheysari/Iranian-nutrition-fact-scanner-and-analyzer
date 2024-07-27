import json


# read json file
with open('/Data/label_detail_min') as json_file:
    data = json.load(json_file)

# extract data
for i in range(0, len(data)):
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    file_name = data[i]['ocr'].split('/')[-1].replace('.jpg', '.txt')

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = bboxes[j]['x'] * original_width / 100, bboxes[j]['y'] * original_height / 100
        x2, y2 = (bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100, (bboxes[j]['y']) * original_height / 100
        x3, y3 = (bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100, (bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100
        x4, y4 = (bboxes[j]['x']) * original_width / 100, (bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100

        transcript = transcriptions[j]

        # Create a file and write to it
        with open(f'/Data/CRAFT/gt_{file_name}', 'a') as file:
            file.write(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')
        