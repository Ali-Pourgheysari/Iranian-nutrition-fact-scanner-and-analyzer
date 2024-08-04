import json
import re
import cv2
import random
import csv
import os
import shutil

# read json file
with open('./Data/label_detail_min.json',  encoding="utf8") as json_file:
    data = json.load(json_file)

Ocr_list = []
Craft_list = []

# extract data
for i in range(0, len(data)):
    Craft_sub_list = []
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    filename = re.split('/ |-', data[i]['ocr'])[-1].replace('.jpg', '.txt')
    title = ""
    # lst = []

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = int(bboxes[j]['x'] * original_width / 100), int(bboxes[j]['y'] * original_height / 100)
        x2, y2 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y']) * original_height / 100)
        x3, y3 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)
        x4, y4 = int((bboxes[j]['x']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)

        transcript = transcriptions[j]
        
        for k, label in enumerate(data[i]['label']):
            if  title != "":
                break
            if 'Title' in label['labels']:
                title = transcriptions[k]
                x1_title, y1_title = int(label['x'] * original_width / 100), int(label['y'] * original_height / 100)
                x3_title, y3_title = int((bboxes[k]['x'] + bboxes[k]['width']) * original_width / 100), int((bboxes[k]['y'] + bboxes[k]['height']) * original_height / 100)
                break
        
        if title != "":
            if x1 > x1_title and y1 > y1_title and x3 < x3_title and y3 < y3_title:
                continue                
        
        Craft_sub_list.append(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')
        
        # Crop the image for Ocr
        img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
        crop_img = img[y1:y3, x1:x3]
        cropped_filename = filename.replace('.txt', f'_{j}.jpg')
        # cv2.imwrite(f'./Data/Ocr/{cropped_filename}', crop_img)

        Ocr_list.append({'filename': cropped_filename, 'words': transcript, 'image': crop_img})
    
    Craft_list.append({'filename': filename, 'data': Craft_sub_list})
    #     lst.append([x1, y1, x2, y2, x3, y3, x4, y4, transcript])

    # # show image
    # img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
    # for x1, y1, x2, y2, x3, y3, x4, y4, transcript in lst:
    #     cv2.rectangle(img, (x1, y1), (x3, y3), (0, 255, 0), 1)

    # # show img
    # cv2.imshow('image', img)
    # cv2.waitKey(0)


# Split data into training and test 80:20
random.shuffle(Craft_list)
train_craft = Craft_list[:int(len(Craft_list)*0.8)]
test_craft = Craft_list[int(len(Craft_list)*0.8):]

random.shuffle(Ocr_list)
train_data = Ocr_list[:int(len(Ocr_list)*0.8)]
test_data = Ocr_list[int(len(Ocr_list)*0.8):]

# Write information of craft to txt file
if not os.path.exists('./Data/CRAFT/data_root_dir'):
    os.makedirs('./Data/CRAFT/data_root_dir')

training_img_path = './Data/CRAFT/data_root_dir/ch4_training_images/'
training_gt_path = './Data/CRAFT/data_root_dir/ch4_training_localization_transcription_gt/'
test_img_path = './Data/CRAFT/data_root_dir/ch4_test_images/'
test_gt_path = './Data/CRAFT/data_root_dir/ch4_test_localization_transcription_gt'
current_imag_path = './Data/Cropped/'

# make dir
if not os.path.exists(training_img_path):
    os.makedirs(training_img_path)
if not os.path.exists(training_gt_path): 
    os.makedirs(training_gt_path)
if not os.path.exists(test_img_path):
    os.makedirs(test_img_path)
if not os.path.exists(test_gt_path):
    os.makedirs(test_gt_path)


for item in train_craft:
    txt = 'gt_' + item['filename']
    img = item['filename'].replace('.txt', '.jpg')

    if item in train_craft:
        shutil.copyfile(current_imag_path + img, training_img_path + img)
        with open(training_gt_path + txt, 'w', encoding="utf-8") as f:
            f.writelines(item['data'])

    else:
        shutil.copyfile(current_imag_path + img, test_img_path + img)
        with open(test_gt_path + txt, 'w', encoding="utf-8") as f:
            f.writelines(item['data'])


if not os.path.exists('./Data/Ocr/all_data/fa_train_filtered'):
    os.makedirs('./Data/Ocr/all_data/fa_train_filtered')
if not os.path.exists('./Data/Ocr/all_data/fa_val'):
    os.makedirs('./Data/Ocr/all_data/fa_val')


# Write information of ocr to csv file
with open('./Data/Ocr/all_data/fa_train_filtered/labels.csv', mode='w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'words'])
    for data in train_data:
        writer.writerow([data['filename'], data['words']])
        cv2.imwrite(f'./Data/Ocr/all_data/fa_train_filtered/{data["filename"]}', data['image'])

with open('./Data/Ocr/all_data/fa_val/labels.csv', mode='w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'words'])
    for data in test_data:
        writer.writerow([data['filename'], data['words']])
        cv2.imwrite(f'./Data/Ocr/all_data/fa_val/{data["filename"]}', data['image'])

