import json
import re
import cv2
import random
import csv
import os
import shutil


# make folders for CRAFT
if not os.path.exists('./Data/CRAFT'):
    os.makedirs('./Data/CRAFT')
if not os.path.exists('./Data/CRAFT/data_root_dir'):
    os.makedirs('./Data/CRAFT/data_root_dir')

training_img_path_CRAFT = './Data/CRAFT/data_root_dir/ch4_training_images/'
training_gt_path_CRAFT = './Data/CRAFT/data_root_dir/ch4_training_localization_transcription_gt/'
test_img_path_CRAFT = './Data/CRAFT/data_root_dir/ch4_test_images/'
test_gt_path_CRAFT = './Data/CRAFT/data_root_dir/ch4_test_localization_transcription_gt/'
current_imag_path = './Data/Cropped/'

# make dir
if not os.path.exists(training_img_path_CRAFT):
    os.makedirs(training_img_path_CRAFT)
if not os.path.exists(training_gt_path_CRAFT): 
    os.makedirs(training_gt_path_CRAFT)
if not os.path.exists(test_img_path_CRAFT):
    os.makedirs(test_img_path_CRAFT)
if not os.path.exists(test_gt_path_CRAFT):
    os.makedirs(test_gt_path_CRAFT)

# make folders for ocr
if not os.path.exists('./Data/Ocr'):
    os.makedirs('./Data/Ocr')
if not os.path.exists('./Data/Ocr/all_data'):
    os.makedirs('./Data/Ocr/all_data')

training_img_path_OCR = './Data/Ocr/all_data/fa_train_filtered/'
test_img_path_OCR = './Data/Ocr/all_data/fa_val/'

if not os.path.exists(training_img_path_OCR):
    os.makedirs(training_img_path_OCR)
if not os.path.exists(test_img_path_OCR):
    os.makedirs(test_img_path_OCR)

with open(training_img_path_OCR + 'labels.csv', mode='w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'words'])

with open(test_img_path_OCR + 'labels.csv', mode='w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'words'])

# read json file
with open('./Data/label_detail_min.json',  encoding="utf8") as json_file:
    data = json.load(json_file)
random.shuffle(data)

train_split = 0.8

# extract data
for i in range(0, len(data)):
    Craft_list = []
    bboxes = data[i]['bbox']
    transcriptions = data[i]['transcription']
    filename = re.split('/ |-', data[i]['ocr'])[-1].replace('.jpg', '.txt')
    title = ""

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
        
        Craft_list.append(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')
        
        # Crop the image for Ocr
        img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
        crop_img = img[y1:y3, x1:x3]
        cropped_filename = filename.replace('.txt', f'_{j}.jpg')

        if random.random() < train_split:
            img_path = training_img_path_OCR
        else:
            img_path = test_img_path_OCR

        with open(img_path + 'labels.csv', mode='a', newline='', encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow([cropped_filename, transcript])
            cv2.imwrite(img_path + f'{cropped_filename}', crop_img)
    
    # save data for CRAFT
    txt = 'gt_' + filename
    img = filename.replace('.txt', '.jpg')

    if random.random() < train_split:
        shutil.copyfile(current_imag_path + img, training_img_path_CRAFT + img)
        with open(training_gt_path_CRAFT + txt, 'a', encoding="utf-8") as f:
            f.writelines(Craft_list)
    else:
        shutil.copyfile(current_imag_path + img, test_img_path_CRAFT + img)
        with open(test_gt_path_CRAFT + txt, 'a', encoding="utf-8") as f:
            f.writelines(Craft_list)

