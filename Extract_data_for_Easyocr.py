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

    for j in range(0, len(bboxes)):
        original_width = bboxes[j]['original_width']
        original_height = bboxes[j]['original_height']

        x1, y1 = int(bboxes[j]['x'] * original_width / 100), int(bboxes[j]['y'] * original_height / 100)
        x2, y2 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y']) * original_height / 100)
        x3, y3 = int((bboxes[j]['x'] + bboxes[j]['width']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)
        x4, y4 = int((bboxes[j]['x']) * original_width / 100), int((bboxes[j]['y'] + bboxes[j]['height']) * original_height / 100)

        transcript = transcriptions[j]
        
        # create combined bbox for OCR including scale + number
        combined_transcript = ''
        combined_x1, combined_y1 = '', ''
        if len(transcriptions) <= 18:
            for k, label in enumerate(data[i]['label']):
                if k == j and 'Number' in label['labels']:
                    for l, label2 in enumerate(data[i]['label']):
                        if 'Scale' in label2['labels']:
                            scale_y1 = int(label2['y'] * original_height / 100)
                            scale_y3 = int((label2['y'] + label2['height']) * original_height / 100)
                            mid = (y1 + y3) / 2

                            if mid > scale_y1 and mid < scale_y3:
                                scale_x1 = int(label2['x'] * original_width / 100)
                                scale_x3 = int((label2['x'] + label2['width']) * original_width / 100)
                                # generalize bbox 
                                if x1 > scale_x1:
                                    combined_x1 = scale_x1
                                    combined_transcript = transcriptions[l] + ' ' + transcript
                                else:
                                    combined_x1 = x1
                                    combined_transcript = transcript + ' ' + transcriptions[l]

                                if x3 < scale_x3:
                                    combined_x3 = scale_x3
                                    combined_transcript = transcript + ' ' + transcriptions[l]
                                else:
                                    combined_x3 = x3
                                    combined_transcript = transcriptions[l] + ' ' + transcript
                                combined_y1 = min(y1, scale_y1)
                                combined_y3 = max(y3, scale_y3)

        Craft_list.append(f'{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{transcript}\n')
        
        # Crop the image for Ocr
        img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
        crop_img = img[y1:y3, x1:x3]
        if combined_x1 != '':
            crop_comb_img = img[combined_y1:combined_y3, combined_x1:combined_x3]
        cropped_filename = filename.replace('.txt', f'_{j}.jpg')

        if random.random() < train_split:
            img_path = training_img_path_OCR
        else:
            img_path = test_img_path_OCR

        with open(img_path + 'labels.csv', mode='a', newline='', encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow([cropped_filename, transcript])
            if combined_x1 != '':
                writer.writerow([cropped_filename.replace('.jpg', '_comb.jpg'), combined_transcript])
            cv2.imwrite(img_path + f'{cropped_filename}', crop_img)
            if combined_x1 != '':
                cv2.imwrite(img_path + f'{cropped_filename.replace(".jpg", "_comb.jpg")}', crop_comb_img)
    
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

