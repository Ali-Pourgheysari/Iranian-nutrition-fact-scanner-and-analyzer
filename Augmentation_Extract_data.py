import json
import re
import cv2
import random
import csv
import os
import albumentations as A


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

# Declare an augmentation pipeline
transform = A.Compose([
    A.AdvancedBlur(blur_limit=(0, 59), sigma_x_limit=(0.9, 1.0), sigma_y_limit=(0.9, 1.0), sigmaX_limit=None, sigmaY_limit=None, rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), p=0.5),
    A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
    A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.2), p=0.4),
    A.Downscale(scale_min=None, scale_max=None, interpolation=None, scale_range=(0.25, 0.25), interpolation_pair={'upscale': 0, 'downscale': 0}, p=0.5),
    A.Emboss(alpha=(0, 1), strength=(0, 1),  p=0.5),
    A.GaussNoise(var_limit=(10.0, 500.0), mean=10, per_channel=True, noise_scale_factor=1, p=0.5),
    A.Rotate(limit=1, interpolation=1, border_mode=4, value=None, mask_value=None, p=1)
])

ocr_transform = A.Compose([
    A.Rotate(limit=10, p=0.5),  # Rotate image by a small angle
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), shear=0, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5),  # Apply a perspective transform
    A.MotionBlur(blur_limit=3, p=0.2),  # Apply motion blur
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Add Gaussian noise
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    # A.PadIfNeeded(min_height=128, min_width=128, p=1.0),
    A.CoarseDropout(max_holes=5, max_height=8, max_width=8, fill_value='random', p=0.2),
    # A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0, p=1.0),
])

# extract data
for i in range(0, len(data)):
    for aug in range(10):
        Craft_list = []
        bboxes = data[i]['bbox']
        transcriptions = data[i]['transcription']
        filename = re.split('/ |-', data[i]['ocr'])[-1].replace('.jpg', '.txt')
        img = cv2.imread('./Data/Cropped/' + re.split('-', data[i]['ocr'])[-1])
        try:
            # Augment an image
            transformed = transform(image=img)
            transformed_image = transformed["image"]
        except:
            aug -= 1 
            continue
        
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
            crop_img = img[y1:y3, x1:x3]
            if random.random() < 0.5:
                crop_img = ocr_transform(image=crop_img)['image']

            if combined_x1 != '':
                crop_comb_img = img[combined_y1:combined_y3, combined_x1:combined_x3]
            cropped_filename = filename.replace('.txt', f'_{j+aug}.jpg')

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
        txt = 'gt_' + filename.replace('.txt', f'_{aug}.txt')
        new_filename = filename.replace('.txt', f'_{aug}.jpg')

        if random.random() < train_split:
            cv2.imwrite(training_img_path_CRAFT + new_filename, transformed_image)
            with open(training_gt_path_CRAFT + txt, 'a', encoding="utf-8") as f:
                f.writelines(Craft_list)
        else:
            cv2.imwrite(test_img_path_CRAFT + new_filename, transformed_image)
            with open(test_gt_path_CRAFT + txt, 'a', encoding="utf-8") as f:
                f.writelines(Craft_list)

