import os
import re
import random
import shutil


txtpath = './Data/GT_localization_transcription'
imgpath = './Data/Cropped/'
txtnames = os.listdir(txtpath)

training_img_path = './Data/CRAFT/data_root_dir/ch4_training_images'
training_gt_path = './Data/CRAFT/data_root_dir/ch4_training_localization_transcription_gt'
test_img_path = './Data/CRAFT/data_root_dir/ch4_test_images'
test_gt_path = './Data/CRAFT/data_root_dir/ch4_test_localization_transcription_gt'

# make dir
if not os.path.exists(training_img_path):
    os.makedirs(training_img_path)
if not os.path.exists(training_gt_path): 
    os.makedirs(training_gt_path)
if not os.path.exists(test_img_path):
    os.makedirs(test_img_path)
if not os.path.exists(test_gt_path):
    os.makedirs(test_gt_path)

random.shuffle(txtnames)

for item in txtnames:
    name = item.replace('gt_', '').replace('.txt', '')
    img = name + '.jpg'

    # copy and move the image and the text file to the new directory 80 : 20
    if random.random() < 0.8:
        shutil.copy(os.path.join(imgpath, img), training_img_path)
        shutil.copy(os.path.join(txtpath, item), training_gt_path)
    else:
        shutil.copy(os.path.join(imgpath, img), test_img_path)
        shutil.copy(os.path.join(txtpath, item), test_gt_path)