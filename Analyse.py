import re
from jinja2 import Template
import shutil
from selenium import webdriver


# Constants
MTEs = {
    'Slow walk': 3.3,
    'Fast walk': 4.5,
    'Slow bike': 8,
    'Moderate bike': 10,
    'Fast bike': 12,
    'Basketball': 8,
    'Bodyweight exercises': 8,
    'Slow swim': 8.3,
    'Moderate swim': 10,
    'Jump rope': 11,
    'Slow run': 12.3,
}

PERSIAN_DIC = {
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '/': '.'
    }

def Find_certificate_number(bbox, text, img_x, img_y, output_path, file_name):
    # find bbox that is placed in the bottom left corner of the image
    (top_left, top_right, bottom_right, bottom_left) = bbox
    if top_left[0] < img_x / 3 and bottom_right[1] > img_y * 2 / 3 and len(text) >= 8 and '/' in text:
        MakeCertificateFile(text, output_path, file_name)

def MakeCertificateFile(text, output_path, file_name):
    PERSIAN_DIC.pop('/', None)
    for key in PERSIAN_DIC:
        text = text.replace(key, PERSIAN_DIC[key])

    url = f'http://fdacrm.ir/show?pavanenum={text}'
    
    # create a web driver instance
    driver = webdriver.Chrome()

    # navigate to the website
    driver.get(url)

    # set window size
    driver.set_window_size(1920, 1080)

    # save a screenshot of the current page
    driver.save_screenshot(output_path + file_name + '_certificate.png')

    # zclose the web driver
    driver.quit()
    return

def isMatched(text, match_case):
    arr = text.split(' ')
    for word in arr:
        edit_distance = minDistance(word, match_case)
        Normalized_edit_distance = edit_distance / len(match_case)
        if Normalized_edit_distance < 0.4:
            return True
    
    return False
        
def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)
    table = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        table[i][0] = i
    for j in range(n + 1):
        table[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                table[i][j] = table[i - 1][j - 1]
            else:
                table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
    return table[-1][-1]

def SearchForCalories(result, energy_mid_y):
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        if top_left[1] < energy_mid_y < bottom_right[1]:
            return ExtractNumber(text)


def ExtractNumber(text):

    for key in PERSIAN_DIC:
        text = text.replace(key, PERSIAN_DIC[key])

    text = re.findall(r'\d+\.\d+|\d+', text)

    return float(text[0]) if text else 0

def Analyse(result, energy_mid_y, body_weight, output_path, file_name):
    earned_calories = SearchForCalories(result, energy_mid_y)
    if earned_calories != 0:
        print(earned_calories)
        burned_calories_per_minute = 0
        time_to_burn_calories_per_exercise = {}

        for key, value in MTEs.items():
            burned_calories_per_minute = 3.5 * body_weight * value / 200

            time_to_burn_calories_per_exercise[key] = round(earned_calories / burned_calories_per_minute)

        shutil.copy('Template.html', output_path + file_name + '_analyse.html')
        with open('Template.html', 'r', encoding="utf-8") as Template_file:
            with open(output_path + file_name + '_analyse.html', 'w', encoding="utf-8") as file:
                template = Template(Template_file.read())

                html_output = template.render(data=time_to_burn_calories_per_exercise, calories=earned_calories)

                file.write(html_output)
        
