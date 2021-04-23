"""
csv in format:
image_{stride_counter}.jpg, 32, stride_width, dots_number, X_center, char_value, char_code,
                                                                       X_center, char_value, char_code etc
"""
import lmdb
import os

import numpy as np
import cv2
import six
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import defaultdict

def get_char_code(char_value, is_char_first):
    '''
    Биты:
    Относятся только к первой букве слова
    0 - 1 если буква первая в слове
    2 - 1 если перевернуто ли слово()
    3 - 1 если слово русское
    4 - 1 если слово английское
    Для любого положения в слове буквенного символа
    1 - 1 если заглавная буква
    Все кроме букв: только бит по первому символу
    Слово "Привет"
    Буква "П":
    2**0 (первая буква) + 2**1 (заглавная буква) + 2**3 (русское слово) = 11
    Буква "р":
    0 (тк не заглавная буква) и тд
    '''

    short_ch = 'абвгеёжзийклмнопстхчшъыьэюя'
    long_ch = 'друфцщ'
    capital_ch = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    numbers_ch = '0123456789'
    char_code = 0
    if is_char_first:
        char_code = char_code + 2**0
    if char_value in capital_ch:
        char_code = char_code + 2**1
    return char_code

path = r'C:\Users\sWX1017677\Desktop\jira\Novgorod\rus_300k_lines_for_AA'
path_to_save = r'C:\Users\sWX1017677\Desktop\jira\Novgorod\ocr_strides'

path_to_save_imgs = os.path.join(path_to_save, 'real_frames')
os.makedirs(path_to_save_imgs, exist_ok=True)

num_crops_to_save = 300000#200

env = lmdb.open(path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)


'''
"char_ann-..." - строка, представляющая собой список (разделитель ";;"), 
каждый элемент которого - буква и 8 цифр, разделенные ",". Цифры - x,y координаты 4 точек четырехугольника вокруг буквы.
Например: "д,1,2,3,4,5,6,7,8;;е,10,11,12,13,14,15,16,17;;..."
'''
with open('ocr_strides/ds_images.csv', 'w', encoding = 'cp1251') as recognizer_label:
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        print('Total number of symbols:', nSamples)
        for index in tqdm(range(1, nSamples)):
            if index > num_crops_to_save:
                break

            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')

            char_ann_key = 'chars_ann-%09d'.encode() % index
            char_ann = txn.get(char_ann_key).decode('utf-8')

            #print(repr(label))
            #print(repr(char_ann))

            image_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(image_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)        

            img = Image.open(buf).convert("L")
            w, h = img.size
            h_coeff = 32 / h
            w = int(w * h_coeff)
            img = img.resize((w*2, 32))
            img = ImageOps.expand(img, border=(192, 0, 192, 0), fill=0)

            out_img_filename = os.path.join(path_to_save_imgs, f"image_{index}.jpg")
            gray_image = ImageOps.grayscale(img)
            gray_image.save(out_img_filename)

            w_new, h_new = img.size
            recognizer_label.write(f"image_{index}.jpg" + ',' + str(h_new) + ',' + str(w_new))
            
            string_without_spaces = label.replace(' ', '')
            splitted_by_char = char_ann.split(';;')

            dots_num = len(string_without_spaces)
            recognizer_label.write(',' + str(dots_num))

            char_position = -1
            is_char_first = True
            for i in splitted_by_char:
                if i[0] == ' ':
                    is_char_first = True
                    continue
                char_value = i[0]
                char_code = get_char_code(char_value, is_char_first)

                annotation_of_one_char = i[2:].split(',')
                mid_x_coord = 192 + int(((float(annotation_of_one_char[0]) + float(annotation_of_one_char[2]) + float(annotation_of_one_char[4]) + float(annotation_of_one_char[6])) * h_coeff * 2)//4)
           
                recognizer_label.write(',' + str(mid_x_coord) + ',' + char_value + ',' + str(char_code))
                is_char_first = False
                #char_position = char_position + 1
                #print('char_position =', char_position)
                #print(i, i[0])
                #print(annotation_of_one_char)
            recognizer_label.write('\n')