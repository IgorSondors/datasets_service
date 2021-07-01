"""
csv in format:
image_{img_counter}_{word_counter}.jpg, 32, stride_width, dots_number, X_center, char_value, char_code,
                                                                       X_center, char_value, char_code etc
"""
import numpy as np
import cv2
import json
import ast
import time

def replace_bad_char(list_x, text, ch_code):
    special_chars = ' 0123456789,°!$%&\/"()*+-_±.:;<=§>?@[]^{}~№µ«»®#€©' + "'"
    chars_rus = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
    chars_eng = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    double_quotes_for_change = '”“〟〞„'
    #quotes_for_change = ""

    text = text.replace('≈', '~')
    text = text.replace(' ', '')
    index_for_remove = []

    for i in range(len(text)):
        if text[i] not in special_chars+chars_rus+chars_eng:
            if text[i] in double_quotes_for_change:
                text = text[0: i:] + '"' + text[i + 1::]
            #elif text[i] in quotes_for_change:
            #    text = text[0: i:] + "'" + text[i + 1::]
            else:
                index_for_remove.append(i)
    index_for_remove.sort(reverse=True)

    for i in index_for_remove:
        text = text[0: i:] + text[i + 1::]
        del list_x[i]
        del ch_code[i]
    assert len(text) == len(list_x) == len(ch_code)
    return list_x, text, ch_code

def get_char_code(char_value, is_char_first):
    '''
    Биты:
    Относятся только к первой букве слова
    0 - если буква первая в слове
    2 - если перевернуто ли слово()
    3 - если слово русское
    4 - если слово английское
    Для любого положения в слове буквенного символа
    1 - если заглавная буква
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

    chars_eng = "abcdefghijklmnopqrstuvwxyz"
    capital_chars_eng = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers_ch = '0123456789'

    char_code = 0

    # first/not first
    if is_char_first:
        char_code = char_code + 2**0  
    # capital/not capital
    if char_value in (capital_ch + capital_chars_eng):
        char_code = char_code + 2**1 
    return char_code

def get_char_code_list(text):
    ch_code = []
    splitted_by_space = text.split(' ')

    for word in splitted_by_space:
        is_char_first = True
        for j in word:
            one_char_code = get_char_code(j, is_char_first)
            ch_code.append(one_char_code)
            is_char_first = False
    return ch_code
    
start_time = time.time()
with open('output.jsonl', encoding = 'utf8') as fp:
    lines = fp.readlines()
    data_annotation = open('./ds_images.csv', 'w')
    stripes_counter = 0
    for line in lines:
        js_line = json.loads(line)
        #print(js_line)

        second = js_line['result']
        if second['readable'] == 'true' and second['status'] == 'APPROVED':
            stripes_counter = stripes_counter + 1

            file_name = js_line['image'].split('/')
            file_name = file_name[-1].split('?')
            print(file_name[0])

            file_path = './stripes_original_names/' + file_name[0]
            file_path0 = file_path[:-4] + '_0.jpg'
            file_path1 = file_path[:-4] + '_1.jpg'
            file_path2 = file_path[:-4] + '_2.jpg'
            file_path3 = file_path[:-4] + '_3.jpg'
            
            im = cv2.imread(file_path)
            im0 = cv2.imread(file_path0)
            im1 = cv2.imread(file_path1)
            im2 = cv2.imread(file_path2)
            im3 = cv2.imread(file_path3)

            height, width = im.shape[:2]
            height0, width0 = im0.shape[:2]
            height1, width1 = im1.shape[:2]
            height2, width2 = im2.shape[:2]
            height3, width3 = im3.shape[:2]

            width_multiplier0 = (width0 - 192*2) / (width - 192*2)
            width_multiplier1 = (width1 - 192*2) / (width - 192*2)
            width_multiplier2 = (width2 - 192*2) / (width - 192*2)
            width_multiplier3 = (width3 - 192*2) / (width - 192*2)

            list_x = ast.literal_eval(second['markup'])
            text = second['text']

            ch_code = get_char_code_list(text)
            list_x, text, ch_code = replace_bad_char(list_x, text, ch_code)
            dots_number = len(list_x)

            data_annotation.write(str('our_01_' + file_name[0]) + ',' + str(height) + ',' + str(width) + ',' + str(dots_number))
            for i in range(dots_number):                  
                data_annotation.write(',' + str(round(list_x[i], 1)) + ',' + str(text[i]) + ',' + '{}'.format(ch_code[i]))                
            data_annotation.write('\n')  

            data_annotation.write(str('our_01_' + file_name[0][:-4] + '_0.jpg') + ',' + str(height0) + ',' + str(width0) + ',' + str(dots_number))
            for i in range(dots_number):                     
                data_annotation.write(',' + str(round((list_x[i] - 192)*width_multiplier0 + 192, 1)) + ',' + str(text[i]) + ',' + '{}'.format(ch_code[i]))                
            data_annotation.write('\n') 

            data_annotation.write(str('our_01_' + file_name[0][:-4] + '_1.jpg') + ',' + str(height1) + ',' + str(width1) + ',' + str(dots_number))
            for i in range(dots_number):                     
                data_annotation.write(',' + str(round((list_x[i] - 192)*width_multiplier1 + 192, 1)) + ',' + str(text[i]) + ',' + '{}'.format(ch_code[i]))                
            data_annotation.write('\n') 

            data_annotation.write(str('our_01_' + file_name[0][:-4] + '_2.jpg') + ',' + str(height2) + ',' + str(width2) + ',' + str(dots_number))
            for i in range(dots_number):                     
                data_annotation.write(',' + str(round((list_x[i] - 192)*width_multiplier2 + 192, 1)) + ',' + str(text[i]) + ',' + '{}'.format(ch_code[i]))                
            data_annotation.write('\n') 

            data_annotation.write(str('our_01_' + file_name[0][:-4] + '_3.jpg') + ',' + str(height3) + ',' + str(width3) + ',' + str(dots_number))
            for i in range(dots_number):   
                data_annotation.write(',' + str(round((list_x[i] - 192)*width_multiplier3 + 192, 1)) + ',' + str(text[i]) + ',' + '{}'.format(ch_code[i]))                
            data_annotation.write('\n') 
print(stripes_counter)
print('end_time = ', time.time() - start_time)