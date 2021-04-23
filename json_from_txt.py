  
"""
input a number of image_{img_counter}.txt in format:
str("text from labelling") with encoding = ANSI

output a number of image_{img_counter}.json with encoding = 'UTF-8' in format:
{"shapes": [{"label": "text from labelling", "points": [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                                                        [x_down_right, y_down_right], [x_down_left, y_down_left]]}]}
"""
import json
import os

x_top_left, y_top_left = 0, 0
x_top_right, y_top_right = 10, 0
x_down_right, y_down_right = 10, 10
x_down_left, y_down_left = 0, 10

txt_folder = r'C:\Users\sWX1017677\Desktop\полосы\output'

f1 = next(os.walk(txt_folder))
txt_names = f1[2]
for i in txt_names:
    print(i)
    with open(os.path.join(txt_folder, i), 'r', encoding = 'ANSI') as txt_file:
        stripe_text = txt_file.readline()
        print('stripe_text = ', stripe_text)

        with open('s/{}.json'.format(i[:-4]), 'w', encoding = 'UTF-8') as outfile:
            data = {}
            data['shapes'] = []
            data['shapes'].append({
                'label': stripe_text[:-1],
                'points': [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                        [x_down_right, y_down_right], [x_down_left, y_down_left]]
            })
            json.dump(data, outfile, ensure_ascii=False)