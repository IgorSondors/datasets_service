  
"""
input csv in format:
image_{img_counter}_{word_counter}.jpg, 32, stride_width, dots_number, X_center, char_value, char_code,
                                                                       X_center, char_value, char_code etc
output a number of image_{img_counter}_{word_counter}.json in format:
{"shapes": [{"label": "text from labelling", "points": [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                                                        [x_down_right, y_down_right], [x_down_left, y_down_left]]}]}
"""
import json

def create_descriptor():
    with open('ds_base_without_real.csv', encoding="cp1251") as fp:
        desc = []
        for line in fp:
            new_line = line[:len(line)-1]
            strings = new_line.split(',')
            string_new = []
            #replace repeated commas to one comma
            for s in range(len(strings) - 1):
                if strings[s] == '' and strings[s + 1] == '':
                    string_new.append(',')
                elif strings[s] != '':
                    string_new.append(strings[s])
            string_new.append(strings[-1])
            desc.append(string_new)
    return desc

list_of_csv_lines = create_descriptor()
print('len(list_of_csv_lines) = ', len(list_of_csv_lines))

json_num = 0
for line in list_of_csv_lines:
    stripe_text = ''
    for i in range(int(line[3])):
        #print(i)
        char = str(line[5 + i*3])
        #print(char)
        if i > 0:
            if int(line[6 + i*3]) == 1 or int(line[6 + i*3]) == 3:
                char = ' ' + char 
        stripe_text = stripe_text + char
    #print('stripe_text = ', stripe_text)

    x_top_left, y_top_left = 0, 0
    x_top_right, y_top_right = 10, 0#int(strings[2]) - 192*2, 0
    x_down_right, y_down_right = 10, 10#int(strings[2]) - 192*2, int(strings[1])
    x_down_left, y_down_left = 0, 10#0, int(strings[1]) 

    with open('out_folder/{}.json'.format(str(line[0])[:-4]), 'w', encoding = 'UTF-8') as outfile:
        data = {}
        data['shapes'] = []
        data['shapes'].append({
            'label': stripe_text,
            'points': [[x_top_left, y_top_left], [x_top_right, y_top_right], 
                    [x_down_right, y_down_right], [x_down_left, y_down_left]]
        })
        json.dump(data, outfile, ensure_ascii=False)

    json_num = json_num + 1
    print(json_num)