"""
check csv in format:
name image (jpeg),height image, width image,number of points, 
x_coordinate of next point, y_coordinate of next point, width of next point, 
height of next point, value of symbol under which is next point, and so on
"""

import cv2

def draw_points(im, string):
    radius = 1
    color = (0,255,0)
    thikness = 1

    for i in range(int(strings[3])):
        
        x = int(float(strings[4 + i*5 + 0]))
        y = int(float(strings[4 + i*5 + 1]))
        
        center_coord = (int(x), int(y))
        im = cv2.circle(im, center_coord, radius, color, thikness)

    return im

with open('real_dataset/ds_images.csv', encoding = 'ANSI') as fp:
    line = fp.readline()
    cnt = 1
    while line:
        new_line = line[:len(line)-1]
        strings = new_line.split(',')
        print(strings[3])
        print( strings[0])
        im = cv2.imread('./real_dataset/real_frames/' + str(strings[0]))
        im = draw_points(im, strings)

        cv2.imwrite('./result/' + strings[0], im)

        line = fp.readline()
        cnt += 1
