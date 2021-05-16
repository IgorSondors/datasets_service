"""
warping strides from csv in format:
name_image_(jpeg), height_image, width_image, number_of_polygons, number_points_in_poligon (for example poligon with 4 points - two pairs - totally 4 points),
                                                                  x_coord_low(of pair), y_coord_low(of pair), x_coord_up(of pair), y_coord_up(of pair)...
"""
import numpy as np
import cv2
import math
import scipy
import scipy.interpolate
import time

start_time = time.time()

def draw_points(im, poligon_counter, bottom_x, bottom_y, top_x, top_y):
    # Radius of circle
    radius = 1
    thickness = 2
    img = im.copy()
    for i in range(len(bottom_x)):
        x, y = bottom_x[i], bottom_y[i]
        center_coordinates = (int(x), int(y))
        color = (0, 0, 255)
        img = cv2.circle(img, center_coordinates, radius, color, thickness)
    
    for i in range(len(top_y)):
        x, y = top_x[i], top_y[i] 
        center_coordinates = (int(x), int(y))
        color = (255, 0, 0)
        img = cv2.circle(img, center_coordinates, radius, color, thickness)

    cv2.imwrite('./draw/{}.jpg'.format(poligon_counter), img)
    return #im

def warp_image(img, src, dst, width, height):
    grid_x, grid_y = np.mgrid[0:height, 0:width]
    #grid_z = griddata(dst, src, (grid_x, grid_y), method='cubic')
    grid_z = scipy.interpolate.griddata(dst, src, (grid_x, grid_y), method='linear')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(height, width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    #warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)
    return warped

def crop_areas(opencv_img, poligons, poligon_height):
    #print('crop_areas')
    #print('x', poligons[0][0],poligons[0][-1])
    #print('y', poligons[1][0],poligons[1][-1])
    boarder = 192
    src = []
    dst = []
    dist = 0
    stripe_height = 32
    dist = 0.0
    dist_cur = 0
    hor_scale = float(stripe_height/poligon_height)
    # xu - x upper, xl - x lower
    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

        src.append([poligons[3][i], poligons[2][i]])
        dst.append([0, dist_cur*hor_scale*2])

    dist = 0.0
    dist_cur = 0
    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

    
        src.append([poligons[1][i], poligons[0][i]])
        dst.append([stripe_height, dist_cur*hor_scale*2])

    src_arr = np.array(src)
    dst_arr = np.array(dst)
    
    warped = warp_image(opencv_img, src_arr, dst_arr, int(dst[len(poligons[0]) - 1][1]), 32)
    warped = cv2.copyMakeBorder( warped, top=0, bottom=0, left=boarder, right=boarder, borderType=cv2.BORDER_CONSTANT )
    
    return warped

def enlarge_coord(bottom_x, bottom_y, is_bottom, mid_arithmetic_h):
    new_bottom_x, new_bottom_y, new_mid_arithmetic_h = [], [], []
    coord_source = []
    for i in range(len(bottom_x) - 1):
        coord_source.append([bottom_x[i], bottom_y[i], mid_arithmetic_h, bottom_x[i + 1], bottom_y[i + 1]])
    for s in coord_source:
        x = s[3] - s[0]
        y = s[1] - s[4]
        sign = 0
        if y < 0.0:
            sign = 1
            y = -y
        ugol_a = 180.0 - 90.0 - math.atan(y/x)*(180.0/math.pi)
        ugol_a_rad = ugol_a * (math.pi/180.0)
        y_ = math.sin(ugol_a_rad)*s[2]
        x_ = math.cos(ugol_a_rad)*s[2]
        if sign == 1:
            x_ = -x_
        #enlarge ceil and floor
        enlarge_c = 0.45#uniform(0.1, 0.7)
        d_x_c = x_*enlarge_c
        d_y_c = y_*enlarge_c
        enlarge_f = 0.45#uniform(0.1, 0.7)
        d_x_f = x_*enlarge_f
        d_y_f = y_*enlarge_f

        if is_bottom == True:
            new_bottom_x.append(s[0] + d_x_f)
            new_bottom_y.append(s[1] + d_y_f)
        else: # top
            new_bottom_x.append(s[0]  - d_x_c)
            new_bottom_y.append(s[1]  - d_y_c)
             
    last_dot = coord_source[-1]
    if is_bottom == True:
        new_bottom_x.append(last_dot[3] + d_x_f)
        new_bottom_y.append(last_dot[4] + d_y_f)
    else: # top
        new_bottom_x.append(last_dot[3]  - d_x_c)
        new_bottom_y.append(last_dot[4]  - d_y_c)
 
    return new_bottom_x, new_bottom_y, enlarge_c, enlarge_f

def parse_coord(string):
    bottom_x, bottom_y, top_x, top_y = [], [], [], []
    number_of_polygons = int(string[3])
    
    string = string[4:]

    all_dots_num_list = string[:number_of_polygons]

    string = string[number_of_polygons:]

    shift = 0
    for j in all_dots_num_list:
        bottom_x_one_poligon, bottom_y_one_poligon, top_x_one_poligon, top_y_one_poligon = [], [], [], []
        for i in range(int(j)//2):
            x = int(float(string[shift + i*4]))
            y = int(float(string[shift + i*4 + 1]))
            bottom_x_one_poligon.append(x)
            bottom_y_one_poligon.append(y)
            
            if x == int(float( string[shift + 2*int(j) - 4] )): 
                bottom_x.append(bottom_x_one_poligon)
                bottom_y.append(bottom_y_one_poligon)   
            
            x_top = int(float(string[2 + shift + i*4]))
            y_top = int(float(string[2 + shift + i*4 + 1]))  
            top_x_one_poligon.append(x_top)
            top_y_one_poligon.append(y_top)


            if x_top == int(float( string[shift + 2*int(j) - 2] )): 
                top_x.append(top_x_one_poligon)
                top_y.append(top_y_one_poligon)

        shift = shift + 2*int(j)
    return bottom_x, bottom_y, top_x, top_y

def poligon_height(bottom_x, bottom_y, top_x, top_y):
    bbox_height_sum = 0
    dots_counter = len(bottom_x)
    for i in range(dots_counter):
        bbox_height_sum = bbox_height_sum + ((top_x[i] - bottom_x[i])**2+(top_y[i] - bottom_y[i])**2)**0.5
    mid_arithmetic_h = bbox_height_sum / dots_counter

    return mid_arithmetic_h

with open('ds_images.csv', encoding = 'ANSI') as fp:
    line = fp.readline()
    poligon_counter = -1
    while line:
        new_line = line[:len(line)-1]
        strings = new_line.split(',')
        print( strings[0] )
        im = cv2.imread('./real_frames/' + str(strings[0]))
        file_name = strings[0][:-4]

        all_poligons_bottom_x, all_poligons_bottom_y, all_poligons_top_x, all_poligons_top_y = parse_coord(strings)

        for i in range(len(all_poligons_bottom_x)):
            #try:
            bottom_x, bottom_y, top_x, top_y = all_poligons_bottom_x[i], all_poligons_bottom_y[i], all_poligons_top_x[i], all_poligons_top_y[i]
        
            mid_arithmetic_h = poligon_height(bottom_x, bottom_y, top_x, top_y)

            #draw_points(im, poligon_counter, bottom_x, bottom_y, top_x, top_y)

            is_bottom = True
            bottom_x, bottom_y, enlarge_c, enlarge_f = enlarge_coord(bottom_x, bottom_y, is_bottom, mid_arithmetic_h)

            is_bottom = False               
            top_x, top_y, enlarge_c, enlarge_f = enlarge_coord(top_x, top_y, is_bottom, mid_arithmetic_h)

            #draw_points(im, poligon_counter, bottom_x, bottom_y, top_x, top_y)

            enlarge_h = mid_arithmetic_h * (1+ enlarge_c + enlarge_f)  

            poligons = [bottom_x, bottom_y, top_x, top_y]

            poligon_counter = poligon_counter + 1
            
            warped = crop_areas(im, poligons, enlarge_h)
            cv2.imwrite('./stripes/{}_{}.jpg'.format(file_name, poligon_counter), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  
            #except:
            #        print('Error! Skip this poligon')
            #        #continue
        line = fp.readline()

print('end_time = ', time.time() - start_time)
