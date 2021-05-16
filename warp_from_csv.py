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
import random

def check_oX_shift(warped, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h, poligon_counter):
    enlarge_c_list = [0.5, 0.45, 0.1, 0.8, 0.8, 1, 0, 1]
    enlarge_f_list = [0.5, 0.45, 0.8, 0.1, 0.8, 1, 1, 0]
    concat = warped.copy()
    width = concat.shape[1]
    boarder = 192

    prepared_bottom_x, prepared_bottom_y, prepared_top_x, prepared_top_y = rm_dots(bottom_x, bottom_y, top_x, top_y)

    enl_counter = -1
    for i in range(len(enlarge_c_list)):
        enl_counter = enl_counter + 1
        enlarge_c, enlarge_f = enlarge_c_list[i], enlarge_f_list[i]
        
        is_bottom = True
        enlarge_bottom_x, enlarge_bottom_y = enlarge_coord(prepared_bottom_x, prepared_bottom_y, is_bottom, mid_arithmetic_h, enlarge_c, enlarge_f)

        is_bottom = False               
        enlarge_top_x, enlarge_top_y = enlarge_coord(prepared_top_x, prepared_top_y, is_bottom, mid_arithmetic_h, enlarge_c, enlarge_f)

        #draw_points(im, poligon_counter, bottom_x, bottom_y, top_x, top_y)

        enlarge_h = mid_arithmetic_h * (1 + enlarge_c + enlarge_f)  

        enlarge_poligon = [enlarge_bottom_x, enlarge_bottom_y, enlarge_top_x, enlarge_top_y]
        poligon = [bottom_x, bottom_y, top_x, top_y]
        
        warped = crop_areas(im, poligon, mid_arithmetic_h, enlarge_poligon, enlarge_h, enlarge_c, enlarge_f)

        res = cv2.resize(warped, dsize=(width - 2*boarder, 32), interpolation=cv2.INTER_CUBIC)
        res = cv2.copyMakeBorder( res, top=0, bottom=0, left=boarder, right=boarder, borderType=cv2.BORDER_CONSTANT )
        #cv2.imwrite('./concat/{}_{}_{}.jpg'.format(file_name, poligon_counter, enl_counter), res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        blank_image = np.zeros((5,width,1), np.uint8)
        concat = cv2.vconcat([concat, blank_image])
        concat = cv2.vconcat([concat, res])

    cv2.imwrite('./concat/{}_{}.jpg'.format(file_name, poligon_counter), concat, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
    return


def rm_dots_random(x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top):
    if len(x_plus_delta) != len(x_plus_delta_top):
        how_many_dots_to_remove = abs(len(x_plus_delta) - len(x_plus_delta_top))
        #print('how_many_dots_to_remove = ', how_many_dots_to_remove)
        rand_int = []
        if len(x_plus_delta) > len(x_plus_delta_top):
            for j in range(how_many_dots_to_remove):
                rand_int.append(random.randint(1, len(x_plus_delta)- 2))
                #rand_int.append(random.randint(1, len(x_plus_delta_top) - 2))
            rand_int.sort(reverse = True)
            for k in rand_int:
                del x_plus_delta[k]
                del y_plus_delta[k]
        else:
            for j in range(how_many_dots_to_remove):
                rand_int.append(random.randint(1, len(x_plus_delta_top) - 2))
                #rand_int.append(random.randint(1, len(x_plus_delta)- 2))
            rand_int.sort(reverse = True)
            for k in rand_int:
                del x_plus_delta_top[k]
                del y_plus_delta_top[k]

    assert len(x_plus_delta) == len(x_plus_delta_top)
    return x_plus_delta, y_plus_delta, x_plus_delta_top, y_plus_delta_top

def Euclidian_distance_sorting(bottom_x, bottom_y, lowest_point):
    xy_pair_bottom = []
    for i in range(len(bottom_x)):
        xy_pair_bottom.append([bottom_x[i], bottom_y[i]])

    Ap = np.array(lowest_point)

    B = np.array(xy_pair_bottom) # sample array of points
    dist = np.linalg.norm(B - Ap, ord=2, axis=1) # calculate Euclidean distance (2-norm of difference vectors)
    sorted_B = B[np.argsort(dist)]

    bottom_x = []
    bottom_y = []
    for i in range(len(sorted_B)):
        bottom_x.append(sorted_B[i][0])
        bottom_y.append(sorted_B[i][1])
    return bottom_x, bottom_y

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
    #grid_z = scipy.interpolate.griddata(dst, src, (grid_x, grid_y), method='cubic')
    grid_z = scipy.interpolate.griddata(dst, src, (grid_x, grid_y), method='linear')

    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(height, width)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(height, width)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    #warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_CUBIC)
    warped = cv2.remap(img, map_x_32, map_y_32, cv2.INTER_LINEAR)
    return warped

def crop_areas2(opencv_img, poligons, poligon_height):
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

def crop_areas(opencv_img, poligons, poligon_height, enlarge_poligons, enlarge_poligon_height, enlarge_c, enlarge_f):
    """
    poligons = [bottom_x, bottom_y, top_x, top_y]
    """
    boarder = 192
    src = []
    dst = []
    stripe_height = 32
    dist = 0.0
    dist_cur = 0

    hor_scale = float(stripe_height/poligon_height)
    enlarge_hor_scale = float(stripe_height/enlarge_poligon_height)

    part_of_height1 = enlarge_hor_scale * poligon_height*enlarge_c
    part_of_height2 = enlarge_hor_scale * poligon_height*(enlarge_f)

    # xu - x upper, xl - x lower
    for i in range(len(enlarge_poligons[2])):
        if(i != 0):
            dist = math.sqrt((enlarge_poligons[2][i] - enlarge_poligons[2][i - 1])**2 + (enlarge_poligons[3][i] - enlarge_poligons[3][i - 1])**2)
        dist_cur += dist

        src.append([enlarge_poligons[3][i], enlarge_poligons[2][i]])
        dst.append([0, dist_cur*hor_scale*2])

    dist = 0.0
    dist_cur = 0
    # xu - x upper, xl - x lower
    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

        src.append([poligons[3][i], poligons[2][i]])
        dst.append([part_of_height1, dist_cur*hor_scale*2])

    dist = 0.0
    dist_cur = 0

    for i in range(len(poligons[2])):
        if(i != 0):
            dist = math.sqrt((poligons[2][i] - poligons[2][i - 1])**2 + (poligons[3][i] - poligons[3][i - 1])**2)
        dist_cur += dist

        src.append([poligons[1][i], poligons[0][i]])
        dst.append([stripe_height - part_of_height2, dist_cur*hor_scale*2])
    dst_width = int(dist_cur*hor_scale*2)
    dist = 0.0
    dist_cur = 0
    for i in range(len(enlarge_poligons[2])):
        if(i != 0):
            dist = math.sqrt((enlarge_poligons[2][i] - enlarge_poligons[2][i - 1])**2 + (enlarge_poligons[3][i] - enlarge_poligons[3][i - 1])**2)
        dist_cur += dist
    
        src.append([enlarge_poligons[1][i], enlarge_poligons[0][i]])
        dst.append([stripe_height, dist_cur*hor_scale*2])
    src_arr = np.array(src)
    dst_arr = np.array(dst)
    assert len(src_arr) == len(dst_arr)
    assert len(src_arr) != 0
    assert dst_width != 0 
    warped = warp_image(opencv_img, src_arr, dst_arr, dst_width, stripe_height)
    #warped = cv2.copyMakeBorder( warped, top=0, bottom=0, left=boarder, right=boarder, borderType=cv2.BORDER_CONSTANT )
    
    return warped

def rm_dots(bottom_x, bottom_y, top_x, top_y):
    prepared_bottom_x, prepared_bottom_y, prepared_top_x, prepared_top_y = [], [], [], []
    for i in range(len(bottom_x) - 1):
        if bottom_x[i + 1] - bottom_x[i] != 0 and top_x[i + 1] - top_x[i] != 0:
            prepared_bottom_x.append(bottom_x[i])
            prepared_bottom_y.append(bottom_y[i])
            prepared_top_x.append(top_x[i])
            prepared_top_y.append(top_y[i])
    
    if bottom_x[-1] - prepared_bottom_x[-1] != 0 and top_x[-1] - prepared_top_x[-1] != 0:
        prepared_bottom_x.append(bottom_x[-1])
        prepared_bottom_y.append(bottom_y[-1])
        prepared_top_x.append(top_x[-1])
        prepared_top_y.append(top_y[-1])
    
    assert len(prepared_bottom_x) == len(prepared_top_x)
    return prepared_bottom_x, prepared_bottom_y, prepared_top_x, prepared_top_y

def enlarge_coord(bottom_x, bottom_y, is_bottom, mid_arithmetic_h, enlarge_c, enlarge_f):
    new_bottom_x, new_bottom_y = [], []

    for i in range(len(bottom_x) - 1):
        x = bottom_x[i + 1] - bottom_x[i]
        y = bottom_y[i] - bottom_y[i + 1]
        sign = 0
        if y < 0.0:
            sign = 1
            y = -y
        ugol_a = 180.0 - 90.0 - math.atan(y/x)*(180.0/math.pi)
        ugol_a_rad = ugol_a * (math.pi/180.0)
        y_ = math.sin(ugol_a_rad)*mid_arithmetic_h
        x_ = math.cos(ugol_a_rad)*mid_arithmetic_h
        if sign == 1:
            x_ = -x_
        #enlarge ceil and floor
        d_x_c = x_*enlarge_c
        d_y_c = y_*enlarge_c

        d_x_f = x_*enlarge_f
        d_y_f = y_*enlarge_f

        if is_bottom == True:
            new_bottom_x.append(bottom_x[i] + d_x_f)
            new_bottom_y.append(bottom_y[i] + d_y_f)
        else: # top
            new_bottom_x.append(bottom_x[i]  - d_x_c)
            new_bottom_y.append(bottom_y[i]  - d_y_c)

    if is_bottom == True:
        new_bottom_x.append(bottom_x[-1] + d_x_f)
        new_bottom_y.append(bottom_y[-1] + d_y_f)
    else: # top
        new_bottom_x.append(bottom_x[-1]  - d_x_c)
        new_bottom_y.append(bottom_y[-1]  - d_y_c)  

    return new_bottom_x, new_bottom_y

def parse_coord(string):
    bottom_x, bottom_y, top_x, top_y = [], [], [], []
    number_of_polygons = int(string[3])
    boarder_shift = 192
    string = string[4:]

    all_dots_num_list = string[:number_of_polygons]
    string = string[number_of_polygons:]

    shift = 0
    for j in all_dots_num_list:
        bottom_x_one_poligon, bottom_y_one_poligon, top_x_one_poligon, top_y_one_poligon = [], [], [], []
        for i in range(int(j)//2):
            x = (float(string[shift + i*4]))
            y = (float(string[shift + i*4 + 1]))
            bottom_x_one_poligon.append(x - boarder_shift)
            bottom_y_one_poligon.append(y - boarder_shift)

            x_top = (float(string[2 + shift + i*4]))
            y_top = (float(string[2 + shift + i*4 + 1]))  
            top_x_one_poligon.append(x_top - boarder_shift)
            top_y_one_poligon.append(y_top - boarder_shift)

        bottom_x.append(bottom_x_one_poligon)
        bottom_y.append(bottom_y_one_poligon)

        top_x.append(top_x_one_poligon)
        top_y.append(top_y_one_poligon) 

        shift = shift + 2*int(j)
    
    assert len(bottom_x) == len(all_dots_num_list)
    return bottom_x, bottom_y, top_x, top_y

def poligon_height(bottom_x, bottom_y, top_x, top_y):
    bbox_height_sum = 0
    dots_counter = len(bottom_x)
    for i in range(dots_counter):
        bbox_height_sum = bbox_height_sum + ((top_x[i] - bottom_x[i])**2+(top_y[i] - bottom_y[i])**2)**0.5
    mid_arithmetic_h = bbox_height_sum / dots_counter

    return mid_arithmetic_h

start_time = time.time()

with open('12.csv', encoding = 'ANSI') as fp:
    line = fp.readline()
    poligon_counter = -1
    while line:
        new_line = line[:len(line)-1]
        strings = new_line.split(',')
        print( strings[0] )
        im = cv2.imread('./real_frames/' + str(strings[0]), cv2.IMREAD_GRAYSCALE)
        file_name = strings[0][:-4]

        all_poligons_bottom_x, all_poligons_bottom_y, all_poligons_top_x, all_poligons_top_y = parse_coord(strings)
        for i in range(len(all_poligons_bottom_x)):      
            bottom_x, bottom_y, top_x, top_y = all_poligons_bottom_x[i], all_poligons_bottom_y[i], all_poligons_top_x[i], all_poligons_top_y[i]

            enlarge_c, enlarge_f = 0, 0
            mid_arithmetic_h = poligon_height(bottom_x, bottom_y, top_x, top_y) 

            poligon = [bottom_x, bottom_y, top_x, top_y]
        
            poligon_counter = poligon_counter + 1

            warped = crop_areas2(im, poligon, mid_arithmetic_h)
            cv2.imwrite('./stripes/{}_{}.jpg'.format(poligon_counter, file_name), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            check_oX_shift(warped, bottom_x, bottom_y, top_x, top_y, mid_arithmetic_h, poligon_counter)            
        line = fp.readline()        
print('end_time = ', time.time() - start_time)
