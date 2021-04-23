"""
check csv writing 
"""
import numpy as np
import h5py 
import cv2
import json
import time

start_time = time.time()

def top_bottom_dots(point_x, point_y, edge_x, edge_y):
    if point_x[0] in edge_x:
        is_good_rect = True

        index_0 = point_x.index(edge_x[0])
        index_1 = point_x.index(edge_x[1])
        index_2 = point_x.index(edge_x[2])
        index_3 = point_x.index(edge_x[3])

        line_0_x = point_x[index_0 : index_1 + 1]
        line_0_y = point_y[index_0 : index_1 + 1]

        line_1_x = point_x[index_1 : index_2 + 1]
        line_1_y = point_y[index_1 : index_2 + 1]

        line_2_x = point_x[index_2 : index_3 + 1]
        line_2_y = point_y[index_2 : index_3 + 1]

        line_3_x = point_x[index_3 : ] + [point_x[0]]
        line_3_y = point_y[index_3 : ] + [point_y[0]]

        lines = [[line_0_x, line_0_y], [line_1_x, line_1_y], [line_2_x, line_2_y], [line_3_x, line_3_y]]
        lines_len = []
        for j in lines:
            line_len = 0
            for i in range(len(j[0]) - 1):
                x1 = j[0][i]
                y1 = j[1][i]
                x2 = j[0][i + 1]
                y2 = j[1][i + 1]
                
                next_len = ((x2 - x1)**2+(y2 - y1)**2)**0.5
                line_len = line_len + next_len
            lines_len.append(line_len)
        lines_len, lines = zip(*sorted(zip(lines_len, lines),reverse = True))
        lines_top_bottom = lines[0 : 3]
        
        first_mid_y = 0
        for i in lines_top_bottom[0][1]:
            first_mid_y = first_mid_y + i/len(lines_top_bottom[0][1])
        second_mid_y = 0
        for i in lines_top_bottom[1][1]:
            second_mid_y = second_mid_y + i/len(lines_top_bottom[1][1])

        if first_mid_y > second_mid_y:
            bottom_x, bottom_y = lines_top_bottom[0][0], lines_top_bottom[0][1]
            top_x, top_y = lines_top_bottom[1][0], lines_top_bottom[1][1]
        else:
            bottom_x, bottom_y = lines_top_bottom[1][0], lines_top_bottom[1][1]
            top_x, top_y = lines_top_bottom[0][0], lines_top_bottom[0][1]
    else:
        is_good_rect = False
        bottom_x, bottom_y, top_x, top_y = [], [], [], []
    return bottom_x, bottom_y, top_x, top_y, is_good_rect

def find_bbox_coord(point_x, point_y):
    is_good_rect = True
    bottom_x, bottom_y = [], []
    top_x, top_y = [], []
    if len(point_x) < 4:
        is_good_rect = False
    if len(point_x) == 4:
        out_of_repeats_x = []
        out_of_repeats_y = []
        delta = 10**(-6)
        for j in range(len(point_x)): # add delta for the reason of not mess in equal angles
            out_of_repeats_x.append(point_x[j] + delta*j)
            out_of_repeats_y.append(point_y[j] + delta*j)
        point_x, point_y = out_of_repeats_x, out_of_repeats_y
        
        quadrate_width = ((point_x[1] - point_x[0])**2+(point_y[1] - point_y[0])**2)**0.5
        quadrate_height = ((point_x[1] - point_x[2])**2+(point_y[1] - point_y[2])**2)**0.5
        aspect_ratio = quadrate_width / quadrate_height
        if aspect_ratio > 0.7 and aspect_ratio < 1.3:
            is_good_rect = False   
            ###Aprint('Квадрат. Закрашиваем')
        elif quadrate_width * quadrate_height < 100:
            is_good_rect = False   
            ###Aprint('Квадрат. Закрашиваем')
        else:
            ###Aprint('Прямоугольник')
            edge_x, edge_y = point_x, point_y
            bottom_x, bottom_y, top_x, top_y, is_good_rect = top_bottom_dots(point_x, point_y, edge_x, edge_y)
                    
    elif len(point_x) > 4:
        ###Aprint('Многоугольник')
        out_of_repeats_x = []
        out_of_repeats_y = []
        delta = 10**(-4)
        for j in range(len(point_x)): # add delta for the reason of not mess in equal angles
            out_of_repeats_x.append(point_x[j] + delta*j)
            out_of_repeats_y.append(point_y[j] + delta*j)
        point_x, point_y = out_of_repeats_x, out_of_repeats_y
        
        edge_x, edge_y = find_4_dots(point_x, point_y)

        bottom_x, bottom_y, top_x, top_y, is_good_rect = top_bottom_dots(point_x, point_y, edge_x, edge_y)
          
    if is_good_rect:
        
        bottom_edge_x, bottom_edge_y = [], []
        for i in bottom_x:
            if i in edge_x:
                index = bottom_x.index(i)
                bottom_edge_x.append(bottom_x[index])
                bottom_edge_y.append(bottom_y[index])
        bottom_edge_x, bottom_edge_y = zip(*sorted(zip(bottom_edge_x, bottom_edge_y)))
        bottom_lowest_point = [bottom_edge_x[0], bottom_edge_y[0]]

        top_edge_x, top_edge_y = [], []
        for i in top_x:
            if i in edge_x:
                index = top_x.index(i)
                top_edge_x.append(top_x[index])
                top_edge_y.append(top_y[index])
        top_edge_x, top_edge_y = zip(*sorted(zip(top_edge_x, top_edge_y)))
        top_lowest_point = [top_edge_x[0], top_edge_y[0]]

        bottom_x, bottom_y = Euclidian_distance_sorting(bottom_x, bottom_y, bottom_lowest_point)
        top_x, top_y = Euclidian_distance_sorting(top_x, top_y, top_lowest_point)
    else:
        bottom_x, bottom_y, top_x, top_y = [], [], [], []
    
    return is_good_rect, bottom_x, bottom_y, top_x, top_y

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

def recalculate_top(bottom_x, bottom_y, top_x, top_y):
    top_lines = []
    bottom_lines = []
    bottom_normal_lines = []
    for i in range(len(top_x) - 1):
        k = (top_y[i] - top_y[i+1])/(top_x[i] - top_x[i+1])
        b = top_y[i] - k * top_x[i]
        top_lines.append((k, b))
    for i in range(len(bottom_x) - 1):
        k = (bottom_y[i] - bottom_y[i+1])/(bottom_x[i] - bottom_x[i+1])
        b = bottom_y[i] - k * bottom_x[i]
        bottom_lines.append((k, b))

        k_normal = -1/k
        b_normal = bottom_y[i + 1] - k_normal * bottom_x[i + 1]
        bottom_normal_lines.append((k_normal, b_normal))
    
    new_top_x = [top_x[0]]
    new_top_y = [top_y[0]]
    if len(bottom_normal_lines) > 1:
        bottom_normal_lines = bottom_normal_lines[:-1]
        bottom_lines = bottom_lines[:-1]

        for j in range(len(bottom_normal_lines)):
            k_normal = bottom_normal_lines[j][0]
            b_normal = bottom_normal_lines[j][1]

            for i in range(len(top_lines)):
                k_top = top_lines[i][0]
                b_top = top_lines[i][1] 

                x0 = (b_top - b_normal) / (k_normal - k_top)
                y0 = k_normal * x0 + b_normal

                x1 = top_x[i]
                y1 = top_y[i]
                x2 = top_x[i + 1]
                y2 = top_y[i + 1]

                if x0 >= x1 and x0 <= x2:
                    if y0 >= y1 and y0 <= y2:
                        new_top_x.append(x0)
                        new_top_y.append(y0)
                        continue
                    elif y0 <= y1 and y0 >= y2:
                        new_top_x.append(x0)
                        new_top_y.append(y0)
                        continue
                elif x0 <= x1 and x0 >= x2:
                    if y0 >= y1 and y0 <= y2:
                        new_top_x.append(x0)
                        new_top_y.append(y0)
                        continue
                    elif y0 <= y1 and y0 >= y2:
                        new_top_x.append(x0)
                        new_top_y.append(y0)
                        continue

    new_top_x.append(top_x[-1])
    new_top_y.append(top_y[-1])
    assert len(new_top_x) == len(bottom_x)
    return new_top_x, new_top_y

def kx_plus_b(bottom_x, bottom_y):
    x_plus_delta = []
    y_plus_delta = []
    pixel_step = 1
    poligon_dots = 0
    for i in range(len(bottom_x) - 1):
        next_x = int(bottom_x[i])
        next_y = bottom_y[i]
        x_plus_delta.append(next_x)
        y_plus_delta.append(round(next_y,1))
        poligon_dots = poligon_dots + 1
        try:
            k = (bottom_y[i] - bottom_y[i+1])/(bottom_x[i] - bottom_x[i+1])
            b = bottom_y[i] - k * bottom_x[i]
            dots_between_edges = int((((bottom_x[i+1] - bottom_x[i])**2 + (bottom_y[i+1] - bottom_y[i])**2)**0.5) / pixel_step)
            X_step = (bottom_x[i+1] - bottom_x[i]) / dots_between_edges
            for j in range(dots_between_edges):
                next_x = next_x + X_step
                next_y = k * next_x + b
                x_plus_delta.append(int(next_x))
                y_plus_delta.append(round(next_y,1))
            poligon_dots = poligon_dots + dots_between_edges
        except:
            print('Расстояние между точками 0 пикселей! Пропуск точки')

    x_plus_delta.append(int(bottom_x[-1]))
    y_plus_delta.append(round(bottom_y[-1],1))
    poligon_dots = poligon_dots + 1
    return poligon_dots, x_plus_delta, y_plus_delta

def recalculate_top2(bottom_x, bottom_y, top_x, top_y):
    print('recalculate_top2')
    new_top_x = [top_x[0]]
    new_top_y = [top_y[0]]
    
    if len(bottom_x) > 2:
        poligon_dots, x_plus_delta, y_plus_delta = kx_plus_b(top_x, top_y)
        del x_plus_delta[0]
        del y_plus_delta[0]
        del x_plus_delta[-1]
        del y_plus_delta[-1]

        copy_bottom_x = bottom_x.copy()
        copy_bottom_y = bottom_y.copy()
        del copy_bottom_x[0]
        del copy_bottom_y[0]
        del copy_bottom_x[-1]
        del copy_bottom_y[-1]

        for i in range(len(copy_bottom_x)):
            lowest_point = [copy_bottom_x[i], copy_bottom_y[i]]
            x, y = Euclidian_distance_sorting(x_plus_delta, y_plus_delta, lowest_point)
            new_top_x.append(x[0])
            new_top_y.append(y[0])
    new_top_x.append(top_x[-1])
    new_top_y.append(top_y[-1])
    
    assert len(new_top_x) == len(bottom_x)
    return new_top_x, new_top_y

def draw_black_rect(im, point_pairs):
    cnt = []
    for i in range(len(point_pairs) - 2, -2, -2):
        start_point = (point_pairs[i], point_pairs[i + 1])
        end_point = (point_pairs[i - 2], point_pairs[i + 1 - 2])
        cnt.append(start_point) 
        cnt.append(end_point) 
    black = cv2.fillPoly(im, np.array([cnt]), (0,0,0), lineType=cv2.LINE_AA)
    return black

def find_4_dots(point_x, point_y):
    i = 0
    curve = []  #c**2 = a**2 + b**2 - 2ab*cos_alpha
    
    while i <= len(point_x) - 2: # except last angle
        x1 =  point_x[i]
        y1 = point_y[i]
        x2 =  point_x[i + 1]
        y2 = point_y[i + 1]
        x0 =  point_x[i - 1]
        y0 = point_y[i - 1]
        
        a = ((x2 - x1)**2+(y2 - y1)**2)**0.5
        b = ((x1 - x0)**2+(y1 - y0)**2)**0.5
        c = ((x2 - x0)**2+(y2 - y0)**2)**0.5

        cos_alpha = (a**2 + b**2 - c**2)/(2*a*b)

        curve.append(abs(cos_alpha))
        i = i + 1
    # last angle    
    x1 =  point_x[-1]
    y1 = point_y[-1]
    x2 =  point_x[0]
    y2 = point_y[0]
    x0 =  point_x[-2]
    y0 = point_y[-2]
    
    a = ((x2 - x1)**2+(y2 - y1)**2)**0.5
    b = ((x1 - x0)**2+(y1 - y0)**2)**0.5
    c = ((x2 - x0)**2+(y2 - y0)**2)**0.5   

    cos_alpha = (a**2 + b**2 - c**2)/(2*a*b)
    curve.append(abs(cos_alpha))

    curve_out_of_repeats = []
    for j in range(len(curve)): # add delta for the reason of not mess in equal angles
        delta = 10**(-10)
        curve_out_of_repeats.append(curve[j] + delta*j)

    curve_out_of_edges = curve_out_of_repeats.copy()
    edge_angles = []
    edge_indexes = []
    edge_x = []
    edge_y = []
    for i in range(4):
        edge_angles.append(min(curve_out_of_edges))
        curve_out_of_edges.remove(min(curve_out_of_edges))

    for i in edge_angles:
        index = curve_out_of_repeats.index(i)
        edge_indexes.append(index)
    edge_indexes = sorted(edge_indexes)

    for i in edge_indexes:
        edge_x.append(point_x[i])
        edge_y.append(point_y[i])
        
    return edge_x, edge_y

def top_down_points(im, bottom_x, bottom_y, top_x, top_y):
    # Radius of circle
    radius = 1
    thickness = 5
    for i in range(len(bottom_x)):
        x, y = bottom_x[i], bottom_y[i]
        center_coordinates = (int(x), int(y))
        color = (0, 0, 255)
        if bottom_x[i] == bottom_x[-1]:
            color = (0, 255, 0)
            thickness = 5
        else:
            start_point = center_coordinates
            end_point = (int(bottom_x[i + 1]), int(bottom_y[i + 1]))
            im = cv2.line(im, start_point, end_point, color, int(thickness/2))

        im = cv2.circle(im, center_coordinates, radius, color, thickness)

        x, y = top_x[i], top_y[i] 
        center_coordinates = (int(x), int(y))
        color = (255, 0, 0)
        if top_x[i] == top_x[-1]:
            color = (0, 255, 0)
            thickness = 5
        else:
            start_point = center_coordinates
            end_point = (int(top_x[i + 1]), int(top_y[i + 1]))
            im = cv2.line(im, start_point, end_point, color, int(thickness/2))
        im = cv2.circle(im, center_coordinates, radius, color, thickness)

    return im

with open('input_total - Copy.txt', encoding = 'utf8') as fp:
    lines = fp.readlines()

    for line in lines:
        js_line = json.loads(line)
        file_name = js_line['file'].split('/')
        file_name = file_name[-1].split('?')
        file_path = './dataset_ocr_good/' + file_name[0]
        im = cv2.imread(file_path)
        if im is None:
            continue
        print(file_name[0])
        height, width = im.shape[:2]

        second = js_line['result']
          
        All_x_under_word, All_y_under_word, All_word_height = [], [], []
        All_image_dots = 0            
        # second - один словарик для одного из изображений из множества {"file":"name.jpg","result":[...]}, 
        # включающий в себя [...] = [{"type":"polygon","data":[{"x":0.4,"y":0.4}, {}...], "readable":t/f}, ...]
        for s in second: # s - один из множества полигонов одной картинки
            is_good_rect = False
            if s['readable']:
                is_good_rect = True
            data = s['data']
            
            point_pairs = []
            point_x = []
            point_y = []

            for d in data: # Множество точек одного полигона
                x = d['x']*width
                y = d['y']*height

                point_x.append(int(x))
                point_y.append(int(y))

                point_pairs.append(int(x))
                point_pairs.append(int(y))

            if is_good_rect:
                #print(file_name[0])
                # функция определяющая где верх и низ полигона и возвращающая их точки
                is_good_rect, bottom_x, bottom_y, top_x, top_y = find_bbox_coord(point_x, point_y)
            if is_good_rect:
                try:
                    top_x, top_y = recalculate_top(bottom_x, bottom_y, top_x, top_y)
                except:
                    top_x, top_y = recalculate_top2(bottom_x, bottom_y, top_x, top_y)

                poligon_dots = len(bottom_x)
                top_down_points(im, bottom_x, bottom_y, top_x, top_y)
            else:
                black = draw_black_rect(im, point_pairs)

                im = black
                poligon_dots = 0
                x_plus_delta, y_plus_delta = [], []               
            All_image_dots = All_image_dots + poligon_dots            
        if All_image_dots > 0:
            cv2.imwrite('./two_line_frames/{}'.format(file_name[0]), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])     
print('end_time = ', time.time() - start_time)