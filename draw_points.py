"""
check csv in format:
name_image_(jpeg) 0, height_image 1, width_image 2, number_of_polygons 3, number_points_in_poligon (for example poligon with 4 points - two pairs - totally 4 points),
                                                                  x_coord_low(of pair), y_coord_low(of pair), x_coord_up(of pair), y_coord_up(of pair)...
"""
import cv2

def draw_points(im, string):
    radius = 1
    number_of_polygons = int(string[3])
    thickness_multiplier = int(string[1])//1000
    if thickness_multiplier == 0:
        thickness_multiplier = 1
    string = string[4:]

    all_dots_num = 0
    all_dots_num_list = string[:number_of_polygons]

    string = string[number_of_polygons:]

    shift = 0
    for j in all_dots_num_list:
        for i in range(int(j)//2):
            thickness = 3 * thickness_multiplier
            color = (0,0,255)
            x = int(float(string[shift + i*4]))
            y = int(float(string[shift + i*4 + 1]))
            center_coord = (int(x), int(y))
            
            if x == int(float( string[shift + 2*int(j) - 4] )): 
                color = (0, 255, 0)
                thickness = 5 * thickness_multiplier
            else:
                end_point = (int(float( string[shift + (i + 1)*4] ) ), int(float( string[shift + (i + 1)*4 + 1] ) ) )
                im = cv2.line(im, center_coord, end_point, color, int(thickness/2))

            im = cv2.circle(im, center_coord, radius, color, thickness)

            color = (255,0,0)
            x = int(float(string[2 + shift + i*4]))
            y = int(float(string[2 + shift + i*4 + 1]))  
            center_coord = (int(x), int(y))

            if x == int(float( string[shift + 2*int(j) - 2] )): 
                color = (0, 255, 0)
                thickness = 5 * thickness_multiplier
            else:
                end_point = (int(float( string[2 + shift + (i + 1)*4] ) ), int(float( string[2 + shift + (i + 1)*4 + 1] ) ) )
                im = cv2.line(im, center_coord, end_point, color, int(thickness/2))

            im = cv2.circle(im, center_coord, radius, color, thickness)
        shift = shift + 2*int(j)

    return im

with open('ds_images.csv', encoding = 'ANSI') as fp:
    line = fp.readline()
    cnt = 1
    while line:
        new_line = line[:len(line)-1]
        strings = new_line.split(',')
        print( strings[0])
        im = cv2.imread('./real_frames/' + str(strings[0]))

        im = draw_points(im, strings)

        cv2.imwrite('./result/' + strings[0], im)

        line = fp.readline()
        cnt += 1
