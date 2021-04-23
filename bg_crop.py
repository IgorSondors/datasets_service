import numpy as np
import cv2
import os

height = 32
width = 2670

bg_folder = 'backgrounds_full'

f1 = next(os.walk(bg_folder))
bg_names = f1[2]
for i in bg_names:
    img_background = cv2.imread("backgrounds_full/{}".format(i), cv2.IMREAD_GRAYSCALE)
    img_height, img_width = img_background.shape
    print(i, img_width, img_height)

    Y_counter = int(img_height / (0.7 * height)) #- 1
    X_counter = int(img_width / (0.7 * width)) #- 1

    y = - int(0.7 * height)
    for j in range(Y_counter):
        x = - int(0.7 * width)
        y = y + int(0.7 * height)
        for k in range(X_counter):
            x = x + int(0.7 * width)

            x_top_left = x_down_left = x
            x_top_right = x_down_right = x + width
            y_top_left = y_top_right = y
            y_down_right = y_down_left = y + height
            
            src_pts =  np.array([[x_top_left, y_top_left], [x_top_right, y_top_right], [x_down_right, y_down_right], [x_down_left, y_down_left]], dtype="float32")
            dst_pts = np.array([[0, 0],[width-1, 0],[width-1, height-1],[0, height-1]], dtype="float32")
            #the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            #directly warp the rotated rectangle to get the straightened rectangle
            dst = cv2.warpPerspective(img_background, M, (width, height))

            dst = cv2.hconcat((dst, np.zeros((np.shape(dst)[0], 192, 1), dtype=np.uint8) ))
            dst = cv2.hconcat((np.zeros((np.shape(dst)[0], 192, 1), dtype=np.uint8), dst ))

            cv2.imwrite('false_strides/{}_{}_{}.jpg'.format(i, j, k), dst)