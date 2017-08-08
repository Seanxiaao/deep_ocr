# -*- coding: utf-8 -*-
#from deep_ocr/python
import numpy as np
import argparse
import cv2
import copy

import matplotlib.pyplot as plt

from deep_ocr.cv2_img_proc import PreprocessBackgroundMask
from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatio
from deep_ocr.cv2_img_proc import PreprocessRemoveNonCharNoise

from deep_ocr.utils import merge_chars_into_line_segments

####util func###
norm_width = 600
norm_height = 600
number = 0  # the parameter is used for merging radicals

indicator  = 0

def merge_lst(lst):
    for num in range(len(lst)):
        if lst[num]:
            lst = lst[:num] + [num] + lst[num + 2:]
    return lst

def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def check_if_good_boundary(boundary, norm_height, norm_width, color_img):
    preprocess_bg_mask = PreprocessBackgroundMask(boundary)
    char_w = norm_width / 20
    remove_noise = PreprocessRemoveNonCharNoise(char_w)

    id_card_img_mask = preprocess_bg_mask.do(color_img)
    id_card_img_mask[0:int(norm_height*0.05),:] = 0
    id_card_img_mask[int(norm_height*0.95): ,:] = 0
    id_card_img_mask[:, 0:int(norm_width*0.05)] = 0
    id_card_img_mask[:, int(norm_width*0.95):] = 0

    remove_noise.do(id_card_img_mask)

        ## remove right head profile
    left_half_id_card_img_mask = np.copy(id_card_img_mask)
    left_half_id_card_img_mask[:, norm_width/2:] = 0

        ## Try to find text lines and chars
    horizontal_sum = np.sum(left_half_id_card_img_mask, axis=1)
    peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
    return len(peek_ranges) >= 5 and len(peek_ranges) <= 7

def remove_white_noise(img):
    for row in range(len(img) - 1):
        for column in range(len(img[0]) - 1):
            if img[row - 1,column] == 0 and img[row ,column + 1] == 0 \
               and img[row + 1,column] == 0 and img[row ,column - 1] == 0:
               img[row,column] = 0
            #if img[row,column] == 255:
            #    lst = []
            #    lst.append(img[row ,column + 1])
            #    lst.append(img[row,column - 1])
            #    lst.append(img[row - 1,column + 1])
            #    lst.append(img[row - 1,column - 1])
            #    lst.append(img[row - 1,column])
            #    lst.append(img[row + 1,column])
            #    lst.append(img[row + 1,column - 1])
            #    lst.append(img[row + 1,column + 1])
            #    lst = np.array(lst)
            #    img[row,column] = np.median(lst)
    return img

boundaries = [
                   ([0, 0, 0], [100, 100, 100]),
                   ([0, 0, 0], [150, 150, 150]),
                   ([0, 0, 0], [200, 200, 200]),
                  ]

###input the image
if True:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	help= "path to input image file")
    ap.add_argument("-t", "--type", required = True,
    help = "type of result to get,thresh or color")
    args = vars(ap.parse_args())
    types = args["type"]
# load the image from disk
    img = cv2.imread(args["image"])
    #img = cv2.imread("picture2/idcard{}.jpg".format(num))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
####this part is for skew the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if thresh[:5,:5].all() == 0:
	    coords = np.column_stack(np.where(thresh > 0))
    else:
	    coords = np.column_stack(np.where(thresh < 255))

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
	       angle = -(90 + angle)
    else:
    	angle = - angle

    (h, w) = img.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    image_color = np.copy(img)
    preprocess_resize = PreprocessResizeKeepRatio(norm_width, norm_height)
    image_color = preprocess_resize.do(image_color)
    #image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #均衡化图片

    #image = 255 -  clahe.apply(image)

    best_boundary = None
    for boundary in boundaries:
        if check_if_good_boundary(
                    boundary,
                    norm_height, norm_width,
                    image_color):
            best_boundary = boundary
            break
    if best_boundary is None:
        print "error"


    #####模板化,将头像和别处变黑###

    ####找到字体为蓝的部分#####
    color_img =  np.copy(image_color)
    lower_blue = np.array([77, 43, 46])
    upper_blue = np.array([120,255,255])
    hsv = cv2.cvtColor(image_color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(hsv,hsv, mask= mask)
    #cv2.imshow("res",res)
    brim = np.column_stack(np.where(res != 0))[::3,0:2]
    for temp in range(len(brim)):
        color_img[brim[temp][0],brim[temp][1]] = 255
    #cv2.imshow("color_img",color_img)
    #通过三通道的hsv值找到位置
    #换成平均颜色


    boundary = best_boundary
    preprocess_bg_mask = PreprocessBackgroundMask(boundary)


    if True:

        id_card_img_mask = preprocess_bg_mask.do(color_img)

        id_card_img_mask[0:int(norm_height*0.05),:] = 0
        id_card_img_mask[int(norm_height*0.95): ,:] = 0
        id_card_img_mask[:, 0:int(norm_width*0.05)] = 0
        id_card_img_mask[:, int(norm_width*0.95):] = 0

        left_half_id_card_img_mask = np.copy(id_card_img_mask)
        left_half_id_card_img_mask[:norm_height*1/2, 3*norm_width/5:] = 0
        #using mask to conceal the photo
        #cv2.imshow("left_half_id_card_img_mask",left_half_id_card_img_mask)
        left_half_id_card_img_mask = remove_white_noise(left_half_id_card_img_mask)
        cv2.imwrite('result/image_{}_b.jpg'.format(args["image"][-5]), left_half_id_card_img_mask)
        #kernel = np.ones((1,2),np.uint8)
        #opening = cv2.morphologyEx(left_half_id_card_img_mask, cv2.MORPH_OPEN, kernel)
        #cv2.imshow("left_half_id_card_img_mask2",left_half_id_card_img_mask)
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2, 1))
        #eroded = cv2.erode(left_half_id_card_img_mask,kernel)
        #cv2.imshow("Eroded Image",eroded)


        horizontal_sum = np.sum(left_half_id_card_img_mask, axis=1)
    #horizontal_sum = np.sum(left_half_id_card_img_mask, axis=1)

    #plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    #plt.gca().invert_yaxis()
    #plt.show()

        peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
        line_seg_left_half_id_card_img_mask = np.copy(left_half_id_card_img_mask)
        line_seg_left_half_id_card_img_mask_inverse = 255 - line_seg_left_half_id_card_img_mask
        for i, peek_range in enumerate(peek_ranges):
            x = 0
            y = peek_range[0]
            w = line_seg_left_half_id_card_img_mask.shape[1]
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(line_seg_left_half_id_card_img_mask, pt1, pt2, 255)
        #cv2.imshow("line_seg_left_half_id_card_img_mask",line_seg_left_half_id_card_img_mask)
        #cv2.waitKey(0)



        vertical_peek_ranges2d = []
        for peek_range in peek_ranges:
            start_y = peek_range[0]
            end_y = peek_range[1]
            line_img = left_half_id_card_img_mask[start_y:end_y, :]
            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = extract_peek_ranges_from_array(
                vertical_sum,
                minimun_val=40,
                minimun_range=1)
            vertical_peek_ranges2d.append(vertical_peek_ranges)
        vertical_peek_ranges2d_instruction = merge_chars_into_line_segments(copy.copy(vertical_peek_ranges2d))

        #get instructing dictionary
        ## name extraction
        instructing_lst = []
        range_y = peek_ranges[0]
        range_x = vertical_peek_ranges2d_instruction[0][0]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"name"])

        """
        ## sex extraction
        range_y = peek_ranges[1]
        range_x = vertical_peek_ranges2d_instruction[1][0]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"sex"])

        ## minzu extraction
        range_y = peek_ranges[1]
        range_x = vertical_peek_ranges2d_instruction[1][1]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"minzu"])

        ## year extraction
        range_y = peek_ranges[2]
        range_x = vertical_peek_ranges2d_instruction[2][0]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"year"])

        ## month extraction
        range_y = peek_ranges[2]
        range_x = vertical_peek_ranges2d_instruction[2][1]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"month"])

        ## day extraction
        range_y = peek_ranges[2]
        range_x = vertical_peek_ranges2d_instruction[2][2]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"day"])

        ## address extraction
        first_line = peek_ranges[3]
        second_line = peek_ranges[4]
        line_range_x = vertical_peek_ranges2d_instruction[3][0]
        line_start_x = line_range_x[0]
        line_end_x = line_range_x[1]
        line_start_y = first_line[0]
        line_end_y = first_line[1]
        second_line_start_y =second_line[0]
        second_line_end_y = second_line[1]
        instructing_lst.append([line_start_x,line_start_y,line_end_x,line_end_y,"address1"])
        instructing_lst.append([line_start_x,second_line_start_y,line_end_x,second_line_end_y,"address2"])

        ## id extraction
        range_y = peek_ranges[-1]
        range_x = vertical_peek_ranges2d_instruction[-1][0]
        start_x, end_x = range_x
        start_y, end_y = range_y
        instructing_lst.append([start_x,start_y,end_x,end_y,"id"])
        """

        #[[153, 97, 209, 118, 'name'], [150, 143, 165, 160, 'sex'], [256, 143, 271, 160, 'minzu'], [150, 184, 192, 199, 'year'], [234, 184, 243, 199, 'month'], [277, 184, 298, 199, 'day'], [149, 225, 357, 272, 'address'], [233, 332, 514, 351, 'id']]
        ## Draw

        color = (0, 0, 255)
        filtered, wh_lst = [], []

        for i, peek_range in enumerate(peek_ranges):
            for j,vertical_range in enumerate(vertical_peek_ranges2d[i]):
                x = vertical_range[0]  #放缩出部分距离
                y = peek_range[0]
                w = vertical_range[1] - x
                h = peek_range[1] - y
                filtered.append([x,y,w,h,j,i])



        ####try to merge the radical and the main part
        ###合并部首
        while number < len(filtered):
            #to avoid the problem occuring by continuous radicals
            #try:
            if number >= len(filtered):
                break
            else:
                if filtered[number][-1] == 0:
                    if filtered[number][-2] == 0: #set for the beginning
                        right_one,after_one = filtered[number], filtered[number + 1]
                    elif filtered[number + 1][-1] == 1: #set for the end
                        before_one,right_one = filtered[number - 1 ], filtered[number]
                    else:
                        before_one,right_one,after_one  = filtered[number - 1], filtered[number], filtered[number + 1]
                else:
                    number += 1
                    continue
                if right_one[-1] == 0 and right_one[-2] < 9: #target the first_line
                    if float(right_one[2])/right_one[3] < 0.43: ###recogize the radicals
                        if right_one[-2] == 0:
                            temp  = [[right_one[0],right_one[1],after_one[0] - right_one[0]+ after_one[2],right_one[3],float(after_one[4]+right_one[4])/2,1]]
                            filtered = filtered[:number] + temp + filtered[number + 2:]
                            number -= 1
                        else:
                        #左
                            if  after_one[0] - (right_one[0] +  right_one[2]) < right_one[0] -(before_one[0] +  before_one[2]): # compare the two border
                                temp  = [[right_one[0],right_one[1],after_one[0] - right_one[0]+ after_one[2],right_one[3],float(after_one[4]+right_one[4])/2,1]]
                                filtered = filtered[:number] + temp + filtered[number + 2:]
                                number -= 2
                        #右
                            else:
                                temp  = [[before_one[0],before_one[1],right_one[0] - before_one[0]+right_one[2],before_one[3],float(before_one[4]+right_one[4])/2,1]]
                                filtered = filtered[:number -1] + temp + filtered[number + 1:]
                                number -= 2
                    else:
                        number += 1
                        continue
                else:
                    number += 1
                    continue

        number = 0#使用了全局变量，在下一次使用的时候需要归零
         ###合并部首



        # have question in filtered j, not always have sequences
        #### try to remove the noise
        for temp in range(len(filtered)):
            wid = filtered[temp][2]
            #if True:
            if wid > 3 and wid <= 30: #set threshold which is easyier than prune the list
                x,y,w,h,j  = filtered[temp][0],filtered[temp][1],filtered[temp][2],filtered[temp][3],filtered[temp][4]
                center = (x + w/2, y + h/2)
                label = "error"
                for index in range(len(instructing_lst)):
                    if center[0] <= instructing_lst[index][2] and center[0] >= instructing_lst[index][0] and \
                       center[1] <= instructing_lst[index][3] and center[1] >= instructing_lst[index][1]:
                       label = instructing_lst[index][-1]
                temp_binary = line_seg_left_half_id_card_img_mask_inverse[y-1:y + h+2, x-2:x + w+2]
                temp_color = image_color[y-1:y + h+2, x-1:x + w+2]

                #reoperate on the pruned img to get a better result
                temp_gray = cv2.cvtColor(temp_color,cv2.COLOR_BGR2GRAY)
                thresh_temp = cv2.adaptiveThreshold(temp_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
                #the paremeter can be operated, as i know, the greater the second number is, the darker will the output be
                if label == "name" or label == "id":
            #in order to get the black words
            #if w > 25: #切图
                    if types == "thresh":
                        cv2.imwrite("restore/words/{}_{}_img{}.png".format(label,j,args["image"][-5]),thresh_temp)
                    elif types == "color":
                            cv2.imwrite("restore/words/{}_{}_imgk{}.png".format(label,j,args["image"][-5]),temp_color)
                    else:
                        raise error
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                cv2.rectangle(image_color, pt1, pt2, color)
        #cv2.imshow("result_{}".format(num), image_color)
        #cv2.waitKey(10500)
        #cv2.destroyAllWindows()
        #cv2.imwrite('result/image_{}_m.jpg'.format(num), image_color)
    #except:
        #continue
