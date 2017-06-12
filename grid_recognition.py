import cv2, sys, rospy, math
import numpy as np
import caffe
from scipy.misc import imresize
import time
import os

file_dir = None
is_debug_mode = True
mode = 'GPU'
image_h = 28
image_w = 28

model_def = 'lenet_train_test_deploy.prototxt'
model_weights = 'weights/lenet.caffemodel'
net = None
transformer = None

"""
Draw a box
"""
def draw_box(img, points, color):
    # cv2.line(img, (0,0), (100,100), color, 5)
    for i in range(len(points)):
        cv2.line(img, tuple(points[i]), tuple(points[(i+1)%(len(points))]), color, 3)



"""
Parse the file direction from argv and load the image
"""
def read_image_from_file():

    # parse the file direction
    file_dir = None
    if(len(sys.argv)>1):
        file_dir = sys.argv[1]
    else:
        print("Error: No file name provided")
        exit()

    # load the image and return
    img = cv2.imread(file_dir)
    return img



"""
Pre-processing image
"""
def preprocessing_for_number_searching(src_img):
    # convert source iamge to gray scale
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    # threshold
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                 cv2.THRESH_BINARY, 15, 3)
    # ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # blur
    gray = cv2.medianBlur(gray,3)

    return gray



"""
Analysis and filter contours
"""
def analysis_and_filter_contours_for_number_searching(contours):
    ratio = 28.0 / 16.0
    sudokuWidth = 127
    sudokuHeight = 71
    angleTolerance = 6
    ratioToleranceRate = 0.2
    dimensionsToleranceRate = 0.4

    num = 0

    contours_filtered = list()
    rects = list()
    boxes = list()
    for contour in contours:
        tempRect = cv2.minAreaRect(contour)
        # if is_debug_mode:
        #     print("[Debug] tempRect:", tempRect)

        width = tempRect[1][0]
        height = tempRect[1][1]

        if not (width > height):
            # tempRect = cv2.boxPoints((tempRect[0],(tempRect[1][0],tempRect[1][1]),tempRect[2] + 90.0))
            tempRect = (tempRect[0],(tempRect[1][1],tempRect[1][0]),tempRect[2] + 90.0)
            width = tempRect[1][0]
            height = tempRect[1][1]

        if(height==0):
            height = -1

        ratio_cur = width / height

        if (ratio_cur > (1.0-ratioToleranceRate) * ratio and \
            ratio_cur < (1.0+ratioToleranceRate) * ratio and \
                        width > (1.0-dimensionsToleranceRate) * sudokuWidth and \
            width < (1.0+dimensionsToleranceRate) * sudokuWidth and \
                        height > (1.0-dimensionsToleranceRate) * sudokuHeight and \
            height < (1.0+dimensionsToleranceRate) * sudokuHeight and \
                        ((tempRect[2] > -angleTolerance and tempRect[2] < angleTolerance) or \
              tempRect[2] < (-180+angleTolerance) or \
              tempRect[2] > (180-angleTolerance))
        ):
              contours_filtered.append(contour)
              rects.append(tempRect)
              if (is_debug_mode):
                  tempRect_points = cv2.boxPoints(tempRect)
                  boxes.append(tempRect_points)

    return contours_filtered, rects, boxes



"""
This function get rid of redundency number boxes
"""
def filter_redundency_boxes(contours, rects, number_boxes):
    bad_box_indexs = list()
    dist_toleration = 10

    for rect_i in range(len(rects)):
        if rect_i not in bad_box_indexs:
            for rect_j in range(rect_i+1, len(rects)):
                rect_i_center_x = rects[rect_i][0][0]
                rect_i_center_y = rects[rect_i][0][1]
                rect_j_center_x = rects[rect_j][0][0]
                rect_j_center_y = rects[rect_j][0][1]

                dist_x = abs(rect_i_center_x - rect_j_center_x)
                dist_y = abs(rect_i_center_y - rect_j_center_y)

                dist_ij = dist_x**2 + dist_y**2

                if dist_ij < dist_toleration**2:
                    rect_i_area = rects[rect_i][1][0] * rects[rect_i][1][1]
                    rect_j_area = rects[rect_j][1][0] * rects[rect_j][1][1]

                    if rect_i_area < rect_j_area:
                        bad_box_indexs.append(rect_j)
                    else:
                        bad_box_indexs.append(rect_i)

    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(len(number_boxes)):
        if i not in bad_box_indexs:
            good_contours.append(contours[i])
            good_rects.append(rects[i])
            good_boxes.append(number_boxes[i])

    return good_contours, good_rects, good_boxes, bad_box_indexs



"""
This function get rid of outlier number boxes
"""
def filter_outlier_boxes(contours, rects, number_boxes):
    dist_list = [0.0] * len(rects)

    for rect_i in range(len(rects)):
        for rect_j in range(rect_i+1,len(rects)):
            rect_i_center_x = rects[rect_i][0][0]
            rect_i_center_y = rects[rect_i][0][1]
            rect_j_center_x = rects[rect_j][0][0]
            rect_j_center_y = rects[rect_j][0][1]

            dist_x = abs(rect_i_center_x - rect_j_center_x)
            dist_y = abs(rect_i_center_y - rect_j_center_y)

            dist_ij = dist_x**2 + dist_y**2

            dist_list[rect_i] += dist_ij
            dist_list[rect_j] += dist_ij

    # print min(dist_list)

    bad_box_indexs = list()
    good_contours = list()
    good_rects = list()
    good_boxes = list()
    for i in range(min(9, len(rects))):
        current_min_index = dist_list.index(min(dist_list))

        bad_box_indexs.append(dist_list.pop(current_min_index))
        good_contours.append(contours.pop(current_min_index))
        good_rects.append(rects.pop(current_min_index))
        good_boxes.append(number_boxes.pop(current_min_index))

    return good_contours, good_rects, good_boxes, bad_box_indexs


def caffe_forward_pass(img):
    global net, transformer
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # input_img = preprocessing_for_number_searching(img)
    input_img = input_img.astype(np.float32) / 255.0
    input_img = imresize(input_img, [image_h, image_w])
    net.blobs['data'].data[...] = transformer.preprocess('data', input_img)
    output = net.forward()
    prob = output['prob'][0] # the prob for the first image (the only image)
    prediction = np.argmax(prob)
    return prediction


def region_of_interest(img, box):
    x0, y0 = box[1]
    x1, y1 = box[3]
    x0 = int(round(x0))
    x1 = int(round(x1))
    y0 = int(round(y0))
    y1 = int(round(y1))
    roi = img[y0:y1, x0:x1, ...]
    return roi


"""
Major structure here
"""
def number_search(src_img):
    #preprocessing image
    processed_img = preprocessing_for_number_searching(src_img)
    # cv2.imshow('processed_img', processed_img)
    # cv2.waitKey(0)

    #find contours
    im2, contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(src_img, contours, -1, (255,0,0), 3)
    # print(contours[0])

    #Analysis to get boxes
    contours, rects, number_boxes = analysis_and_filter_contours_for_number_searching(contours)
    # cv2.drawContours(src_img, contours, -1, (255,0,255), 3)

    #Avoid redundency boxes
    contours, rects, number_boxes, _ = filter_redundency_boxes(contours, rects, number_boxes)
    # print(bad_box_indexs)
    # print(number_boxes)

    #Avoid outliers
    contours, rects, number_boxes, _ = filter_outlier_boxes(contours, rects, number_boxes)

    #Debug
    box_index = 0
    for box in number_boxes:
        prediction = caffe_forward_pass(region_of_interest(src_img, box))
        draw_box(src_img, box, (255,0,255))
        box_center = rects[box_index][0]
        cv2.circle(src_img, (int(round(box_center[0])), int(round(box_center[1]))), 1, (0,0,255), 5)
        label_y = int(round(box[1, 1]))
        label_x = int(round(box[1, 0] + .5 * (box[2, 0] - box[1, 0])))
        cv2.putText(src_img, str(prediction), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        box_index += 1
    print("The size of filtered contour list is:", len(contours))
    # print(cv2.minAreaRect(contours[0]))

    return src_img


"""
Main function
"""
if __name__ == "__main__":
    # Caffe init:
    if mode == 'GPU':
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    #load src image
    # src_img = read_image_from_file()
    # cam = cv2.VideoCapture('./../Buff2017.mp4')
    cam = cv2.VideoCapture('./../buff_test_video_00.mpeg')

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = None#cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

    frame_num = 1
    segment_num = 1
    frame_rate = 24
    recording = False

    while True:
        ret, frame = cam.read()
        assert ret == True

        src_img = number_search(frame)

        cv2.imshow('src_img', src_img)
        key = cv2.waitKey(1000/frame_rate) & 0xff
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
