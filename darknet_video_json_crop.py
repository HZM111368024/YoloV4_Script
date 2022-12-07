from ctypes import *
import random
import os
import json
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import math
import sys


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="x64/backup/yolo-obj_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="x64/cfg/yolo-obj.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="x64/data/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.5,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise (ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image, bbox, begin_x, begin_y):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * math.floor(width / 4))
    orig_y = int(y * math.floor(height / 4))
    orig_width = int(w * math.floor(width / 4))
    orig_height = int(h * math.floor(height / 4))

    bbox_converted = (begin_x + orig_x, begin_y + orig_y, orig_width, orig_height)

    return bbox_converted


# 建立資料夾並儲存照片
def storageImg(path, imgName, img):
    outputName = '.\\Yolov4_Output_Image\\' + path + '\\' + imgName + '.jpg'
    folder = os.path.exists('.\\Yolov4_Output_Image\\' + path)
    if not folder:
        # 如果不存在，則建立新目錄
        os.makedirs('.\\Yolov4_Output_Image\\' + path)
        print('-----建立資料夾成功-----')
        cv2.imwrite(outputName, img)

    else:
        # 如果目錄已存在，則不建立
        print('-----儲存成功-----')
        cv2.imwrite(outputName, img)


def video_capture(frame_queue, darknet_image_queue):
    # 6FPS
    timeStamp = 5;
    t = 0;
    crop_width = math.floor(width / 4)
    crop_height = math.floor(height / 4)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if t%timeStamp == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_queue.put(frame)
        for crop_x in range(0, math.floor(width - (crop_width * 4 / 10)), math.floor(crop_width * 0.6)):
            for crop_y in range(0, math.floor(height - (crop_height * 4 / 10)), math.floor(crop_height * 0.6)):
                crop_img = frame_rgb[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
                crop_img_resized = cv2.resize(crop_img, (darknet_width, darknet_height),
                                              interpolation=cv2.INTER_LINEAR)
                img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
                darknet.copy_image_from_bytes(img_for_detect, crop_img_resized.tobytes())
                crop_x_queue.put(crop_x)
                crop_y_queue.put(crop_y)
                darknet_image_queue.put(img_for_detect)
        t = t + 1
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    t = 1;
    while cap.isOpened():
        prev_time = time.time()
        for i in range(0, 6, 1):
            for j in range(0, 6, 1):
                darknet_image = darknet_image_queue.get()
                detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh,
                                                  hier_thresh=.5, nms=.1)
                detections_queue.put(detections)
                darknet.print_detections(detections, args.ext_output)
                darknet.free_image(darknet_image)
        fps = int(1 / (time.time() - prev_time))
        print("Fps" + str(fps))
        fps_queue.put(fps)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    # video = set_saved_video(cap, args.out_filename, (width, height))
    person = set(['Pistol'])
    t = 1;
    # 6FPS

    time = 0
    timeCount = 0
    storetime = 0
    crop_width = math.floor(width / 4)
    crop_height = math.floor(height / 4)
    timestamp_fps = [0, 0.166667, 0.333333, 0.5, 0.666667, 0.833333]
    while cap.isOpened():
        frame = frame_queue.get()
        fps = fps_queue.get()
        all_confidence = []
        all_boxes = []
        all_label = []
        detections_adjusted = []
        if frame is not None:
            for i in range(0, 6, 1):
                for j in range(0, 6, 1):
                    detections = detections_queue.get()
                    crop_x = crop_x_queue.get()
                    crop_y = crop_y_queue.get()
                    for label, confidence, bbox in detections:
                        bbox_adjusted = convert2original(frame, bbox, crop_x, crop_y)
                        if bbox_adjusted[2] < crop_width / 2 and bbox_adjusted[3] < crop_height / 2:
                            all_confidence.append(float(confidence))
                            all_boxes.append(bbox_adjusted)
                            all_label.append(label)
            idxs = cv2.dnn.NMSBoxes(all_boxes, all_confidence, 0.5, 0.2)
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (all_boxes[i][0], all_boxes[i][1])
                    (w, h) = (all_boxes[i][2], all_boxes[i][3])
                    print((all_label[i], all_confidence[i], (x, y, w, h)))
                    detections_adjusted.append((all_label[i], all_confidence[i], (x, y, w, h)))
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            if timeCount % 5 == 0:
                imgName = targetName + '#t=' + str(time + timestamp_fps[storetime % 6])
                # 儲存圖片
                storageImg(targetName, imgName, image)
                calculating(detections_adjusted, len(detections_adjusted), time + timestamp_fps[storetime % 6], imgName)
                storetime += 1
                if storetime % 6 == 0:
                    time = time + 1
            # if args.out_filename is not None:
            #     video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break

        t = t + 1
        # print(str(time + timestamp_fps[timeCount]))
        timeCount = timeCount + 1

    cap.release()
    output_json_file()
    # video.release()
    cv2.destroyAllWindows()


# 組合匯出結果
def calculating(detections, person_number, time, imgName):
    calculates_item = {}
    calculates_item['boundingBox'] = []
    print('timestamp: ' + str(time))
    print('人數: ' + str(person_number))
    calculates_item['name'] = imgName
    calculates_item['timestamp'] = time
    calculates_item['count'] = person_number

    for label, confidence, bbox in detections:
        # 取得bbox
        xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax < 0:
            xmax = 0
        if ymax < 0:
            ymax = 0

        calculates_item['boundingBox'].append([ymin, xmin, ymax, xmax])

    caculates['regions'].append(calculates_item)
    # 人數 person_number


def output_json_file():
    jsonfolderpath = "./Yolov4_Output_Json"
    # 檢查目錄是否存在 
    if not os.path.isdir(jsonfolderpath):
        # 如果json資料夾不存在，建立一個
        print("-------------目錄不存在-------------")
        os.makedirs('./Yolov4_Output_Json')
        print('-----建立資料夾成功-----')
    # else:
    #     print("-------------目錄存在-------------")

    file = jsonfolderpath + "/" + targetName + '.json'
    with open(file, 'w') as obj:
        # 輸出成JSON並格式化
        json.dump(caculates, obj, indent=4, separators=(',', ':'))


if __name__ == '__main__':
    # 用法: python x64/darknet_video_json_crop.py --input C:\Users\WIN\Desktop\碩士\test.mp4
    args = parser()
    input_path = str2int(args.input)
    substrName = input_path.rfind('\\')
    targetName = input_path[substrName + 1:-4]

    caculates = {}
    caculates['regions'] = []
    caculates['name'] = targetName
    regions = []

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=36)
    detections_queue = Queue(maxsize=36)
    fps_queue = Queue(maxsize=1)
    crop_x_queue = Queue(maxsize=36)
    crop_y_queue = Queue(maxsize=36)

    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        # 'x64/cfg/yolo-obj.cfg',
        args.data_file,
        # 'x64/data/obj.data',
        args.weights,
        # 'x64/backup/yolo-obj_best.weights',
        batch_size=1
    )
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    # width = darknet.network_width(network)
    # height = darknet.network_height(network)
    # 測試

    # input_path = sys.argv[1]
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

    sys.exit()
