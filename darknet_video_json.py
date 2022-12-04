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
    parser.add_argument("--thresh", type=float, default=.25,
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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if t%timeStamp == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
        t = t + 1
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    t = 1;
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1 / (time.time() - prev_time))
        fps_queue.put(fps)
        darknet.print_detections(detections, args.ext_output)
        t = t + 1
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
    timestamp_fps = [0, 0.166667, 0.333333, 0.5, 0.666667, 0.833333]
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        # 每五個一循還
        print("storetime%d" % storetime)

        if frame_resized is not None:
            # 只取person
            detections = [a for a in detections if a[0] in person]
            print('\n偵測人數：' + str(len(detections)))

            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("time%d" % time)
            if timeCount % 5 == 0:
                imgName = targetName + '#t=' + str(time + timestamp_fps[storetime % 6])
                # 儲存圖片
                storageImg(targetName, imgName, image)
                calculating(detections, len(detections), time + timestamp_fps[storetime % 6], imgName)
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
    # 用法: python x64/darknet_video_json.py --input C:\Users\WIN\Desktop\碩士\test.mp4
    args = parser()
    input_path = str2int(args.input)
    substrName = input_path.rfind('\\')
    targetName = input_path[substrName + 1:-4]

    caculates = {}
    caculates['regions'] = []
    caculates['name'] = targetName
    regions = []

    frame_queue = Queue(maxsize=1)
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

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

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    darknet_image = darknet.make_image(width, height, 3)

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()
