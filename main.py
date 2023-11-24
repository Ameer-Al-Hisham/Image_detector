
# Code video
import cv2

# use cv2 to enable video capture and cofigure it
source = cv2.VideoCapture(0)
source.set(10, 100)

# import and read coco file

objects = []
objectfile = "coco.names"
f = open(objectfile, "r")
objects = f.read().rstrip("\n").split("\n")
f.close

# Basic Config. open files required to detect images

configfile = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightfile = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightfile, configfile)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    res, img = source.read()
    indexes, confs, boxes = net.detect(img, confThreshold=0.6)
    print(indexes, boxes)
    for index, conf, box in zip(indexes, confs, boxes):
        cv2.rectangle(img, box, color=(100, 0, 0), thickness=2)
        cv2.putText(img, objects[index-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 0, 0), 2)

    # Displaying output
    cv2.imshow("output", img)
    cv2.waitKey(1)


# Code image

# import cv2
# img = cv2.imread("image2.jpg")
#
#
# # import and read coco file
#
# objects = []
# objectfile = "coco.names"
# f = open(objectfile, "r")
# objects = f.read().rstrip("\n").split("\n")
# print(objects)
# f.close
#
# # Basic Config
#
# configfile = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
# weightfile = "frozen_inference_graph.pb"
#
# net = cv2.dnn_DetectionModel(weightfile, configfile)
# net.setInputSize(320, 320)
# net.setInputScale(1.0 / 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)
#
# indexes, confs, boxes = net.detect(img, confThreshold=0.6)
# print(indexes, boxes)
# for index, conf, box in zip(indexes, confs, boxes):
#     cv2.rectangle(img, box, color=(100, 0, 0), thickness=2)
#     cv2.putText(img, objects[index-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 0, 0), 2)
# cv2.imshow("Output", img)
# cv2.waitKey(0)
