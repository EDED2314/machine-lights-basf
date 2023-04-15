from darknet.darknet import *
import cv2
import sys
import time

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



os.chdir("darknet")
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network(
    "cfg/yolov4-tiny-custom.cfg",
    "data/obj.data",
    "../training/yolov4-tiny-custom_best.weights",
)
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio


def main():
    # os.chdir("..")
    # img = cv2.imread("3.jpg")
    # img = cv2.resize(img, (480, 360))

    # detections, width_ratio, height_ratio = darknet_helper(img, width, height)
    # for label, confidence, bbox in detections:
    #     left, top, right, bottom = bbox2points(bbox)
    #     left, top, right, bottom = (
    #         int(left * width_ratio),
    #         int(top * height_ratio),
    #         int(right * width_ratio),
    #         int(bottom * height_ratio),
    #     )
    #     cv2.rectangle(img, (left, top), (right, bottom), class_colors[label], 2)
    #     cv2.putText(
    #         img,
    #         "{} [{:.2f}]".format(label, float(confidence)),
    #         (left, top - 5),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         class_colors[label],
    #         2,
    #     )
    #     print(f"{label}: {confidence}%")

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    timestart = time.time()*1000
    window_title = "machine status"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    #video_capture = cv2.VideoCapture(1)
    if video_capture.isOpened():
            try:
                cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                while True:
                    ret, frame = video_capture.read()
                    #print(frame)
                    frame = cv2.resize(frame, (width, height))
                    detections, width_ratio, height_ratio = darknet_helper(frame, width, height)
                    print(detections)
                    newTime = time.time()*1000
                    print(newTime - timestart)
                    timestart = newTime
                    # # loop through detections and draw them on webcam image
                    # for label, confidence, bbox in detections:
                    #     left, top, right, bottom = bbox2points(bbox)
                    #     left, top, right, bottom = (
                    #         int(left * width_ratio),
                    #         int(top * height_ratio),
                    #         int(right * width_ratio),
                    #         int(bottom * height_ratio),
                    #     )
                    #     cv2.rectangle(frame, (left, top), (right, bottom), class_colors[label], 2)
                    #     cv2.putText(
                    #         frame,
                    #         "{} [{:.2f}]".format(label, float(confidence)),
                    #         (left, top - 5),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.5,
                    #         class_colors[label],
                    #         2,
                    #     )
                    

                    cv2.imshow(window_title, frame)
                    keyCode = cv2.waitKey(10) & 0xFF
                    # Stop the program on the ESC key or 'q'
                    if keyCode == 27 or keyCode == ord('q'):
                        break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
