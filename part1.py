import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import os
import telebot
import time

# Telegram Bot Token
TELEGRAM_TOKEN = '6562636784:AAGXC4za291SB8vyvVja6WPQvCxMqRcbNqk'

# Create a Telegram bot instance
bot = telebot.TeleBot(TELEGRAM_TOKEN)

def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    ap.add_argument('-f', '--file', help='Path to video file (if not using camera)')
    ap.add_argument('-col', '--color', type=str, default='rgb', help='Color space: "gray", "rgb" (default), or "lab"')
    ap.add_argument('-b', '--bins', type=int, default=16, help='Number of bins per channel (default 16)')
    ap.add_argument('-w', '--width', type=int, default=0, help='Resize video to specified width in pixels (maintains aspect)')
    ap.add_argument('-ct', '--color_threshold', type=int, default=100, help='Color tampering threshold: show warning if a certain color exceeds this count')
    args = vars(ap.parse_args())
    return args

def generate_bounding_boxes(layerOutputs, W, H, conf_thresh):
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def paint_box_on_object(frame, box, color, label, confidence):
    x, y, w, h = box[0], box[1], box[2], box[3]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(label, confidence)
    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def run_combined_detection(args, W, H, video_capture):
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    WEIGHTS_PATH = os.path.sep.join([args["yolo"], "yolov3.weights"])
    CONFIG_PATH = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    CONFIDENCE = args["confidence"]
    THRESHOLD = args["threshold"]

    print("[INFO] running combined detection...")
    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

    try:
        ln = net.getUnconnectedOutLayersNames()
    except Exception as e:
        print(f"Error: {e}")
        print("Unable to get unconnected output layers names. Check your YOLO model configuration.")
        return
    
    # Create a Matplotlib figure and axis for the histogram
    fig_hist, ax_hist = plt.subplots()

    tampering_detected = False

    #plt.figure(figsize=(8, 4))  # Initialize the histogram figure

    while video_capture.isOpened():
        (ret, frame) = video_capture.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

        if not ret:
            break

        H, W = frame.shape[:2]

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes, confidences, classIDs = generate_bounding_boxes(layerOutputs, W, H, CONFIDENCE)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

        # Histogram analysis
        histogram = cv2.calcHist([frame], [0], None, [args['bins']], [0, 256])
        histogram = (histogram / 1000).flatten()

        ax_hist.clear()
        ax_hist.plot(histogram, color='b')
        ax_hist.set_title('Histogram')
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Frequency(*10000)')

        if np.any(histogram > args['color_threshold']):
            tampering_detected = True
            warning_text = "Tampering Detected: Excessive Color"
            cv2.putText(frame, warning_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Save the tampered frame as an image
            cv2.imwrite("tampered_frame.jpg", frame)
            # Send message to Telegram
            bot.send_message(chat_id="1946827206", text="Tampering Detected! Image Attached.")
            # Send the tampered image
            bot.send_photo(chat_id="1946827206", photo=open("tampered_frame.jpg", "rb"))
        else:
            tampering_detected = False

        if len(idxs) > 0:
            for i in idxs.flatten():
                color = [int(c) for c in COLORS[classIDs[i]]]
                label = LABELS[classIDs[i]]
                confidence = confidences[i]
                paint_box_on_object(frame, boxes[i], color, label, confidence)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the Matplotlib figure
        fig_hist.canvas.draw()
        plt.pause(0.01)
        
    print("[INFO] cleaning up...")
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    args = argsParser()
    video_capture = cv2.VideoCapture(0) if not args.get('file', False) else cv2.VideoCapture(args['file'])
    _, frame = video_capture.read()
    H, W = frame.shape[:2]
    run_combined_detection(args, W, H, video_capture)  # Pass the args and video_capture

if __name__ == '__main__':
    main()
