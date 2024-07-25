import cv2
from keras.src.preprocessing.image import ImageDataGenerator

from tracker import *
from trafficSignal import *
from model import model


# Create tracker object
tracker = EuclideanDistTracker()

sourc = "vid1.mp4"
cap = cv2.VideoCapture(sourc)

# Object detection from Stable camera.
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
Vehical_count = 0

try:
    while True:
        ret, frm = cap.read()
        if not ret:
            print("Error: Unable to read frame from video source.")
            break  # Exit the loop if unable to read a frame

        height, width, _ = frm.shape
        print(frm.shape[1], "width", frm.shape[0], "Hight")

        frames = cv2.resize(frm, (780, 640), interpolation=cv2.INTER_NEAREST)

        # Traffic Signal Light       TrafficLight Function
        Yello, Green, Red, frame = trafficSignal.trafficLigh(frames)

        finalframe = frame
        if Red > Green and Red > Yello and Red > 30:
            print("Value Of Red in Signal > ", Red, "-")

            roi = frame[230:270, 100:800]

            # Traffic Line Function
            trafficSignal.Signalline(frame)

            # 1. Object Detection
            mask = object_detector.apply(roi)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            detections = []
            for cnt in cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
                area = cv2.contourArea(cnt)
                if area > 900:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    detections.append([x, y, w, h])

            print(len(detections), "Length of The detection ")

            # 2. Object Tracking    # EuclideanDistTracker Formula
            boxes_ids = tracker.update(detections)
            for x, y, w, h, id in boxes_ids:
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)
                if 195 < x < 205:
                    Vehical_count += 1

            cv2.putText(frame, f"Vehical Count: {Vehical_count}", (150, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (170, 10), (650, 50), (255, 255, 255), -1)
            cv2.putText(frame, f"Traffic Violation Count: {len(detections)}", (200, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 0, 255), 2)
            cv2.putText(frame, f"Violation{id}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            #cv2.imshow("Mask", mask)
            #cv2.imshow("roi", roi)
            #cv2.imshow("dilated", dilated)

        #cv2.imshow("Video _ Frames", frame)
        cv2.imshow("Final and Only Final Frame", finalframe)
        key = cv2.waitKey(30)
        if key == 27:
            break

    # Evaluate the model on the test set (replace 'path/to/your/test' with actual path)
    test_dir = 'test'
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

    cap.release()
    cv2.destroyAllWindows()
    print("----Video Ends---")
except Exception as e:
    print(f"An error occurred: {e}")


