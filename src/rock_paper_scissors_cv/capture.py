import cv2
import numpy as np
import os
import time
from pathlib import Path

CAPTURE_INTERVAL_MS = 250
TRAIN_DATA_FOLDER_NAME = "train"
VALIDATION_DATA_FOLDER_NAME = "validation"


def current_milli_time():
    return int(round(time.time() * 1000))


# Double texts for better contrast
def add_texts(frame, is_capturing, capture_type):
    font = cv2.FONT_HERSHEY_SIMPLEX
    black, white = (0, 0, 0), (255, 255, 255)

    cv2.putText(frame, "Hotkeys to start capturing: 'a' - Rock, 's' - Scissors, 'd' - Paper", (20, 20), font, 0.5,
                black, 3, cv2.LINE_AA)
    cv2.putText(frame, "Hotkeys to start capturing: 'a' - Rock, 's' - Scissors, 'd' - Paper", (20, 20), font, 0.5,
                white, 1, cv2.LINE_AA)

    cv2.putText(frame, "Is Capturing: {}".format(is_capturing), (20, 40), font, 0.5, black, 3, cv2.LINE_AA)
    cv2.putText(frame, "Is Capturing: {}".format(is_capturing), (20, 40), font, 0.5, white, 1, cv2.LINE_AA)
    if is_capturing:
        cv2.putText(frame, "Now capturing: {}".format(capture_type), (20, 60), font, 0.5, black, 3, cv2.LINE_AA)
        cv2.putText(frame, "Now capturing: {}".format(capture_type), (20, 60), font, 0.5, white, 1, cv2.LINE_AA)


def create_folders(capture_type):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(current_dir, TRAIN_DATA_FOLDER_NAME, capture_type)
    validation_path = os.path.join(current_dir, VALIDATION_DATA_FOLDER_NAME, capture_type)
    Path(train_path).mkdir(parents=True, exist_ok=True)
    Path(validation_path).mkdir(parents=True, exist_ok=True)
    #os.chdir(path)


def start_capture():
    cap = cv2.VideoCapture(0)

    last_capture_time = 0
    is_capturing = False
    capture_type = 'Unknown'
    capture_count_dict = {}
    data_type = TRAIN_DATA_FOLDER_NAME

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

        if is_capturing and current_milli_time() - last_capture_time > CAPTURE_INTERVAL_MS:
            last_capture_time = current_milli_time()
            if capture_type not in capture_count_dict:
                capture_count_dict[capture_type] = 0
            capture_count_dict[capture_type] += 1
            resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            height, width, channels = gray.shape
            cv2.rectangle(gray, (0, 0), (width, height), (0, 0, 255), 50)
            # Save 25% of data as validation data
            if capture_count_dict[capture_type] % 4 == 0:
                data_type = VALIDATION_DATA_FOLDER_NAME
            else:
                data_type = TRAIN_DATA_FOLDER_NAME

            os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), data_type, capture_type))
            cv2.imwrite("{}_{}.jpg".format(capture_type, capture_count_dict[capture_type]), resized)

        add_texts(gray, is_capturing, capture_type)

        cv2.imshow('Dataset', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(10) & 0xFF == ord('a'):
            print('a Pressed')
            is_capturing = True
            capture_type = 'rock'
            create_folders(capture_type)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            print('sas Pressed')
            is_capturing = True
            capture_type = 'scissors'
            create_folders(capture_type)

        if cv2.waitKey(10) & 0xFF == ord('d'):
            print('d Pressed')
            is_capturing = True
            capture_type = 'paper'
            create_folders(capture_type)

    cap.release()
    cv2.destroyAllWindows()


def main():
    start_capture()


if __name__ == '__main__':
    main();