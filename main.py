import cv2
import numpy as np
import TrackModule as tm
import time
import autopy


def right_click():
    """Функция для клика правой кнопкой мыши."""
    autopy.mouse.click(autopy.mouse.Button.RIGHT)


cap = cv2.VideoCapture(0)
w_cam, h_cam = 640, 480
cap.set(3, w_cam)
cap.set(4, h_cam)

detector = tm.DetectHand()

w_scr, h_scr = autopy.screen.size()

frame = 100

while not cv2.waitKey(1) & 0xFF == ord('0'):

    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list, bbox = detector.find_position(img)

    if len(lm_list) != 0:
        x1, y1 = lm_list[8][1:]

    fingers = detector.fingers_up()

    cv2.rectangle(img, (frame, frame), (w_cam - frame, h_cam - frame), (255, 0, 0), 2)

    if fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 0:
        x4 = np.interp(x1, (frame, w_cam - frame), (0, w_scr))
        y4 = np.interp(y1, (frame, h_cam - frame), (0, h_scr))

        x4 = min(max(0, x4), w_scr - 1)
        y4 = min(max(0, y4), h_scr - 1)

        x4_inverted = w_scr - x4
        x4_inverted = min(max(0, x4_inverted), w_scr - 1)

        autopy.mouse.move(x4_inverted, y4)

    elif fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 0:
        length, img, line_info = detector.find_distance(8, 12, img)

        if length < 40:
            cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            autopy.mouse.click()
            time.sleep(0.1)

    elif fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 1:
        length, img, line_info = detector.find_distance(8, 20, img)

        if length < 50:
            cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
            right_click()
            time.sleep(0.1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
