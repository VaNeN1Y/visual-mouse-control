import cv2
import mediapipe as mp
import time
import math


class DetectHand:
    """
    Класс для обнаружения и отслеживания рук с использованием библиотеки Mediapipe.

    Методы:
        - __init__: Инициализация параметров детектора.
        - find_hands: Поиск рук на изображении.
        - find_position: Находит координаты ключевых точек руки.
        - fingers_up: Определяет, какие пальцы подняты.
        - find_distance: Вычисляет расстояние между двумя точками.
    """

    def __init__(self):
        """
        Инициализация параметров.
        """
        self.mode = False
        self.max_hands = 1
        self.detection_con = 0.5
        self.track_con = 0.5

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        """
        Поиск рук на изображении и отрисовка ключевых точек.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_num=0, draw=True):
        """
        Находит координаты всех ключевых точек руки и рисует рамку вокруг руки.
        """
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_num]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

        return self.lm_list, bbox

    def fingers_up(self):
        """
        Определяет, какие пальцы подняты.
        """
        fingers = []

        if len(self.lm_list) >= max(self.tip_ids) + 1:
            for i in range(5):
                if self.lm_list[self.tip_ids[i]][2] < self.lm_list[self.tip_ids[i] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
        else:
            return [0, 0, 0, 0, 0]

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Вычисляет расстояние между двумя точками на руке.
        """
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 127, 80), t)
            cv2.circle(img, (x1, y1), r, (255, 127, 80), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 127, 80), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 127, 80), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    """
    Основная функция, которая запускает захват изображения с камеры и обработку в реальном времени.
    """
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = DetectHand()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list, bbox = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[16])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break


if __name__ == "__main__":
    main()
