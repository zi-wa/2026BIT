# 미디어파이 핸즈

import cv2
import mediapipe as mp

# 미디어 파이프의 Hand 모델을 로드합니다.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()  # 변수 이름을 'frame'으로 변경
        if not success:
            continue
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

        # 프레임을 미디어 파이프에 전달합니다.
        results = hands.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 랜드마크 좌표를 화면에 그립니다.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('frame', frame)  # 창 이름도 'frame'으로 변경

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()