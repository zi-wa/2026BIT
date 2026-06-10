"""
BIT Hand - Finger Bend Tracker (양손 지원)
mediapipe 0.10+ Tasks API 사용 — 로컬 CPU에서 완전히 동작합니다.
최초 실행 시 모델 파일(~8 MB)을 자동 다운로드합니다.

필요 패키지:
    pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import time
import os
import urllib.request

# ──────────────────────────────────────────
#  설정 변수 (필요에 따라 조절)
# ──────────────────────────────────────────
REFRESH_RATE_FPS = 30       # 목표 Refresh Rate (FPS)
CAMERA_INDEX     = 0        # 카메라 인덱스 (0: 기본 웹캠)
MAX_NUM_HANDS    = 2        # 인식할 최대 손 개수 (1 또는 2)

# 모델 파일 (최초 실행 시 자동 다운로드, 이후 로컬 캐시 사용)
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "hand_landmarker.task")

# UI 색상
COLOR_BG         = (255,255,255)
COLOR_TITLE      = (200, 230, 255)
COLOR_LABEL      = (180, 180, 200)
COLOR_VALUE      = (100, 220, 180)
COLOR_BAR_BG     = (50, 50, 65)
COLOR_BAR_FG     = (80, 200, 160)
COLOR_BAR_HIGH   = (60, 120, 220)
COLOR_DELAY_OK   = (100, 220, 150)
COLOR_DELAY_WARN = (60, 170, 255)
COLOR_DELAY_BAD  = (80, 100, 255)
COLOR_HAND_LEFT  = (255, 100, 100)   # 왼손 헤더 — 황금색
COLOR_HAND_RIGHT = (100, 100, 255)   # 오른손 헤더 — 파란색
COLOR_ABSENT     = (70, 70, 90)     # 미감지

# 스켈레톤 색 (왼손 / 오른손): (선, 관절)
SKEL_COLORS = {
    "Left":  ((100,  100, 255), (100,  100, 255)),
    "Right": ((255,  100, 100), (255,  100, 100)),
}

# 레이아웃
PANEL_WIDTH   = 720
BAR_HEIGHT    = 10
BAR_MAX_WIDTH = 360
HAND_ROW_H    = 25
FONT          = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────
#  손가락 이름 / 랜드마크 인덱스
# ──────────────────────────────────────────
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

FINGER_LANDMARKS = {
    "Thumb":  [1,  2,  3,  4],
    "Index":  [5,  6,  7,  8],
    "Middle": [9,  10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20],
}

# 스켈레톤 연결 (고정값 — mediapipe 버전 무관)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ──────────────────────────────────────────
#  모델 파일 관리
# ──────────────────────────────────────────

def ensure_model() -> None:
    """hand_landmarker.task 파일이 없으면 자동 다운로드."""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"[INFO] 모델 파일 확인됨 ({size_mb:.1f} MB): {MODEL_PATH}")
        return

    print("[INFO] 모델 파일(~8 MB)을 다운로드합니다. 최초 1회만 필요합니다.")

    def _progress(block_num, block_size, total_size):
        pct = min(block_num * block_size / total_size * 100, 100)
        print(f"\r       진행: {pct:5.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=_progress)
        print(f"\n[INFO] 다운로드 완료 → {MODEL_PATH}")
    except Exception as exc:
        print(f"\n[ERROR] 다운로드 실패: {exc}")
        raise

# ──────────────────────────────────────────
#  수학 유틸리티
# ──────────────────────────────────────────

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def compute_finger_bend(lm_np: np.ndarray, finger_name: str) -> float:
    """손가락 굽힘 정도 (0.0 = 완전 펴짐, 1.0 = 완전 굽힘)."""
    MAX_FLEX_DEG = 90.0
    idxs = FINGER_LANDMARKS[finger_name]
    pts  = [lm_np[i] for i in idxs]

    flex_angles = []
    for i in range(len(pts) - 2):
        ang = angle_between(pts[i] - pts[i + 1], pts[i + 2] - pts[i + 1])
        flex_angles.append(max(0.0, 180.0 - ang))

    if not flex_angles:
        return 0.0
    return min(float(np.mean(flex_angles)) / MAX_FLEX_DEG, 1.0)


def bend_to_label(bend: float) -> str:
    if bend < 0.25:   return "Straight"
    elif bend < 0.55: return "Slightly"
    elif bend < 0.80: return "Bent"
    else:             return "Closed"


def delay_color(ms: float) -> tuple:
    if ms < 33:   return COLOR_DELAY_OK
    elif ms < 66: return COLOR_DELAY_WARN
    else:         return COLOR_DELAY_BAD

# ──────────────────────────────────────────
#  렌더링 유틸리티
# ──────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    if x2 <= x1 or y2 <= y1:
        return
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, thickness)


def draw_hand_section(panel, y: int, side: str,
                      bend_data, header_color: tuple) -> int:
    """손 1개의 게이지 섹션을 그리고 다음 y 좌표를 반환."""
    pw = panel.shape[1]

    cv2.putText(panel, f"< {side} Hand >", (16, y),
                FONT, 0.50, header_color, 1, cv2.LINE_AA)
    y += 6

    if bend_data is None:
        cv2.putText(panel, "  -- not detected --", (16, y + 18),
                    FONT, 0.42, COLOR_ABSENT, 1, cv2.LINE_AA)
        return y + HAND_ROW_H * 2

    for fname in FINGER_NAMES:
        bend  = bend_data.get(fname, 0.0)
        y    += HAND_ROW_H

        cv2.putText(panel, fname, (16, y),
                    FONT, 0.43, COLOR_LABEL, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{bend * 100:.0f}%", (pw - 46, y),
                    FONT, 0.40, COLOR_VALUE, 1, cv2.LINE_AA)

        bx, by = 16, y + 3
        draw_rounded_rect(panel, bx, by, bx + BAR_MAX_WIDTH, by + BAR_HEIGHT,
                          BAR_HEIGHT // 2, COLOR_BAR_BG, -1)
        fill_w = max(int(BAR_MAX_WIDTH * bend), 0)
        if fill_w > 0:
            c = COLOR_BAR_HIGH if bend > 0.75 else COLOR_BAR_FG
            draw_rounded_rect(panel, bx, by, bx + fill_w, by + BAR_HEIGHT,
                              BAR_HEIGHT // 2, c, -1)
        cv2.putText(panel, bend_to_label(bend),
                    (bx + BAR_MAX_WIDTH + 5, by + BAR_HEIGHT - 2),
                    FONT, 0.33, COLOR_VALUE, 1, cv2.LINE_AA)

    return y + 8


def draw_panel(panel, hands_data: dict, fps: float, delay_ms: float) -> None:
    panel[:] = COLOR_BG
    pw, ph = panel.shape[1], panel.shape[0]

    cv2.putText(panel, "BIT Hand Tracker", (16, 28),
                FONT, 0.62, COLOR_TITLE, 2, cv2.LINE_AA)
    cv2.line(panel, (16, 36), (pw - 16, 36), (60, 60, 80), 1)

    cv2.putText(panel, f"Target FPS : {REFRESH_RATE_FPS}", (16, 54),
                FONT, 0.43, COLOR_LABEL, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Actual FPS : {fps:.1f}", (16, 70),
                FONT, 0.43, COLOR_VALUE, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Delay      : {delay_ms:.1f} ms", (16, 86),
                FONT, 0.43, delay_color(delay_ms), 1, cv2.LINE_AA)
    cv2.line(panel, (16, 96), (pw - 16, 96), (60, 60, 80), 1)

    y = draw_hand_section(panel, 114, "LEFT",
                          hands_data.get("Left"), COLOR_HAND_LEFT)
    cv2.line(panel, (16, y + 2), (pw - 16, y + 2), (60, 60, 80), 1)
    draw_hand_section(panel, y + 16, "RIGHT",
                      hands_data.get("Right"), COLOR_HAND_RIGHT)

    cv2.line(panel, (16, ph - 26), (pw - 16, ph - 26), (60, 60, 80), 1)
    cv2.putText(panel, "[Q / ESC] Quit", (16, ph - 10),
                FONT, 0.38, (100, 100, 120), 1, cv2.LINE_AA)


def draw_skeleton(frame, lm_np: np.ndarray, hand_label: str) -> None:
    col_line, col_joint = SKEL_COLORS.get(hand_label, SKEL_COLORS["Right"])
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(lm_np[a][:2].astype(int)),
                 tuple(lm_np[b][:2].astype(int)), col_line, 2, cv2.LINE_AA)
    for pt in lm_np:
        cx, cy = int(pt[0]), int(pt[1])
        cv2.circle(frame, (cx, cy), 5, col_joint, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)

# ──────────────────────────────────────────
#  메인 루프
# ──────────────────────────────────────────

def main():
    ensure_model()

    # Tasks API 초기화 — 로컬 CPU 추론
    BaseOptions           = mp_python.BaseOptions
    HandLandmarker        = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    VisionRunningMode     = mp_vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=BaseOptions.Delegate.CPU,   # 로컬 CPU
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] 카메라 인덱스 {CAMERA_INDEX}를 열 수 없습니다.")
        landmarker.close()
        return
    cap.set(cv2.CAP_PROP_FPS, REFRESH_RATE_FPS)

    target_interval  = 1.0 / REFRESH_RATE_FPS
    prev_time        = time.perf_counter()
    fps_display      = 0.0
    delay_ms_display = 0.0
    ts_origin        = int(time.perf_counter() * 1000)   # 타임스탬프 기준점

    print(f"[INFO] BIT Hand Tracker 시작 (로컬 CPU, 목표 FPS: {REFRESH_RATE_FPS})")
    print("[INFO] [Q / ESC] 종료")

    while True:
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)      # 거울 모드
        h, w  = frame.shape[:2]

        # ── 추론 ───────────────────────────
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms    = int(time.perf_counter() * 1000) - ts_origin
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        hands_data = {"Left": None, "Right": None}

        if result.hand_landmarks and result.handedness:
            for raw_lms, handedness in zip(result.hand_landmarks,
                                           result.handedness):
                label = handedness[0].display_name   # "Left" 또는 "Right"

                lm_np = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]
                    for lm in raw_lms
                ])

                hands_data[label] = {
                    fname: compute_finger_bend(lm_np, fname)
                    for fname in FINGER_NAMES
                }
                draw_skeleton(frame, lm_np, label)

        # ── 패널 & 합성 ────────────────────
        panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
        draw_panel(panel, hands_data, fps_display, delay_ms_display)
        cv2.imshow("BIT Hand Tracker", np.hstack([frame, panel]))

        # ── FPS / Delay 갱신 ───────────────
        now              = time.perf_counter()
        elapsed          = now - prev_time
        fps_display      = 1.0 / elapsed if elapsed > 0 else 0.0
        delay_ms_display = elapsed * 1000.0
        prev_time        = now

        # ── 대기 & 키 입력 ─────────────────
        loop_elapsed = time.perf_counter() - loop_start
        wait_ms      = max(1, int((target_interval - loop_elapsed) * 1000))
        key          = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] 종료되었습니다.")


if __name__ == "__main__":
    main()
