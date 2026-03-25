import argparse
import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import pygame


# FaceMesh landmark indices that work well for a simple head-pitch heuristic.
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1
CHIN = 152


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def parse_args():
    parser = argparse.ArgumentParser(description="RockLook: play rock music when you look down.")
    parser.add_argument("--music", default="rock.mp3", help="Path to an .mp3 file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (try 1 if 0 fails)")
    parser.add_argument("--calib-frames", type=int, default=30, help="Frames used to calibrate neutral pose")
    parser.add_argument("--threshold-offset", type=float, default=0.18, help="Extra offset added after calibration")
    parser.add_argument("--show-landmarks", action="store_true", help="Draw face mesh landmarks")
    return parser.parse_args()


def landmark_xy(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)


def down_score(landmarks, w, h):
    left_eye = landmark_xy(landmarks[LEFT_EYE_OUTER], w, h)
    right_eye = landmark_xy(landmarks[RIGHT_EYE_OUTER], w, h)
    nose = landmark_xy(landmarks[NOSE_TIP], w, h)
    chin = landmark_xy(landmarks[CHIN], w, h)

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    eye_dist = math.dist(left_eye, right_eye)
    eye_dist = max(eye_dist, 1.0)

    # Positive values generally increase as the head pitches downward.
    score = (nose[1] - eye_mid_y) / eye_dist
    return score, left_eye, right_eye, nose, chin


def main():
    args = parse_args()
    music_path = Path(args.music)

    if not music_path.exists():
        print(f"[ERROR] Music file not found: {music_path.resolve()}")
        print("Put an .mp3 file next to this script or pass --music path/to/file.mp3")
        sys.exit(1)

    pygame.mixer.init()
    pygame.mixer.music.set_volume(1.0)

    print("Loading music from:", music_path.resolve())
    print("Exists?", music_path.exists())

    pygame.mixer.music.load(str(music_path))
    print("Music loaded successfully ✅")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[WARN] Camera {args.camera} failed. Trying camera 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    play_state = False
    threshold = None
    calibration_scores = []
    printed_state = None

    print("[INFO] Press q to quit.")
    print("[INFO] Calibrating neutral head position... look straight ahead.")

    with face_mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            current_score = None
            status = "No face"

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                current_score, left_eye, right_eye, nose, chin = down_score(
                    face_landmarks.landmark, w, h
                )

                if threshold is None:
                    calibration_scores.append(current_score)
                    if len(calibration_scores) >= args.calib_frames:
                        baseline = sum(calibration_scores) / len(calibration_scores)
                        threshold = baseline + args.threshold_offset
                        print(f"[INFO] Calibration complete. Baseline={baseline:.3f}, threshold={threshold:.3f}")
                else:
                    # Hysteresis so audio does not flicker.
                    on_threshold = threshold
                    off_threshold = threshold - 0.06

                    looking_down = current_score >= on_threshold
                    looking_up = current_score <= off_threshold

                    if looking_down and not play_state:
                        pygame.mixer.music.play()
                        play_state = True
                        

                    elif looking_up and play_state:
                        pygame.mixer.music.pause()
                        play_state = False
                        

                    status = "LOOKING DOWN" if play_state else "UP / NEUTRAL"

                if args.show_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_style.get_default_face_mesh_contours_style(),
                    )

                # Draw debugging markers.
                cv2.circle(frame, left_eye, 4, (255, 255, 0), -1)
                cv2.circle(frame, right_eye, 4, (255, 255, 0), -1)
                cv2.circle(frame, nose, 5, (0, 255, 0), -1)
                cv2.circle(frame, chin, 5, (0, 0, 255), -1)

            # Overlay status text.
            y = 30
            cv2.putText(frame, "RockLook", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y += 30

            if threshold is None:
                cv2.putText(frame, f"Calibrating... {len(calibration_scores)}/{args.calib_frames}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 30
            else:
                cv2.putText(frame, f"Score: {current_score:.3f}" if current_score is not None else "Score: ---",(15, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 30
                cv2.putText(frame, f"Threshold: {threshold:.3f}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 30
                cv2.putText(frame, f"State: {status}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if play_state else (0, 0, 255), 2)
                y += 30

            cv2.putText(frame, "Press q to quit", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("RockLook", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if play_state:
        pygame.mixer.music.stop()

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
