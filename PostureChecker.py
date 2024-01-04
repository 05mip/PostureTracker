import cv2
import mediapipe as mp
import numpy as np
import winsound

class PoseDetector:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(0)
        self.frequency = 2500
        self.duration = 250
        self.calibrating = False
        self.calibration_angles = []

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180:
            angle = 360 - angle

        return angle

    def calibration_process(self):
        self.calibration_angles = []
        for _ in range(90):  # Collect angles for 3 seconds
            ret, frame = self.cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                angle = self.calculate_angle(nose, right_shoulder, right_hip)
                self.calibration_angles.append(angle)

                visibles = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility, 
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility, 
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility]

                cv2.putText(image, "Calibrating...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except:
                pass

            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('c'):
                break

        avg_calibration_angle = np.mean(self.calibration_angles)
        calibrated_angle = max(0, avg_calibration_angle - 10)
        return calibrated_angle

    def run_pose_detection(self):
        calibrated_angle = 140
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as self.pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = self.pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark
                    nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
                    right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                    if self.calibrating:
                        calibrated_angle = self.calibration_process()
                        print(f"Calibrated angle: {calibrated_angle}")
                        self.calibrating = False

                    else:
                        angle = self.calculate_angle(nose, right_shoulder, right_hip)

                        visibles = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility, 
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility, 
                                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].visibility]

                        cv2.putText(image, f"Angle: {angle:.2f}", tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        if angle < calibrated_angle and all(visible > .5 for visible in visibles):
                            winsound.Beep(self.frequency, self.duration)

                except:
                    pass

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                cv2.imshow('Feed', image)

                # Check for window close event
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

                if cv2.waitKey(10) & 0xFF == ord('c'):
                    self.calibrating = True

            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    pose_detector = PoseDetector()
    pose_detector.run_pose_detection()
