
from pathlib import Path
import os
import joblib

import cv2
import mediapipe as mp

import numpy as np
import pandas as pd


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmark_to_index = {landmark.name.lower() : landmark.value for landmark in mp_pose.PoseLandmark}
index_to_landmark = {value: key for key, value in landmark_to_index.items()}



class PoseFeatureExtractor():
    def __init__(self, torso_size_multiplier=2.5):
        self._torso_size_multiplier = torso_size_multiplier
        
        self._landmark_to_index = {landmark.name.lower() : landmark.value for landmark in mp_pose.PoseLandmark}
        self._index_to_landmark = {value: key for key, value in landmark_to_index.items()}
        
    def __call__(self, landmarks):
        # Get pose landmarks.
        landmarks = np.copy(landmarks)#[:,:2]

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)
        
        # Get embedding.
        embedding = self._get_pose_embedding(landmarks)

        return embedding
        
    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks
    
    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_to_index['left_hip']]
        right_hip = landmarks[self._landmark_to_index['right_hip']]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_to_index['left_shoulder']]
        right_shoulder = landmarks[self._landmark_to_index['right_shoulder']]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)
    
    
    def _get_pose_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding as well as the
        angle between different joints

        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')), #"torso_length", 
                           

            # Two joints.
            #        "left_shoulder_wrist", "right_shoulder_wrist",
            #        "left_hip_ankle", "right_hip_ankle", 

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.
            #         "left_hip_wrist", "right_hip_wrist",
            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.
            #         "left_shoulder_ankle", "right_shoulder_ankle", 
            #         "left_hip_right_wrist", "right_hip_left_wrist"
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'left_wrist'),

            # Cross body.
            #          "left_right_elbow", "left_right_knee", 
            #          "left_right_wrist", "left_right_ankle",
            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.
            #          "left_wrist_ankle_from_left_hip", "right_wrist_ankle_from_right_hip", 
            self._get_distance(
                self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
                landmarks[self._landmark_to_index['left_hip']]),
            self._get_distance(
                self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
                landmarks[self._landmark_to_index['right_hip']]),

            # Angles
            #           "left_hip_shoulder_elbow_angle", "right_hip_shoulder_elbow_angle",
            #           "left_shoulder_elbow_wrist", "right_shoulder_elbow_wrist",
            #           "left_knee_hip_shoulder", "right_knee_hip_shoulder",
            #           "left_ankle_knee_hip", "right_ankle_knee_hip"
            self._get_angle_by_names(landmarks, 'left_hip', 'left_shoulder', 'left_elbow'),
            self._get_angle_by_names(landmarks, 'right_hip', 'right_shoulder', 'right_elbow'),

            self._get_angle_by_names(landmarks, 'left_shoulder', 'left_elbow', 'left_wrist'),
            self._get_angle_by_names(landmarks, 'right_shoulder', 'right_elbow', 'left_wrist'),

            self._get_angle_by_names(landmarks, 'left_knee', 'left_hip', 'left_shoulder'),
            self._get_angle_by_names(landmarks, 'right_knee', 'right_hip', 'right_shoulder'),

            self._get_angle_by_names(landmarks, 'left_ankle', 'left_knee', 'left_hip'),
            self._get_angle_by_names(landmarks, 'right_ankle', 'right_knee', 'right_hip'),

        ])

        return embedding
    
    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_to_index['left_hip']]
        right_hip = landmarks[self._landmark_to_index['right_hip']]
        center = (left_hip + right_hip) * 0.5
        return center
    
    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_to_index[name_from]]
        lmk_to = landmarks[self._landmark_to_index[name_to]]
        return (lmk_from + lmk_to) * 0.5
    
    def _get_angle_by_names(self, landmarks, lmk_a, lmk_b, lmk_c):
        vec_ab = landmarks[self._landmark_to_index[lmk_b]] - landmarks[self._landmark_to_index[lmk_a]]
        vec_bc = landmarks[self._landmark_to_index[lmk_c]] - landmarks[self._landmark_to_index[lmk_b]]

        cosine_angle = np.dot(vec_ab, vec_bc)/(np.linalg.norm(vec_ab)*np.linalg.norm(vec_bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_to_index[name_from]]
        lmk_to = landmarks[self._landmark_to_index[name_to]]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return np.linalg.norm(lmk_to - lmk_from)


def get_landmarks(filepath):
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        image = cv2.imread(filepath)
        
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        pose_landmarks = np.array(
                [[lmk.x * image_width, lmk.y * image_height, lmk.z * image_width]
                 for lmk in results.pose_landmarks.landmark],
                dtype=np.float32)
            
    return pose_landmarks


def predict_pose(lmodel, landmarks):
    embeddings = pose_embedder(landmarks)
    result = model.predict_proba(embeddings.reshape(1, -1))
    if result.max() < 0.70:
        return "Pose not clear"
    return code_to_category[result.argmax()]

def calculate_angle(results, lmk1, lmk2, lmk3):
    landmarks = results.pose_landmarks.landmark
    lmk1 = np.array([landmarks[landmark_to_index[lmk1]].x, landmarks[landmark_to_index[lmk1]].y])
    lmk2 = np.array([landmarks[landmark_to_index[lmk2]].x, landmarks[landmark_to_index[lmk2]].y])
    lmk3 = np.array([landmarks[landmark_to_index[lmk3]].x, landmarks[landmark_to_index[lmk3]].y])

    lmk21 = lmk1 - lmk2
    lmk23 = lmk3 - lmk2
    cosine_angle = np.dot(lmk21, lmk23) / (np.linalg.norm(lmk21) * np.linalg.norm(lmk23))
    angle = np.arccos(cosine_angle)
    return lmk2, np.degrees(angle)


if __name__=="__main__":
    pose_embedder = PoseFeatureExtractor()
    model = joblib.load("my_pipeline.pkl")

    train_df = pd.read_csv('pose_features_train.csv')

    category_to_code = {category: code for code, category in enumerate(train_df['category'].unique())}
    code_to_category = {code: category for category, code in category_to_code.items()}

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        model_complexity=2,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_height, image_width, _ = image.shape
            results = pose.process(image)
            if not results.pose_landmarks:
                continue

            pose_landmarks = np.array(
                    [[lmk.x * image_width, lmk.y * image_height, lmk.z * image_width]
                    for lmk in results.pose_landmarks.landmark],
                    dtype=np.float32)
            
            prediction = predict_pose(model, pose_landmarks)

            right_elbow, right_elbow_angle = calculate_angle(results, "right_shoulder", "right_elbow", "right_wrist")

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(image, (0,0), (image_width, 50), (250,206,135), -1)
            cv2.putText(image, f"Predicted Pose: {prediction}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 128), 2, cv2.LINE_AA  )

            cv2.putText(image, str(int(right_elbow_angle)), 
            tuple(np.multiply(right_elbow, [image_width, image_height]).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 2, 
            (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()