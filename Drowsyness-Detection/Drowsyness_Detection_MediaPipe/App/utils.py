import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp 
import  time
from pygame import mixer


mp_facemesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
denormalize_coords = mp_drawing._normalized_to_pixel_coordinates

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.
    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of the captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio 
    """
    try:
        # compute the eclidean distance between the horizontal
        coordinate_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coordinate = denormalize_coords(lm.x, lm.y,
                                            frame_width, frame_height)
            
            coordinate_points.append(coordinate)

        # Eye landmark (x, y) coordinates
        P2_P6 = distance(coordinate_points[1], coordinate_points[5])
        P3_P5 = distance(coordinate_points[2], coordinate_points[4])
        P1_P4 = distance(coordinate_points[2], coordinate_points[5])

        # Compute the eye aspect ratio

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coordinate_points = None
    
    return ear, coordinate_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)

    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear, (left_lm_coordinates, right_lm_coordinates)


def estimate_head_pose(landmarks, imgW, imgH):
    # 3D model points of key facial landmarks (relative positions)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0), # Left eye corner
        (225.0, 170.0, -135.0), # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # 2D image points from landmarks
    image_points = np.array([
        denormalize_coords(landmarks[1].x, landmarks[1].y, imgW, imgH),  # Nose tip
        denormalize_coords(landmarks[152].x, landmarks[152].y, imgW, imgH),  # Chin
        denormalize_coords(landmarks[263].x, landmarks[263].y, imgW, imgH),  # Left eye corner
        denormalize_coords(landmarks[33].x, landmarks[33].y, imgW, imgH),    # Right eye corner
        denormalize_coords(landmarks[61].x, landmarks[61].y, imgW, imgH),    # Left mouth corner
        denormalize_coords(landmarks[291].x, landmarks[291].y, imgW, imgH)   # Right mouth corner
    ], dtype="double")

    # Camera internals (assume no distortion for simplicity)
    focal_length = imgW
    center = (imgW / 2, imgH / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    return pitch, yaw, roll
