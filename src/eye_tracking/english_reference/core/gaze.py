"""Webcam frame -> six numbers describing where the eyes point."""

import cv2
import mediapipe as mp
import numpy as np

# Face mesh landmark ids. The iris ids (468+) only exist when the mesh is
# opened with refine_landmarks=True.
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)
LEFT_LID = (159, 145)
RIGHT_LID = (386, 374)
NOSE_TIP = 1
CHEEKS = (234, 454)
FOREHEAD = 10
CHIN = 152

FEATURE_NAMES = ["left_x", "left_y", "right_x", "right_y", "yaw", "pitch"]


def open_face_mesh():
    # Legacy solutions API. It is deprecated but still shipped in mediapipe
    # 0.10.x, and it is what the proposal names, so keep the pinned version in
    # requirements.txt. Newer builds ship only the Tasks API and this breaks.
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _point(landmarks, index, w, h):
    lm = landmarks[index]
    return np.array([lm.x * w, lm.y * h])


def _iris_offset(landmarks, w, h, corners, lid, iris):
    outer = _point(landmarks, corners[0], w, h)
    inner = _point(landmarks, corners[1], w, h)
    top = _point(landmarks, lid[0], w, h)
    bottom = _point(landmarks, lid[1], w, h)
    centre = _point(landmarks, iris, w, h)

    # divide by the eye's own size so the numbers stay comparable when the
    # reader leans in or out and the face gets bigger in frame
    eye_w = max(np.linalg.norm(inner - outer), 1e-6)
    eye_h = max(np.linalg.norm(bottom - top), 1e-6)
    mid = (outer + inner) / 2
    return (centre[0] - mid[0]) / eye_w, (centre[1] - (top[1] + bottom[1]) / 2) / eye_h


def head_pose(landmarks, w, h):
    left = _point(landmarks, CHEEKS[0], w, h)
    right = _point(landmarks, CHEEKS[1], w, h)
    top = _point(landmarks, FOREHEAD, w, h)
    bottom = _point(landmarks, CHIN, w, h)
    nose = _point(landmarks, NOSE_TIP, w, h)

    # a turned head moves the irises without the gaze moving, so the mapping
    # needs to know roughly how the head sits
    face_w = max(np.linalg.norm(right - left), 1e-6)
    face_h = max(np.linalg.norm(bottom - top), 1e-6)
    yaw = (nose[0] - (left[0] + right[0]) / 2) / face_w
    pitch = (nose[1] - (top[1] + bottom[1]) / 2) / face_h
    return yaw, pitch


def eye_vector(landmarks, w, h):
    lx, ly = _iris_offset(landmarks, w, h, LEFT_CORNERS, LEFT_LID, LEFT_IRIS)
    rx, ry = _iris_offset(landmarks, w, h, RIGHT_CORNERS, RIGHT_LID, RIGHT_IRIS)
    yaw, pitch = head_pose(landmarks, w, h)
    return np.array([lx, ly, rx, ry, yaw, pitch])


def frame_vector(mesh, frame):
    """Return the six numbers for one BGR frame, or None if no face was found."""
    h, w = frame.shape[:2]
    result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None
    return eye_vector(result.multi_face_landmarks[0].landmark, w, h)


def video_vectors(path):
    """Yield (frame_index, vector) for every frame of a recorded video.

    The vector is None when no face was found. Every frame is yielded, not
    just the good ones, so frames line up one-to-one with the timestamp CSV.
    """
    cap = cv2.VideoCapture(path)
    mesh = open_face_mesh()
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame_vector(mesh, frame)
            index += 1
    finally:
        cap.release()
        mesh.close()


def smooth(vectors, window=5):
    # A median, not an average. An average smears every jump between words
    # across the whole window, so saccades look slow and I-VT merges
    # neighbouring fixations into one. The median removes jitter but keeps
    # the jumps sharp.
    out = []
    for i in range(len(vectors)):
        chunk = vectors[max(0, i - window + 1) : i + 1]
        out.append(np.median(chunk, axis=0))
    return out
