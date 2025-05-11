from typing import List, Tuple

import cv2
import numpy as np


class FaceDetectionService:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    # def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
    #     return [(x, y, x + w, y + h) for (x, y, w, h) in faces]

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Adjusted parameters: increased scaleFactor and minNeighbors
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [(x, y, x + w, y + h) for (x, y, w, h) in faces]