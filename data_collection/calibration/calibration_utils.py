import numpy as np
import cv2
import cv2.aruco as aruco

def detect_aruco_corner_pixels(pic: np.ndarray, dictionary: int) -> tuple[list[np.ndarray], np.ndarray]:
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    # parameters.adaptiveThreshWinSizeMin = 3  # default 3
    # parameters.adaptiveThreshWinSizeMax = 700  # default 23
    # parameters.adaptiveThreshWinSizeStep = 5  # default 10
    # parameters.adaptiveThreshConstant = 10      # default 7
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, detected_ids, _ = detector.detectMarkers(gray)
    if detected_ids is None:
        return None, None
    return corners, detected_ids.flatten()
