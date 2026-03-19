import sys
import os
import tempfile

import cv2
import numpy as np
from itertools import combinations

class ImageProcessor:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.output_path = None

    def detect_aruco(self):
        aruco_dict = [
            cv2.aruco.DICT_4X4_50,
            # cv2.aruco.DICT_4X4_100,
            # cv2.aruco.DICT_4X4_250,
            # cv2.aruco.DICT_4X4_1000,
            # cv2.aruco.DICT_5X5_50,
            # cv2.aruco.DICT_5X5_100,
            # cv2.aruco.DICT_5X5_250,
            # cv2.aruco.DICT_5X5_1000,
            # cv2.aruco.DICT_6X6_50,
            # cv2.aruco.DICT_6X6_100,
            # cv2.aruco.DICT_6X6_250,
            # cv2.aruco.DICT_6X6_1000,    
            # cv2.aruco.DICT_7X7_50,
            # cv2.aruco.DICT_7X7_100,
            # cv2.aruco.DICT_7X7_250,
            # cv2.aruco.DICT_7X7_1000,
        ]
        parameters = cv2.aruco.DetectorParameters()
        found_bounding_boxes = []
        
        self.image = cv2.imread(self.image_path)
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_01_orig.jpg")
        cv2.imwrite(self.output_path, self.image)

        image_copy = self.image.copy()

        for dict in aruco_dict:
            # We only want to find 4 markers, so we can stop once we find 4
            if len(found_bounding_boxes) == 4:
                return found_bounding_boxes

            predefined_dict = cv2.aruco.getPredefinedDictionary(dict)
            detector = cv2.aruco.ArucoDetector(predefined_dict, parameters)
            bounding_boxes, ids, _ = detector.detectMarkers(image_copy)

            if bounding_boxes is not None and ids is not None:
                for id, box in zip(ids, bounding_boxes):
                    cv2.polylines(image_copy, [box.astype(int)], True, (0, 255, 0), 2)
                found_bounding_boxes.extend(bounding_boxes)

        if len(found_bounding_boxes) == 4:
            self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_02_marked.jpg")
            cv2.imwrite(self.output_path, image_copy)

        return found_bounding_boxes

    def crop_image(self, box_corners):
        image_copy = self.image.copy()
        centers = []
        max_width = 0
        max_height = 0
        
        for box in box_corners:
            box_corners = box[0]
            center_x = int(box_corners[:, 0].mean())
            center_y = int(box_corners[:, 1].mean())

            width = int(box_corners[:, 0].max() - box_corners[:, 0].min())
            height = int(box_corners[:, 1].max() - box_corners[:, 1].min())

            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

            centers.append((center_x, center_y))

        min_x = min(centers, key=lambda x: x[0])[0] + max_width / 2
        max_x = max(centers, key=lambda x: x[0])[0] - max_width / 2
        min_y = min(centers, key=lambda x: x[1])[1] + max_height / 2
        max_y = max(centers, key=lambda x: x[1])[1] - max_height / 2

        centers_np = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], dtype="float32")

        rect = cv2.convexHull(centers_np)
        rect = np.array(rect, dtype="float32")

        width = int(max_x - min_x)
        height = int(max_y - min_y)
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)

        cropped = cv2.warpPerspective(image_copy, M, (width, height))
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_03_cropped.jpg")
        cv2.imwrite(self.output_path, cropped)
        self.image = cropped.copy()

    def gray_image(self):
        image_copy = self.image.copy()

        grayed = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_04_grayed.jpg")
        cv2.imwrite(self.output_path, grayed)
        self.image = grayed.copy()

    def blurr_image(self):
        image_copy = self.image.copy()

        ret, thresh = cv2.threshold(image_copy, 130, 255, 0) #130 is the threshold value, 0-255, I found it via experimentation
        blurred = cv2.GaussianBlur(thresh, (5, 5), 0)
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_05_blurred.jpg")
        cv2.imwrite(self.output_path, blurred)
        self.image = blurred.copy()

    def edge_image(self):
        image_copy = self.image.copy()

        edges = cv2.Canny(image_copy, 50, 150)
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_06_edges.jpg")
        cv2.imwrite(self.output_path, edges)
        self.image = edges.copy()

    def trim_image(self):
        image_copy = self.image.copy()

        contours, _ = cv2.findContours(image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        max_area = 0
        best_quad = None
        for quad in combinations(approx, 4):
            quad = np.array(quad, dtype="float32")
            area = cv2.contourArea(quad)
            if area > max_area:
                max_area = area
                best_quad = quad

        if best_quad is None:
            self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_01_orig.jpg")
            return

        points = np.squeeze(best_quad)
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)] 
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]

        top_width =  rect[1][0] - rect[0][0] 
        bottom_width = rect[3][0] - rect[2][0]

        left_height = rect[0][1] - rect[3][1]
        right_height = rect[2][1] - rect[1][1]

        width = int(max(top_width, bottom_width))
        height = int(max(left_height, right_height))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(rect, dst)

        trimmed = cv2.warpPerspective(image_copy, matrix, (width, height))
        self.output_path = os.path.join(tempfile.gettempdir(), os.path.basename(self.image_path) + "_07_trimmed.jpg")
        cv2.imwrite(self.output_path, trimmed)
        self.image = trimmed.copy()

    def process_image(self):
        box_corners = self.detect_aruco()

        if box_corners is not None and len(box_corners) == 4:
            self.crop_image(box_corners)
            self.gray_image()
            self.blurr_image()
            self.edge_image()
            self.trim_image()
        return self.output_path

def main(image_path):
    processor = ImageProcessor(image_path)
    processor.process_image()

# __name__
if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_processor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    main(image_path)