from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from skimage import measure


@dataclass
class Box:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __lt__(self, other) -> bool:
        return self.area < other.area

    def apply(self, img: np.ndarray):
        return img[self.y_min : self.y_max, self.x_min : self.x_max]

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def area(self):
        return self.width * self.height

    @property
    def is_valid(self):
        if 250 < self.width and 50 < self.height:
            return True
        return False

    def draw(self, img: np.ndarray):
        out_img = img.copy()
        cv2.rectangle(out_img, (self.x_min, self.y_min), (self.x_max, self.y_max), (255, 0, 0), 10)
        return out_img

    def iou(self, other) -> float:
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x_min, other.x_min)
        yA = max(self.y_min, other.y_min)
        xB = min(self.x_max, other.x_max)
        yB = min(self.y_max, other.y_max)

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(self.area + other.area - interArea)

        # return the intersection over union value
        return iou


def get_box_coords_from_contours(contour: List[List[Tuple[int, int]]]) -> Tuple[int, int, int, int]:
    ((x, y), (width, height), angle_of_rotation) = cv2.minAreaRect(contour)
    # Get box
    if height > width:
        height, width = width, height
    x_min = x - width // 2
    x_max = x_min + width
    y_min = y - height // 2
    y_max = y_min + height

    return x_min, y_min, x_max, y_max


def get_channel_thr_bbox(
    area_to_search: np.ndarray,
    h_added: int,
    value_thr: int,
    channel_idx: int,
    inverse: bool = False,
    area_thr: int = 100,
) -> Box:
    # Get some channel
    channel_img = cv2.cvtColor(area_to_search, cv2.COLOR_RGB2HSV)[:, :, channel_idx]
    if inverse:
        channel_img = 255 - channel_img

    # This may not work for all of the images
    # CAREFUL: check with all images
    gray = ((channel_img > value_thr) * 255).astype(np.uint8)

    # Leave only large connected components
    labeled = measure.label(gray)
    props = measure.regionprops(labeled)
    for prop in props:
        if prop.area < area_thr:
            labeled[labeled == prop.label] = 0
    gray = (labeled > 0).astype(np.uint8)

    # Make morphology closing ang opening
    kernel = np.ones((50, 50), dtype=np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = np.expand_dims(gray, 2)

    # Find all contours on image
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find max area bounding box
    max_area = 0
    max_box = Box(0, 0, 0, 0)
    for contour in contours:
        x_min, y_min, x_max, y_max = get_box_coords_from_contours(contour)
        box = Box(*map(int, [x_min, y_min + h_added, x_max, y_max + h_added]))
        if box.area > max_area and box.is_valid:
            max_area = box.area
            max_box = box

    return max_box


def get_score_bbox(image: np.ndarray, cut_corner: bool = True) -> Box:
    # Search box in bottom left area
    im_h, im_w = image.shape[:2]
    if cut_corner:
        h_added = int(im_h * 0.8)
        area_to_search = image[h_added:, : int(im_w * 0.4)]
    else:
        h_added = 0
        area_to_search = image.copy()

    boxes = []
    for value_thr, channel_idx, inverse in zip([100, 210], [0, 2], [False, True]):
        box = get_channel_thr_bbox(
            area_to_search, h_added=h_added, value_thr=value_thr, channel_idx=channel_idx, inverse=inverse
        )
        if box.is_valid:
            boxes.append(box)

    if boxes:
        box = min(boxes)

    return box
