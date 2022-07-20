import cv2
import numpy
import numpy.typing
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import yellow_filter
#TODO: refatorar pra ser um recognizer, o tracker faz depois da fusão dos sinais



@dataclass
class EllipseTrackerParams:
    yellow_filtering: YellowFilteringParams
    contour_size_threshold: int
    residue_thresh: float
    dist_threshold: float
    alpha: float

class EllipseTracker:
    def __init__(self, params : EllipseTrackerParams):
        self.last_center_candidates = None
        self.params = params
        self.best_center = None
    def __call__(self, new_image):
        is_ellipse, coef_list, centroid = \
            fit_ellipse(extract_contours(
                filter_yellow(new_image,
                    self.params.yellow_filtering),
                self.params.contour_size_threshold),
                self.params.residue_thresh
        )

        candidate_ellipse_centroids = [c for i, c in enumerate(centroid) if is_ellipse[i]]

        if self.last_center_candidates is not None and len(self.last_center_candidates) > 0:
            candidate_distances = []
            for c in candidate_ellipse_centroids:
                distances_to_last = [numpy.linalg.norm(c - last) for last in self.last_center_candidates]
                candidate_distances.append(min(distances_to_last))
            if len(candidate_distances) > 0:
                best_index = numpy.argmin(candidate_distances)
                self.last_center_candidates = candidate_ellipse_centroids
                if self.best_center is None:
                    self.best_center = candidate_ellipse_centroids[best_index]
                else:
                    self.best_center = self.params.alpha * candidate_ellipse_centroids[best_index] \
                                    + (1 - self.params.alpha) * self.best_center
                
                return self.best_center
        
        self.last_center_candidates = candidate_ellipse_centroids
        return None
    
    def draw_ellipse(self, new_image, radius):
        display_image = new_image.copy()
        center = self.__call__(new_image)
        if center is not None:
            cv2.circle(display_image, numpy.int32(center[0]), radius, (255, 255, 255), -1)
        return display_image