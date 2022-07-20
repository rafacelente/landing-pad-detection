import cv2
import numpy
import numpy.typing
from dataclasses import dataclass
from typing import List, Callable
import time

@dataclass
class TrackbarParams:
    name : str
    init_val : int
    max_val : int
    callback : Callable

# Função auxiliar p vídeo
#fazer pausa em 1 frame
def test_with_video(processing_func, params, video_source = 0, trackbar_params=[]):
    video = cv2.VideoCapture(video_source)
    win = cv2.namedWindow('video')
    tb_win = cv2.namedWindow("Trackbars")
    #evita mudar o argumento default
    tb_param_list = trackbar_params.copy()
    for tb_p in tb_param_list:
        cv2.createTrackbar(tb_p.name, "Trackbars", tb_p.init_val, tb_p.max_val, tb_p.callback)
    paused = False
    ret = True
    while True:
        if not paused:
            ret, frame = video.read()
        if ret:
            result_image = processing_func(frame, params)
        cv2.imshow('video', result_image)

        #interação
        key = cv2.waitKey(1)
        if key & 0xFF == ord('p'):
            paused = not paused
        if key & 0xFF == ord('n'):
            ret, frame = video.read()
        if key & 0xFF == ord('q'):
            break
        time.sleep(0.01)

    video.release()
    cv2.destroyAllWindows()

#######################################################################################
#                       FIltragem do amarelo                                          #
#######################################################################################

@dataclass
class YellowFilterParams:
    #passar blur pra etapa própria
    # kernel_size: int
    # std_dev: int
    yellow_min: numpy.ndarray
    yellow_max: numpy.ndarray
    def set_yellow_min_h(self, val):
        self.yellow_min[0] = val
    def set_yellow_min_s(self, val):
        self.yellow_min[1] = val
    def set_yellow_min_v(self, val):
        self.yellow_min[2] = val
    def set_yellow_max_h(self, val):
        self.yellow_max[0] = val
    def set_yellow_max_s(self, val):
        self.yellow_max[1] = val
    def set_yellow_max_v(self, val):
        self.yellow_max[2] = val
    # def set_kernel_size(self, val):
    #     if val % 2 == 0:
    #         return
    #     self.kernel_size = val
    # def set_std_dev(self, val):
    #     self.std_dev = val  

def filter_yellow(rgb_img: cv2.Mat, params: YellowFilterParams) -> cv2.Mat:
    # img = cv2.GaussianBlur(img, (params.kernel_size, params.kernel_size), params.std_dev)
    img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV_FULL)
    yellow_min = params.yellow_min  # valores HSV mínimos
    yellow_max = params.yellow_max  # valroes HSV máximos
    yellow_region = cv2.inRange(img, yellow_min, yellow_max)
    return yellow_region

#função de display para debug
def display_yellow(rgb_img: cv2.Mat, params: YellowFilterParams):
    yellow_region = filter_yellow(rgb_img, params)
    return yellow_region

##################################################################################################
#                                   Extração de contornos                                        #
##################################################################################################

@dataclass
class ContourExtractParams:
    size_threshold: int
    def set_size_threshold(self, val):
        self.size_threshold = val

def extract_contours(yellow_region : cv2.Mat, params : ContourExtractParams) -> tuple:
    #fazer conversão certa
    yellow_region_u8 = yellow_region.astype(numpy.uint8)
    #retornar informação de hierarquia p identificação do + tbm?
    yellow_contours, hierarchy = cv2.findContours(yellow_region_u8, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    return tuple(filter(lambda cont: len(cont) > params.size_threshold, yellow_contours))

#função de display para debug
def display_contours(rgb_img : cv2.Mat, params):
    yellow_region = filter_yellow(rgb_img, params[0])
    contours = extract_contours(yellow_region, params[1])
    display_image = numpy.zeros_like(yellow_region)
    cv2.drawContours(display_image, contours, -1, 255)
    return display_image    #1 canal

##################################################################################################
#                                   Fit de elipse                                                #
##################################################################################################

@dataclass
class EllipseFitParams:
    residue_threshold: float
    def set_residue_threshold(self, val):
        self.residue_threshold = val

def fit_ellipse(yellow_contours : tuple, params : EllipseFitParams):
    is_ellipse = []
    coef_list = []
    centroid = []
    for cont in yellow_contours:
        positions = numpy.float64(cont)
        positions = positions.reshape((positions.shape[0], positions.shape[2]))
        avg_pos = numpy.mean(positions, axis=0)
        positions -= avg_pos
        x = positions[:, 0]
        y = positions[:, 1]

        #problema
        if not max(abs(x)) < 1e-5:
            x /= max(abs(x))
        if not max(abs(y)) < 1e-5:
            y /= max(abs(y))

        try:
            coefs, resid, rank, sing = numpy.linalg.lstsq(
                numpy.column_stack((x**2, x*y, y**2)), 
                numpy.ones_like(x), rcond=None)
        except numpy.linalg.LinAlgError:
            return False, [], numpy.array([-1,-1])
        
        if len(resid) > 0:
            #rank da matriz é 1 - ocorre quando o marco tá muito inclinado
            is_ellipse.append(resid[0] <= params.residue_threshold)
        else:
            is_ellipse.append(False)
        coef_list.append(coefs)
        centroid.append(avg_pos)
    #homogeinizar retorno com None
    return is_ellipse, coef_list, centroid

#função de display para debug
def display_ellipse(rgb_img: cv2.Mat, params):
    yellow_region = filter_yellow(rgb_img, params[0])
    contours =  extract_contours(yellow_region, params[1])
    are_ellipses, coef_list, centroids = fit_ellipse(contours, params[2])

    display_image = rgb_img.copy()
    for is_ellipse, centroid in zip(are_ellipses, centroids):
        if is_ellipse:
            cv2.circle(display_image, [int(centroid[0]), int(centroid[1])], 5, (255, 255, 255), -1)

    return display_image

###############################################################################################
#                               Identificação do mais                                         #
###############################################################################################

@dataclass
class PlusIdParams:
    contour_epsilon:float
    minimal_area:float
    vertex_count_tolerance:int
    def set_contour_epsilon(self, val):
        self.contour_epsilon = val / 1000
    def set_minimal_area(self, val):
        self.minimal_area = val
    def set_vertex_count_tolerance(self, val):
        self.vertex_count_tolerance = val

def identify_plus(contour, params):
    approx_polygonal_vertices = cv2.approxPolyDP(contour, params.contour_epsilon * cv2.arcLength(contour, True), True)
    M = cv2.moments(contour)
    area = M['m00']
    if area > params.minimal_area:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        vertex_criterion = abs(len(approx_polygonal_vertices) - 12) < params.vertex_count_tolerance
        concave_criterion = not cv2.isContourConvex(contour)
        #diferença de círculo
        perimeter = cv2.arcLength(contour, True)
        area_to_perimeter_squared = area / perimeter ** 2
        #1/16 é a razão para quadrado
        area_to_perimeter_criterion = area_to_perimeter_squared < 1/16
        if vertex_criterion and concave_criterion and area_to_perimeter_criterion:
            return numpy.array([cX, cY])
    return None

def identify_pluses(contours, params):
    return [identify_plus(cont, params) for cont in contours]

def display_plus(rgb_img : cv2.Mat, params):
    yellow_region = filter_yellow(rgb_img, params[0])
    contours =  extract_contours(yellow_region, params[1])
    pluses = identify_pluses(contours, params[2])

    display_image = rgb_img.copy()
    
    for plus, cont in zip(pluses, contours):
        if plus is not None:
            cv2.drawContours(display_image, [cont], -1, (0,0,0), 5)
            cv2.circle(display_image, plus, 5, (0, 0, 0), -1)
    
    return display_image

############################################################################################
#                                  Estimador de centro                                     #
############################################################################################

@dataclass
class CenterEstimateParams:
    tolerance:float
    def set_tolerance(self, val):
        self.tolerance = val

def estimate_center(plus_centers : List, ellipse_centers : List, params: CenterEstimateParams):
    #nenhuma detecção de centros
    if len(plus_centers) == 0 or len(ellipse_centers) == 0:
        return None
    
    #matriz de distâncias centro elipses x centro +
    #p/ encontrar centro de um mais *e* de uma elipse no msm ponto
    distances = numpy.empty((len(plus_centers), len(ellipse_centers)))

    for i, plus_center in enumerate(plus_centers):
        for j, ellipse_center in enumerate(ellipse_centers):
            if plus_center is not None and ellipse_center is not None:
                distances[i, j] = numpy.linalg.norm(plus_center - ellipse_center)
            else:
                distances[i, j] = numpy.inf
    
    #filtra distâncias menores que uma tolerância
    acceptable_distance_indexes = numpy.argwhere(distances <= params.tolerance)
    
    if len(acceptable_distance_indexes) > 0:
        best_plus_centers = [plus_centers[acc_i] for acc_i in acceptable_distance_indexes[:, 0]]
        best_elli_centers = [ellipse_centers[acc_i] for acc_i in acceptable_distance_indexes[:, 1]]
    else:
        return None
    #o marco de pouso tem 2 elipses e 1 mais, então os centros reais estão dobrados
    unfiltered_landing_pad_centers = [(bpc + bec) / 2 for bpc, bec in zip(best_plus_centers, best_elli_centers)]
    landing_pad_centers = []
    for i, unfiltered_center in enumerate(unfiltered_landing_pad_centers):
        distance_to_other_centers = [numpy.linalg.norm(unfiltered_center - other) for other in unfiltered_landing_pad_centers[:i]]
        if all([distance > params.tolerance for distance in distance_to_other_centers]):
            landing_pad_centers.append(unfiltered_center)
    return landing_pad_centers

def display_center(rgb_img : cv2.Mat, params : List):
    yellow_region                      = filter_yellow(rgb_img, params[0])
    contours                           =  extract_contours(yellow_region, params[1])
    pluses                             = identify_pluses(contours, params[2])
    are_ellipses, coef_list, centroids = fit_ellipse(contours, params[3])
    ellipse_centroids                  = [centroid for i, centroid in enumerate(centroids) if are_ellipses[i]]
    
    #continuar daqui
    centers = estimate_center(pluses, ellipse_centroids, params[4])
    display_image = rgb_img.copy()

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2

    if centers is not None:
        for i, center in enumerate(centers):
            if center is not None:
                cv2.putText(display_image, str(i), 
                    numpy.int32(center), 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                cv2.circle(display_image, numpy.int32(center), 5, (0, 0, 0), -1)
    return display_image

############################################################################################
#                                       Tracker                                            #
############################################################################################

@dataclass
class TrackerParams:
    alpha:float
    max_frames_wo_detection:int
    max_delta_position: float   #maxima diferença de distancia aceitavel p um ponto entre 2 frames
    def set_alpha(self, val):
        self.alpha = val/100
    def set_max_frames_wo_detection(self, val):
        self.max_frames_wo_detection = val
    def set_max_delta_position(self, val):
        self.max_delta_position = val

@dataclass
class Target:
    position: numpy.ndarray
    frames_wo_detection: int

class Tracker:
    def __init__(self, params : TrackerParams):
        self.params = params
        self.targets = []

    def __call__(self, new_measurement):
        #fazer pra cada alvo
        if new_measurement is None:
            if len(self.targets) > 0:
                for target in self.targets:
                    target.frames_wo_detection += 1
        else:
            if len(self.targets) == 0:
                self.targets = [Target(new_meas, 0) for new_meas in new_measurement]
            else:
                #matriz de distâncias alvo x medida
                distances = numpy.empty((len(self.targets), len(new_measurement)))
                for i, target in enumerate(self.targets):
                    for j, meas in enumerate(new_measurement):
                        distances[i, j] = numpy.linalg.norm(target.position - meas) if meas is not None else numpy.inf

                #atualização dos alvos já existentes
                for i, target in enumerate(self.targets):
                    arg_min_dist = numpy.argmin(distances[i, :])
                    if distances[i, arg_min_dist] <= self.params.max_delta_position:
                        #trackear com filtro exponencial
                        new_position_measurement = new_measurement[arg_min_dist]
                        target.position = self.params.alpha * new_position_measurement + (1 - self.params.alpha) * target.position
                    else:
                        target.frames_wo_detection += 1
                
                #inserção de novos alvos
                #encontrar todos os índices das medidas que NÃO estão a uma distância aceitável de qualquer alvo
                is_new_target = numpy.all(distances >= self.params.max_delta_position, axis=0)
                new_targets = [Target(new_target, 0) for i, new_target in enumerate(new_measurement) if is_new_target[i]]
                if len(new_targets) > 0:
                    self.targets = [*self.targets, *new_targets]
        
        self.targets = list(filter(lambda x: x.frames_wo_detection <= self.params.max_frames_wo_detection, self.targets))
        return [t.position for t in self.targets]

def display_tracker(rgb_img : cv2.Mat, params : List):
    yellow_region                      = filter_yellow(rgb_img, params[0])
    contours                           =  extract_contours(yellow_region, params[1])
    pluses                             = identify_pluses(contours, params[2])
    are_ellipses, coef_list, centroids = fit_ellipse(contours, params[3])
    ellipse_centroids                  = [centroid for i, centroid in enumerate(centroids) if are_ellipses[i]]
    centers                            = estimate_center(pluses, ellipse_centroids, params[4])
    tracked_centers                    = params[5](centers)
    # print(center, tracked_center)
    display_image = rgb_img.copy()
    for tracked_center in tracked_centers:
        if tracked_center is not None:
            cv2.circle(display_image, numpy.int32(tracked_center), 5, (0, 0, 0), -1)
    
    return display_image