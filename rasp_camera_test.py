import landing_pad_detector
import numpy
yellow_filter_params = landing_pad_detector.YellowFilterParams(
            numpy.array([29, 93, 0]),
            numpy.array([52, 256,256])
)

yellow_filter_params = landing_pad_detector.YellowFilterParams(
            numpy.array([29, 93, 0]),
            numpy.array([52, 256,256])
)

contour_extract_params = landing_pad_detector.ContourExtractParams(50)

ellipse_fit_params = landing_pad_detector.EllipseFitParams(10)

plus_identify_params = landing_pad_detector.PlusIdParams(0.01, 400, 2)

circle_estimate_params = landing_pad_detector.CenterEstimateParams(20)
tracker = landing_pad_detector.Tracker(landing_pad_detector.TrackerParams(0.8, 5))

landing_pad_detector.test_with_video(landing_pad_detector.display_tracker, [yellow_filter_params,
                                        contour_extract_params,
                                        plus_identify_params,
                                        ellipse_fit_params,
                                        circle_estimate_params,
                                        tracker])