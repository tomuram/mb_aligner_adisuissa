import numpy as np
from . import ransac
from rh_renderer import models

class FeaturesMatcher(object):

    def __init__(self, detector, **kwargs):
        self._detector = detector

        self._params = {}
        # get default values if no value is present in kwargs
        #self._params["num_filtered_percent"] = kwargs.get("num_filtered_percent", 0.25)
        #self._params["filter_rate_cutoff"] = kwargs.get("filter_rate_cutoff", 0.25)
        self._params["ROD_cutoff"] = kwargs.get("ROD_cutoff", 0.92)
        self._params["min_features_num"] = kwargs.get("min_features_num", 40)

        # Parameters for the RANSAC
        self._params["model_index"] = kwargs.get("model_index", 3)
        self._params["iterations"] = kwargs.get("iterations", 5000)
        self._params["max_epsilon"] = kwargs.get("max_epsilon", 30.0)
        self._params["min_inlier_ratio"] = kwargs.get("min_inlier_ratio", 0.01)
        self._params["min_num_inlier"] = kwargs.get("min_num_inliers", 7)
        self._params["max_trust"] = kwargs.get("max_trust", 3)
        self._params["det_delta"] = kwargs.get("det_delta", 0.9)
        self._params["max_stretch"] = kwargs.get("max_stretch", 0.25)

        self._params["use_regularizer"] = True if "use_regularizer" in kwargs.keys() else False
        self._params["regularizer_lambda"] = kwargs.get("regularizer_lambda", 0.1)
        self._params["regularizer_model_index"] = kwargs.get("regularizer_model_index", 1)


    def match(self, features_kps1, features_descs1, features_kps2, features_descs2):
        matches = self._detector.match(features_descs1, features_descs2)

        good_matches = []
        for m, n in matches:
            #if (n.distance == 0 and m.distance == 0) or (m.distance / n.distance < actual_params["ROD_cutoff"]):
            if m.distance < self._params["ROD_cutoff"] * n.distance:
                good_matches.append(m)

        match_points = (
            np.array([features_kps1[m.queryIdx].pt for m in good_matches]),
            np.array([features_kps2[m.trainIdx].pt for m in good_matches]),
            np.array([m.distance for m in good_matches])
        )

        return match_points

    def match_and_filter(self, features_kps1, features_descs1, features_kps2, features_descs2):
        match_points = self.match(features_kps1, features_descs1, features_kps2, features_descs2)

        model, filtered_matches = ransac.filter_matches(match_points, match_points, self._params['model_index'],
                    self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                    self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'])

        if model is None:
            return None, None

        if self._params["use_regularizer"]:
            regularizer_model, _ = ransac.filter_matches(match_points, match_points, self._params['regularizer_model_index'],
                        self._params['iterations'], self._params['max_epsilon'], self._params['min_inlier_ratio'],
                        self._params['min_num_inlier'], self._params['max_trust'], self._params['det_delta'], self._params['max_stretch'])

            if regularizer_model is None:
                return None, None

            result = model.get_matrix() * (1 - self._params["regularizer_lambda"]) + regularizer_model.get_matrix() * self._params["regularizer_lambda"]
            model = models.AffineModel(result)

        return model, filtered_matches

