import numpy as np
from scipy.spatial import KDTree
import rh_renderer.models
from mb_aligner.common import ransac
from rh_logger.api import logger
import logging
import rh_logger



class AffineTransformsGrouper(object):
    """
    A class that helps clustering affine transforms
    """
    def __init__(self, **kwargs):
        self._seed = kwargs.get("seed", None)
        #self._inlier_distance_threshold = kwargs.get("inlier_distance_threshold", 30)

        self._model_index = 3 #kwargs.get("model_index", 3) # Affine
        self._min_matches = kwargs.get("min_matches", 10)
        self._iterations = kwargs.get("iterations", 50)
        self._max_epsilon = kwargs.get("max_epsilon", 40)
        self._min_inlier_ratio = kwargs.get("min_inlier_ratio", 0)
        self._min_num_inlier = kwargs.get("min_num_inlier", 10)
        self._max_trust = kwargs.get("max_trust", 3)
        self._det_delta = kwargs.get("det_delta", None)
        self._max_stretch = kwargs.get("max_stretch", None)
        self._robust_filter = True if "robust_filter" in kwargs else False

    def group_matches(self, pts1, pts2):
        """
        Receives a list of matches (pts1 <-> pts2), and returns a list of masks, each of a different cluster.
        """
        assert(pts1.shape == pts2.shape)

        matches_num = pts1.shape[0]
        outlier_mask = np.zeros((matches_num, ), dtype=np.bool)
        already_handled_mask = np.zeros((matches_num, ), dtype=np.bool)
        groups_masks = []
        covered_matches_num = 0


        np.random.seed(self._seed)

        while covered_matches_num < matches_num:
            
            cur_group_mask = np.zeros((matches_num, ), dtype=np.bool)

            # apply ransac on the current leftovers
            # TODO: make the parameters modifiable from the c'tor
            cur_matches = np.array([pts1[~already_handled_mask], pts2[~already_handled_mask]])
            model, filtered_matches, filtered_matches_mask = ransac.filter_matches(cur_matches, cur_matches, self._model_index, self._iterations, self._max_epsilon, self._min_inlier_ratio, self._min_num_inlier, self._max_trust, self._det_delta, self._max_stretch, robust_filter=self._robust_filter)

            # if haven't found a model, set the rest of the matches to outliers
            if model is None:
                outlier_mask[:] = ~already_handled_mask
                covered_matches_num = matches_num
            else:
                # otherwise, add the filtered_matches to the groups mask
                cur_group_mask[~already_handled_mask] = filtered_matches_mask

                groups_masks.append(cur_group_mask)
                covered_matches_num += np.sum(filtered_matches_mask)
                already_handled_mask |= cur_group_mask

#             # create a kdtree of the non-covered matches
#             # TODO - use a different datastructure that is faster
#             kdtree = KDTree(pts1[~already_handled_mask])
# 
#             # choose a single match
#             match_idx = np.random.choice(len(pts1[~already_handled_mask]))
#             
#             # find 2-closest matches to it
#             # Compute the affine transform of these points
# 
#             # find which other points are in the same transform

        return groups_masks, outlier_mask


if __name__ == '__main__':
    # create 2 sets of random pts and their affine transform matches
    transformA = np.array([
            [np.cos(30), -np.sin(30), 300],
            [np.sin(30), np.cos(30), 150],
            [0., 0., 1.]
        ])
    ptsA_1 = np.random.uniform(0, 800, ((100, 2)))
    noiseA = np.random.uniform(0, 7, ptsA_1.shape)
    ptsA_2 = np.dot(ptsA_1, transformA[:2, :2].T) + transformA[:2, 2].T + noiseA

    transformB = np.array([
            [np.cos(50), -np.sin(50), 600],
            [np.sin(50), np.cos(50), 250],
            [0., 0., 1.]
        ])
    ptsB_1 = np.random.uniform(0, 800, ((100, 2)))
    noiseB = np.random.uniform(-7, 7, ptsB_1.shape)
    ptsB_2 = np.dot(ptsB_1, transformB[:2, :2].T) + transformB[:2, 2].T + noiseB

    # Add outliers
    ptsC_1 = np.random.uniform(0, 800, ((5, 2)))
    noiseC = np.random.uniform(-3, 3, ptsC_1.shape)
    ptsC_2 = ptsC_1 + noiseC


    # stack the matches
    pts1 = np.vstack((ptsA_1, ptsB_1, ptsC_1))
    pts2 = np.vstack((ptsA_2, ptsB_2, ptsC_2))

    logger.start_process('main', 'affine_transforms_grouper.py', [])
    # create the clusterer
    grouper = AffineTransformsGrouper()

    groups_masks, outlier_mask = grouper.group_matches(pts1, pts2)

    print("Found {} groups masks".format(len(groups_masks)))
    for gm_idx, gm in enumerate(groups_masks):
        print("Group masks {}: size:{}".format(gm_idx, np.sum(gm)))

    print("Found {} outlier matches".format(np.sum(outlier_mask)))
    logger.end_process('main ending', rh_logger.ExitCode(0))


