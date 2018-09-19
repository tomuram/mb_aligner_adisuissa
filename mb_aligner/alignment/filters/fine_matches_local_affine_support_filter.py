from __future__ import print_function
'''
Recevies matches (entire matches for pair of sections),
and finds for each match (p1, p2), whether p2 agrees with the target points of the surrounding matches of p1.
'''
from rh_logger.api import logger
import rh_logger
import logging
import numpy as np
import tinyr
from mb_aligner.common import ransac
import argparse


class FineMatchesAffineSupportFilter(object):

    def __init__(self, **kwargs):
        if kwargs is None:
            kwargs = {}
        self._support_radius = kwargs.get("support_radius", 7500)
        self._model_index = kwargs.get("model_index", 3) # Affine
        self._min_matches = kwargs.get("min_matches", 3)
        self._iterations = kwargs.get("iterations", 50)
        self._max_epsilon = kwargs.get("max_epsilon", 30)
        self._min_inlier_ratio = kwargs.get("min_inlier_ratio", 0)
        self._min_num_inlier = kwargs.get("min_num_inlier", 3)
        self._max_trust = kwargs.get("max_trust", 3)
        self._det_delta = kwargs.get("det_delta", 0.99)
        self._max_stretch = kwargs.get("max_stretch", 0.99)
        self._robust_filter = True if "robust_filter" in kwargs else False
        #print("self._max_epsilon", self._max_epsilon)
        assert(self._support_radius > 1)
        #assert(self._min_matches >= 3)

    def _run_ransac(self, matches):
        # TODO: make the parameters modifiable from the c'tor
        model, filtered_matches, filtered_matches_mask = ransac.filter_matches(matches, matches, self._model_index, self._iterations, self._max_epsilon, self._min_inlier_ratio, self._min_num_inlier, self._max_trust, self._det_delta, self._max_stretch, robust_filter=self._robust_filter)

        return model, filtered_matches, filtered_matches_mask

    def filter_matches(self, in_matches):
        assert(in_matches.shape[0] == 2)
        assert(in_matches.shape[2] == 2)

        # Build an r-tree of all the matches source points
        in_matches_rtree = tinyr.RTree(interleaved=True, max_cap=5, min_cap=2)
        for m_id, m_src_pt in enumerate(in_matches[0]):
            # using the (x_min, y_min, x_max, y_max) notation
            in_matches_rtree.insert(m_id, (m_src_pt[0], m_src_pt[1], m_src_pt[0]+1, m_src_pt[1]+1))

        matches_mask = np.zeros((len(in_matches[0]), ), dtype=bool)
        # For each match, search around for all matches in the supprt radius, compute the defined affine transform of these,
        # and points, and if it is part of the filtered matches keep it
        fail1_cnt = 0
        fail2_cnt = 0
        fail3_cnt = 0
        for m_id, m_src_pt in enumerate(in_matches[0]):
            rect_res = in_matches_rtree.search( (m_src_pt[0] - self._support_radius, m_src_pt[1] - self._support_radius, m_src_pt[0] + self._support_radius, m_src_pt[1] + self._support_radius) )
            m_other_ids = [m_other_id for m_other_id in rect_res]
            #logger.report_event("{}: found neighbors#:{}".format(m_id, len(m_other_ids)), log_level=logging.DEBUG)
            if len(m_other_ids) < self._min_matches:
                # There aren't enough matches in the radius, filter out the match
                #print("fail 1")
                fail1_cnt += 1
                continue
            support_matches = np.array([
                in_matches[0][m_other_ids],
                in_matches[1][m_other_ids]
            ])
            model, support_matches_filtered, support_matches_mask = self._run_ransac(support_matches)
            if model is None:
                # Couldn't find a valid model, filter out the match point
                fail2_cnt += 1
                continue
            #logger.report_event("m_src_pt: {}, support_matches_filtered[0]: {}".format(m_src_pt, support_matches_filtered[0]), log_level=logging.DEBUG)
            # make sure the point is part of the valid model support matches
            # (checking that m_src_pt is inside support_matches_filtered[0])
            if not np.any(np.all(np.abs(support_matches_filtered[0] - m_src_pt) <= 0.0001, axis=1)):
                fail3_cnt += 1
                continue
#             if support_matches_mask[m_id] == False:
#                 fail3_cnt += 1
#                 continue
            #logger.report_event("{}: model: {}".format(m_id, model.get_matrix()), log_level=logging.DEBUG)

            matches_mask[m_id] = True

        logger.report_event("Parsed {} matches, {} didn't have enough matches in the radius, {} failed finding good model for, {} had a surrounding model but weren't part of that".format(len(in_matches[0]), fail1_cnt, fail2_cnt, fail3_cnt), log_level=logging.INFO)
        return in_matches[:, matches_mask, :], matches_mask


