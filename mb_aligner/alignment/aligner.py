import sys
import os
import glob
import yaml
import cv2
import ujson
import numpy as np
from rh_logger.api import logger
import logging
import rh_logger
import time
from mb_aligner.dal.section import Section
import multiprocessing as mp
from multiprocessing.pool import ThreadPool # for debug ?
import threading
#import queue
from collections import defaultdict
import tinyr
from mb_aligner.common.section_cache import SectionCacheProcesses as SectionCache
from mb_aligner.alignment.mesh_pts_model_exporter import MeshPointsModelExporter
from mb_aligner.alignment.normalize_coordinates import normalize_coordinates
import importlib


class StackAligner(object):

    def __init__(self, conf):
        self._conf = conf

        #self._processes_factory = ProcessesFactory(self._conf)
        self._processes_num = conf.get('processes_num', 1)
        assert(self._processes_num > 0)
        self._processes_pool = mp.Pool(processes=self._processes_num)
        #self._processes_pool = ThreadPool(processes=self._processes_num)
        # Initialize the pre_matcher, block_matcher and optimization algorithms
        pre_match_type = conf.get('pre_match_type')
        pre_match_params = conf.get('pre_match_params', {})
        self._pre_matcher = StackAligner.load_plugin(pre_match_type)(**pre_match_params)

        fine_match_type = conf.get('fine_match_type', None)
        self._fine_matcher = None
        if fine_match_type is not None:
            fine_match_params = conf.get('fine_match_params', {})
            self._fine_matcher = StackAligner.load_plugin(fine_match_type)(**fine_match_params)

        optimizer_type = conf.get('optimizer_type')
        optimizer_params = conf.get('optimizer_params', {})
        self._optimizer = StackAligner.load_plugin(optimizer_type)(**optimizer_params)

        # initialize other values
        self._compare_distance = conf.get('compare_distance', 1)
        self._work_dir = conf.get('work_dir', '3d_work_dir')
        self._output_dir = conf.get('output_dir', '3d_output_dir')

        self._create_directories()




    def __del__(self):
        self._processes_pool.close()
        self._processes_pool.join()

    def _create_directories(self):
        def create_dir(dir_name):
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

        create_dir(self._work_dir)
        self._pre_matches_dir = os.path.join(self._work_dir, 'pre_matches')
        create_dir(self._pre_matches_dir)
        self._fine_matches_dir = os.path.join(self._work_dir, 'fine_matches')
        create_dir(self._fine_matches_dir)
        self._post_opt_dir = os.path.join(self._work_dir, 'post_optimization_{}'.format(os.path.basename(self._output_dir)))
        create_dir(self._post_opt_dir)
        create_dir(self._output_dir)
            

    @staticmethod
    def _read_directory(in_dir):
        fnames_set = set(glob.glob(os.path.join(in_dir, '*')))
        return fnames_set

    @staticmethod
    def load_plugin(class_full_name):
        package, class_name = class_full_name.rsplit('.', 1)
        plugin_module = importlib.import_module(package)
        plugin_class = getattr(plugin_module, class_name)
        return plugin_class

    @staticmethod
    def read_imgs(folder):
        img_fnames = sorted(glob.glob(os.path.join(folder, '*')))[:10]
        print("Loading {} images from {}.".format(len(img_fnames), folder))
        imgs = [cv2.imread(img_fname, 0) for img_fname in img_fnames]
        return img_fnames, imgs


    @staticmethod
    def load_conf_from_file(conf_fname):
        '''
        Loads a given configuration file from a yaml file
        '''
        print("Using config file: {}.".format(conf_fname))
        if conf_fname is None:
            return {}
        with open(conf_fname, 'r') as stream:
            conf = yaml.load(stream)
            conf = conf['alignment']
        
        logger.report_event("loaded configuration: {}".format(conf), log_level=logging.INFO)
        return conf


    @staticmethod
    def _compute_l2_distance(pts1, pts2):
        delta = pts1 - pts2
        s = np.sum(delta**2, axis=1)
        return np.sqrt(s)


    @staticmethod
    def _create_section_rtree(section):
        '''
        Receives a section, and returns an rtree of all the section's tiles bounding boxes
        '''
        tiles_rtree = tinyr.RTree(interleaved=False, max_cap=5, min_cap=2)
        # Insert all tiles bounding boxes to an rtree
        for t in section.tiles():
            bbox = t.bbox
            # using the (x_min, x_max, y_min, y_max) notation
            tiles_rtree.insert(t, bbox)

        return tiles_rtree



    @staticmethod
    def _find_overlapping_tiles_gen(section):
        '''
        Receives a section, and yields triplets of (tile1, tile2, overlap_bbox ())
        '''
        tiles_rtree = Stitcher._create_section_rtree(section)
        # Iterate over the section tiles, and for each tile find all of its overlapping tiles
        for t in section.tiles():
            bbox = t.bbox
            rect_res = tiles_rtree.search(bbox)
            for overlap_t in rect_res:
                # We want to create a directed comparison (each tile with tiles that come after it in a lexicographical order)
                if overlap_t.mfov_index > t.mfov_index or (overlap_t.mfov_index == t.mfov_index and overlap_t.tile_index > t.tile_index):
                    yield t, overlap_t
#                     # Compute overlap area
#                     overlap_bbox = overlap_t.bbox
#                     intersection = [max(bbox[0], overlap_bbox[0]),
#                                     min(bbox[1], overlap_bbox[1]),
#                                     max(bbox[2], overlap_bbox[2]),
#                                     min(bbox[3], overlap_bbox[3])]
# 
#                     yield t, overlap_t, intersection
            

    def align_sections(self, sections):
        '''
        Receives a list of sections that were already stitched and need to be registered into a single volume.
        '''

        logger.report_event("align_sections starting.", log_level=logging.INFO)

        layout = {}
        layout['sections'] = sections
        layout['neighbors'] = defaultdict(set)

        # TODO - read the intermediate results directories (so we won't recompute steps)

        # for each pair of neighboring sections (up to compare_distance distance)
        pre_match_results = {}
        fine_match_results = {}
        sec_caches = defaultdict(SectionCache)
        for sec1_idx, sec1 in enumerate(sections):
            for j in range(1, self._compare_distance + 1):
                sec2_idx = sec1_idx + j
                if sec2_idx >= len(sections):
                    break

                sec2 = sections[sec2_idx]

                # TODO - check if the pre-match was already computed
                logger.report_event("Performing pre-matching between sections {} and {}".format(sec1.layer, sec2.layer), log_level=logging.INFO)
                # Result will be a map between mfov index in sec1, and (the model and filtered matches to section 2)
                pre_match_results[sec1_idx, sec2_idx] = self._pre_matcher.pre_match_sections(sec1, sec2, sec_caches[sec1.layer], sec_caches[sec2.layer], self._processes_pool)

                # Make sure that there are pre-matches between the two sections
                assert(np.any([model is not None for (model, _) in pre_match_results[sec1_idx, sec2_idx].values()]))

        
                layout['neighbors'][sec1_idx].add(sec2_idx)
                layout['neighbors'][sec2_idx].add(sec1_idx)
                if self._fine_matcher is None:
                    # No block matching, use the pre-match results as bi-directional fine-matches
                    cur_matches = [filtered_matches for model, filtered_matches in pre_match_results[sec1_idx, sec2_idx].values() if filtered_matches is not None]
                    if len(cur_matches) == 1:
                        fine_match_results[sec1_idx, sec2_idx] = cur_matches
                        fine_match_results[sec2_idx, sec1_idx] = [cur_matches[1], cur_matches[0]]
                    else:
                        fine_match_results[sec1_idx, sec2_idx] = np.concatenate(cur_matches, axis=1)
                        fine_match_results[sec2_idx, sec1_idx] = [fine_match_results[sec1_idx, sec2_idx][1], fine_match_results[sec1_idx, sec2_idx][0]]


                else:
                    # Perform block matching
                    # TODO - check if the fine-match was already computed
                    logger.report_event("Performing fine-matching between sections {} and {}".format(sec1.layer, sec2.layer), log_level=logging.INFO)
                    sec1_sec2_matches, sec2_sec1_matches = self._fine_matcher.match_layers_fine_matching(sec1, sec2, sec_caches[sec1_idx], sec_caches[sec2_idx], pre_match_results[sec1_idx, sec2_idx], self._processes_pool)
                    logger.report_event("fine-matching between sections {0} and {1} results: {0}->{1} {2} matches, {0}<-{1} {3} matches ".format(sec1.layer, sec2.layer, len(sec1_sec2_matches[0]), len(sec2_sec1_matches[0])), log_level=logging.INFO)
                    fine_match_results[sec1_idx, sec2_idx] = sec1_sec2_matches
                    fine_match_results[sec2_idx, sec1_idx] = sec2_sec1_matches



                # Make sure that there are matches between the two sections
                assert(len(fine_match_results[sec1_idx, sec2_idx]) > 0)
                assert(len(fine_match_results[sec2_idx, sec1_idx]) > 0)

        # optimize the matches
        logger.report_event("Optimizing the matches...", log_level=logging.INFO)

        self._optimizer.optimize(layout, fine_match_results, lambda section, orig_pts, new_pts, mesh_spacing: update_section_post_optimization_and_save(section, orig_pts, new_pts, mesh_spacing, self._post_opt_dir), self._processes_pool)


        # TODO - normalize all the sections (shift everything so we'll have a (0, 0) coordinate system for the stack)
        normalize_coordinates([self._post_opt_dir], self._output_dir, self._processes_pool)

def update_section_post_optimization_and_save(section, orig_pts, new_pts, mesh_spacing, out_dir):
    logger.report_event("Exporting section {}".format(section.canonical_section_name), log_level=logging.INFO)
    exporter = MeshPointsModelExporter()
    exporter.update_section_points_model_transform(section, orig_pts, new_pts, mesh_spacing)

    # TODO - should also save the mesh as h5s

    # save the output section
    out_fname = os.path.join(out_dir, '{}.json'.format(section.canonical_section_name))
    print('Writing output to: {}'.format(out_fname))
    section.save_as_json(out_fname)
    



if __name__ == '__main__':
    secs_ts_fnames = [
        '/n/home10/adisuis/Harvard/git_lichtmangpu01/mb_aligner/scripts/ECS_test9_cropped_010_S10R1.json',
        '/n/home10/adisuis/Harvard/git_lichtmangpu01/mb_aligner/scripts/ECS_test9_cropped_011_S11R1.json',
        '/n/home10/adisuis/Harvard/git_lichtmangpu01/mb_aligner/scripts/ECS_test9_cropped_012_S12R1.json',
        '/n/home10/adisuis/Harvard/git_lichtmangpu01/mb_aligner/scripts/ECS_test9_cropped_013_S13R1.json',
        '/n/home10/adisuis/Harvard/git_lichtmangpu01/mb_aligner/scripts/ECS_test9_cropped_014_S14R1.json'
    ]
    out_folder = './output_aligned_ECS_test9_cropped'
    conf_fname = '../../conf/conf_example.yaml'


    logger.start_process('main', 'aligner.py', [secs_ts_fnames, conf_fname])
    conf = StackAligner.load_conf_from_file(conf_fname)
    logger.report_event("Loading sections", log_level=logging.INFO)
    sections = []
    # TODO - Should be done in a parallel fashion
    for ts_fname in secs_ts_fnames:
        with open(ts_fname, 'rt') as in_f:
            tilespec = ujson.load(in_f)
        wafer_num = 1
        sec_num = int(os.path.basename(ts_fname).split('_')[-1].split('S')[1].split('R')[0])
        sections.append(Section.create_from_tilespec(tilespec, wafer_section=(wafer_num, sec_num)))
    logger.report_event("Initializing aligner", log_level=logging.INFO)
    aligner = StackAligner(conf)
    logger.report_event("Aligning sections", log_level=logging.INFO)
    aligner.align_sections(sections) # will align and update the section tiles' transformations


    del aligner

    logger.end_process('main ending', rh_logger.ExitCode(0))

