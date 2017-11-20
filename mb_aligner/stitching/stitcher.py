import sys
import os
import glob
import yaml
import cv2
import numpy as np
from rh_logger.api import logger
import logging
import rh_logger
import time
from ..common.detector import FeaturesDetector
from ..common.matcher import FeaturesMatcher
from .optimize_2d_mfovs import OptimizerRigid2D
from pyrtree import RTree, Rect
import multiprocessing as mp
from collections import defaultdict

class Stitcher(object):

    def __init__(self, conf, processes_num=1):
        self._conf = conf

        # TODO Initialize the detector, amtcher and optimizer objects
        detector_params = conf.get('detector_params', {})
        matcher_params = conf.get('matcher_params', {})
        self._detector = FeaturesDetector(conf['detector_type'], **detector_params)
        self._matcher = FeaturesMatcher(self._detector, **matcher_params)
        optimizer_params = conf.get('optimizer_parans', {})
        self._optimizer = OptimizerRigid2D(**optimizer_params)

        self._processes_num = processes_num



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
        with open(conf_fname, 'r') as stream:
            conf = yaml.load(stream)
            conf = conf['stitching']
        return conf


    @staticmethod
    def _compute_l2_distance(pts1, pts2):
        delta = pts1 - pts2
        s = np.sum(delta**2, axis=1)
        return np.sqrt(s)

    @staticmethod
    def _compute_features(detector, img):
        result = detector.detect(img)
        return result

    def _compute_tile_features(self, tile, bbox=None):
        img = tile.image
        if bbox is not None:
            # Find the overlap between the given bbox and the tile actual bounding box,
            # and crop that overlap area
            t_bbox = tile.bbox
            # normalize the given bbox to the tile coordinate system, and then make sure that bounding box is in valid with the image size
            crop_bbox = [bbox[0] - t_bbox[0], bbox[1] - t_bbox[0], bbox[2] - t_bbox[2], bbox[3] - t_bbox[2]]
            crop_bbox = [
                            max(int(crop_bbox[0]), 0),
                            min(int(crop_bbox[1]), tile.width),
                            max(int(crop_bbox[2]), 0),
                            min(int(crop_bbox[3]), tile.height)
                        ]
            img = img[crop_bbox[2]:crop_bbox[3], crop_bbox[0]:crop_bbox[1]]
            
        result = self._detector.detect(img)

        # Change the features key points to the world coordinates
        delta_x = tile.bbox[0]
        delta_y = tile.bbox[2]
        if bbox is not None:
            delta_x += crop_bbox[0]
            delta_y += crop_bbox[2]
        for kp in result[0]:
            cur_point = list(kp.pt)
            cur_point[0] += delta_x
            cur_point[1] += delta_y
            kp.pt = tuple(cur_point)

        return result


    @staticmethod
    def _match_features(features_result1, features_result2, i, j):
        transform_model, filtered_matches = self._matcher.match_and_filter(*features_result1, *features_result2)
        assert(transform_model is not None)
        transform_matrix = transform_model.get_matrix()
        logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, j, transform_matrix, np.mean(Stitcher._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)
        return transform_matrix
 

    @staticmethod
    def _find_section_tiles_bboxes(section):
        '''
        Receives a section, and returns an rtree of all the section's tiles bounding boxes
        '''
        tiles_rtree = RTree()
        # Insert all tiles bounding boxes to an rtree
        for t in section.tiles():
            bbox = t.bbox
            # pyrtree uses the (x_min, y_min, x_max, y_max) notation
            tiles_rtree.insert(t, Rect(bbox[0], bbox[2], bbox[1], bbox[3]))

        return tiles_rtree



    @staticmethod
    def _find_overlapping_tiles(section):
        '''
        Receives a section, and returns 3 lists (of the same length):
            tiles_lst1 - list of tiles
            tiles_lst2 - list of tiles each element corresponds to the matching element in tiles_lst1 and indicates an overlap
            overlap_bboxes - to each pair of overlaping tiles, their overlap bounding box
        '''
        tiles_lst1 = []
        tiles_lst2 = []
        overlap_bboxes = []
        tiles_rtree = Stitcher._find_section_tiles_bboxes(section)
        # Iterate over the section tiles, and for each tile find all of its overlapping tiles
        for t in section.tiles():
            bbox = t.bbox
            rect_res = tiles_rtree.query_rect( Rect(bbox[0], bbox[2], bbox[1], bbox[3]) )
            for rtree_node in rect_res:
                if not rtree_node.is_leaf():
                    continue
                overlap_t = rtree_node.leaf_obj()
                # We want to create a directed comparison (each tile with tiles that come after it in a lexicographical order)
                if overlap_t.mfov_index > t.mfov_index or (overlap_t.mfov_index == t.mfov_index and overlap_t.tile_index > t.tile_index):
                    # Compute overlap area
                    overlap_bbox = overlap_t.bbox
                    intersection = [max(bbox[0], overlap_bbox[0]),
                                    min(bbox[1], overlap_bbox[1]),
                                    max(bbox[2], overlap_bbox[2]),
                                    min(bbox[3], overlap_bbox[3])]

                    tiles_lst1.append(t)
                    tiles_lst2.append(overlap_t)
                    overlap_bboxes.append(intersection)

        return tiles_lst1, tiles_lst2, overlap_bboxes

    def stitch_section(self, section):
        '''
        Receives a single section (assumes no transformations), stitches all its tiles, and updaates the section tiles' transformations.
        '''

        #pool = mp.Pool(processes=processes_num)

        # Compute features
        logger.start_process('stitch_section', 'stitcher.py', [section.layer, self._conf])
        logger.report_event('Finding neighboring tiles in section {}'.format(section.layer), log_level=logging.INFO)
        tiles_lst1, tiles_lst2, overlap_bboxes = Stitcher._find_overlapping_tiles(section)

        logger.report_event("Computing features on overlapping areas...", log_level=logging.INFO)
        st_time = time.time()
        overlap_features = [[], []] # The first sub-list will correspond to tiles_lst1, and the second to tiles_lst2
        pool_results = []
        extend_delta = 50 # TODO - should be a value
        for t1, t2, overlap_bbox in zip(tiles_lst1, tiles_lst2, overlap_bboxes):
            extended_overlap_bbox = [overlap_bbox[0] - extend_delta, overlap_bbox[1] + extend_delta, overlap_bbox[2] - extend_delta, overlap_bbox[3] + extend_delta] # increase overlap bbox by delta
            features1 = self._compute_tile_features(t1, extended_overlap_bbox)
            features2 = self._compute_tile_features(t2, extended_overlap_bbox)
            #res = pool.apply_async(StackAligner._compute_features, (self._detector, img, i))
            #pool_results.append(res)
            #all_features.append(self._detector.detect(img))
            overlap_features[0].append(features1)
            overlap_features[1].append(features2)
            logger.report_event("Overlap between tiles {} {}, found {} and {} features.".format(t1, t2, len(features1[0]), len(features2[0])), log_level=logging.INFO)
        #for res in pool_results:
        #    all_features.append(res.get())
        logger.report_event("Features computation took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)

        # match features of overlapping images
        # TODO - stopped here
        logger.report_event("Feature matching...", log_level=logging.INFO)
        st_time = time.time()
        #overlaps_filtered_matches = []
        tile_pairs_matches = {} # A map between two tiles (their url) and a list of all their matched points (world coordinates)
        tile_pts = defaultdict(list) # a map between a tile (its url) and a list of point lists (the points that are part of tile_pairs_matches)
        for i, (features1, features2) in enumerate(zip(overlap_features[0], overlap_features[1])):
            kps1, descs1 = features1
            kps2, descs2 = features2
            transform_model, filtered_matches = self._matcher.match_and_filter(kps1, descs1, kps2, descs2)
            if transform_model is None:
                # TODO - either use a different approach, or just create random matches
                logger.report_event("Matching tiles {} -> {}, No filtered matches found".format(tiles_lst1[i], tiles_lst2[i]), log_level=logging.INFO)
                #overlaps_filtered_matches.append(None)
                #tile_pairs_matches[tiles_lst1[i].img_fname, tiles_lst2[i].img_fname] = 
            else:
                logger.report_event("Matching tiles {} -> {}, found {} filtered matches, and the average displacement: {} px".format(tiles_lst1[i], tiles_lst2[i], len(filtered_matches[0]), np.mean(Stitcher._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)
                #overlaps_filtered_matches.append(filtered_matches)
                tile_pairs_matches[tiles_lst1[i].img_fname, tiles_lst2[i].img_fname] = filtered_matches
                tile_pts[tiles_lst1[i].img_fname].append(filtered_matches[0])
                tile_pts[tiles_lst2[i].img_fname].append(filtered_matches[1])
        logger.report_event("Feature matching took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)

        # Optimize all the matches and get a per-tile transformation
        logger.report_event("Optimizing transformations...", log_level=logging.INFO)
        st_time = time.time()

        tile_start_pts = {t.img_fname:np.array([t.bbox[0], t.bbox[2]]) for t in section.tiles()} # a map between a tile (its url) and the point where it starts (the minimal x,y of that tile)
        optimized_models = self._optimizer.optimize_2d_tiles(tile_pairs_matches, tile_pts, tile_start_pts, "layer {}".format(section.layer))
        logger.report_event("Optimization took {} seconds.".format(time.time() - st_time), log_level=logging.INFO)
        
        # Update the section transforms
        for tile in section.tiles():
            tile.set_transform(optimized_models[tile.img_fname])

        # TODO update the section's (and mfovs) bounding boxes

        #pool.close()
        #pool.join()

        logger.end_process('stitch_section ending', rh_logger.ExitCode(0))




    @staticmethod
    def align_img_files(imgs_dir, conf, processes_num):
        # Read the files
        _, imgs = StackAligner.read_imgs(imgs_dir)

        aligner = StackAligner(conf, processes_num)
        return aligner.align_imgs(imgs)


if __name__ == '__main__':
    section_dir = '/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1/full_image_coordinates.txt'
    section_num = 10
    conf_fname = '../../conf/conf_example.yaml'
    processes_num = 8
    out_fname = './output_stitched_sec{}.json'.format(section_num)

    section = Section.create_from_full_image_coordinates(section_dir, section_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf, processes_num)
    stitcher.stitch_section(section) # will stitch and update the section

    # Save the transforms to file
    import json
    print('Writing output to: {}'.format(out_fname))
    section.save_as_json(out_fname)
#     img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
#     for img_fname, img, transform in zip(img_fnames, imgs, transforms):
#         # assumption: the output image shape will be the same as the input image
#         out_fname = os.path.join(out_path, os.path.basename(img_fname))
#         img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
#         cv2.imwrite(out_fname, img_transformed)

