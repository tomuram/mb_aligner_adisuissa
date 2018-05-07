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
from mb_aligner.dal.section import Section
from mb_aligner.common.detector import FeaturesDetector
from mb_aligner.common.matcher import FeaturesMatcher
from mb_aligner.factories.processes_factory import ProcessesFactory
from mb_aligner.stitching.optimize_2d_mfovs import OptimizerRigid2D
import multiprocessing as mp
import threading
#import queue
from collections import defaultdict
import tinyr

#from ..pipeline.task_runner import TaskRunner
#import queue

class DetectorWorker(object):
    def __init__(self, processes_factory, input_queue, all_result_queues):
        self._detector = processes_factory.create_2d_detector()
        self._input_queue = input_queue
        self._all_result_queues = all_result_queues

    def run(self):
        # Read a job from input queue (blocking)
        print('Detector running')

        while True:
            print("Detector queue size:", self._input_queue.qsize())
            job = self._input_queue.get()
            if job is None:
                break
            # job = (matcher's result queue idx, tile fname, start_point, crop_bbox)
            #out_queue, tile_fname, start_point, crop_bbox = job
            out_queue_idx, tile_fname, start_point, crop_bbox = job
            out_queue = self._all_result_queues[out_queue_idx]
            # process the job
            print("Received job:", job)
            print("Reading file:", tile_fname)
            img = cv2.imread(tile_fname, 0)
            if crop_bbox is not None:
                # Find the overlap between the given bbox and the tile actual bounding box,
                # and crop that overlap area
                # normalize the given bbox to the tile coordinate system, and then make sure that bounding box is in valid with the image size
                local_crop_bbox = [crop_bbox[0] - start_point[0], crop_bbox[1] - start_point[0], crop_bbox[2] - start_point[1], crop_bbox[3] - start_point[1]]
                local_crop_bbox = [
                                max(int(local_crop_bbox[0]), 0),
                                min(int(local_crop_bbox[1]), img.shape[1]),
                                max(int(local_crop_bbox[2]), 0),
                                min(int(local_crop_bbox[3]), img.shape[0])
                                  ]
                img = img[local_crop_bbox[2]:local_crop_bbox[3], local_crop_bbox[0]:local_crop_bbox[1]]
                
            print("detecting features file:", job[1])
            kps, descs = self._detector.detect(img)
            # Create an array of the points of the kps
            kps_pts = np.empty((len(kps), 2), dtype=np.float64)
            for kp_i, kp in enumerate(kps):
                kps_pts[kp_i][:] = kp.pt
            # Change the features key points to the world coordinates
            delta = np.array(start_point)
            if crop_bbox is not None:
                delta[0] += local_crop_bbox[0]
                delta[1] += local_crop_bbox[2]
            kps_pts += delta
            # result = (tile fname, local area, features pts, features descs)
            # Put result in matcher's result queue
            print("submitting result for:", tile_fname)
            out_queue.put((tile_fname, kps_pts, descs))


class MatcherWorker(object):
    def __init__(self, processes_factory, input_queue, output_queue, matcher_queue, detectors_in_queue, matcher_thread_idx):
        self._matcher = processes_factory.create_2d_matcher()
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._matcher_queue = matcher_queue
        self._detectors_in_queue = detectors_in_queue
        self._matcher_thread_idx = matcher_thread_idx

    def run(self):
        # Read a job from input queue (blocking)
        print('Detector running')

        while True:
            print("Matcher queue size:", self._input_queue.qsize())
            job = self._input_queue.get()
            if job is None:
                break
            # job = (match_idx, tile1, tile2)
            match_idx, tile1, tile2 = job
            bbox1 = tile1.bbox
            bbox2 = tile2.bbox
#             # job = (match_idx, img_shape, all_tiles_list, tile_idx1, tile_idx2)
#             match_idx, img_shape, all_tiles_list, tile_idx1, tile_idx2 = job
#             tile_fname1, start_point1 = all_tiles_list[tile_idx1]
#             tile_fname2, start_point2 = all_tiles_list[tile_idx2]
            # process the job
            print("Received match job:", match_idx)
            # Find shared bounding box
            extend_delta = 50 # TODO - should be a parameter
            intersection = [max(bbox1[0], bbox2[0]) - extend_delta,
                            min(bbox1[1], bbox2[1]) + extend_delta,
                            max(bbox1[2], bbox2[2]) - extend_delta,
                            min(bbox1[3], bbox2[3]) + extend_delta]

#             intersection = [max(start_point1[0], start_point2[0]) - extend_delta,
#                             min(start_point1[0] + img_shape[1], start_point2[0] + img_shape[1]) + extend_delta,
#                             max(start_point1[1], start_point2[1]) - extend_delta,
#                             min(start_point1[1] + img_shape[0], start_point2[1] + img_shape[0]) + extend_delta]

            # Send two detector jobs
            print('Submitting job1 to detectors_in_queue')
            #job1 = (self._matcher_queue, tile1.img_fname, (bbox1[0], bbox1[2]), intersection)
            job1 = (self._matcher_thread_idx, tile1.img_fname, (bbox1[0], bbox1[2]), intersection)
            self._detectors_in_queue.put(job1)
            print('Submitting job2 to detectors_in_queue')
            #job2 = (self._matcher_queue, tile2.img_fname, (bbox2[0], bbox2[2]), intersection)
            job2 = (self._matcher_thread_idx, tile2.img_fname, (bbox2[0], bbox2[2]), intersection)
            self._detectors_in_queue.put(job2)

            # fetch the results
            res_a = self._matcher_queue.get()
            res_b = self._matcher_queue.get()
            # res_a = (tile_fname_A, kps_pts_A, descs_A))
            if res_a[0] == tile1.img_fname:
                _, kps_pts1, descs1 = res_a
                _, kps_pts2, descs2 = res_b
            else:
                _, kps_pts1, descs1 = res_b
                _, kps_pts2, descs2 = res_a

            # perform the actual matching
            transform_model, filtered_matches = self._matcher.match_and_filter(kps_pts1, descs1, kps_pts2, descs2)
            self._output_queue.put((match_idx, filtered_matches))

            # return the filtered matches (the points for each tile in the global coordinate system)
            #assert(transform_model is not None)
            #transform_matrix = transform_model.get_matrix()
            #logger.report_event("Imgs {} -> {}, found the following transformations\n{}\nAnd the average displacement: {} px".format(i, j, transform_matrix, np.mean(Stitcher._compute_l2_distance(transform_model.apply(filtered_matches[1]), filtered_matches[0]))), log_level=logging.INFO)



class ThreadWrapper(object):
    def __init__(self, ctor, *args, **kwargs):
        self._process = threading.Thread(target=ThreadWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
        self._process.start()

    @staticmethod
    def init_and_run(ctor, args, **kwargs):
        print("ctor:", ctor)
        #print("args:", args[0])
        worker = ctor(*args[0], **kwargs)
        worker.run()

    def join(self, timeout=None):
        return self._process.join(timeout)
    # TODO - add a process kill method

class ProcessWrapper(object):
    def __init__(self, ctor, *args, **kwargs):
        self._process = mp.Process(target=ProcessWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
        self._process.start()

    @staticmethod
    def init_and_run(ctor, args, **kwargs):
        print("ctor:", ctor)
        #print("args:", args[0])
        worker = ctor(*args[0], **kwargs)
        worker.run()

    def join(self, timeout=None):
        return self._process.join(timeout)
    # TODO - add a process kill method



# class ProcessWrapper(object):
#     def __init__(self, ctor, *args, **kwargs):
#         self._process = mp.Process(target=ProcessWrapper.init_and_run, args=(ctor, args), kwargs=kwargs)
#         self._process.start()
# 
#     @staticmethod
#     def init_and_run(ctor, args, **kwargs):
#         print("ctor:", ctor)
#         print("args:", args[0])
#         worker = ctor(*args[0], **kwargs)
#         worker.run()
# 
#     def join(self, timeout=None):
#         return self._process.join(timeout)
#     # TODO - add a process kill method



class Stitcher(object):

    def __init__(self, conf):
        self._conf = conf

        self._processes_factory = ProcessesFactory(self._conf)
        # Initialize the detectors, matchers and optimizer objects
        #reader_params = conf.get('reader_params', {})
        detector_params = conf.get('detector_params', {})
        matcher_params = conf.get('matcher_params', {})
        #reader_threads = reader_params.get('threads', 1)
        detector_threads = conf.get('detector_threads', 1)
        matcher_threads = conf.get('matcher_threads', 1)
        #assert(reader_threads > 0)
        assert(detector_threads > 0)
        assert(matcher_threads > 0)

        # The design is as follows:
        # - There will be N1 detectors and N2 matchers (each with its own thread/process - TBD)
        # - All matchers have a single queue of tasks which they consume. It will be populated before the optimizer is called
        # - Each matcher needs to detect features in overlapping areas of 2 tiles. To that end, the matcher adds 2 tasks (for each tile and area)
        #   to the single input queue that's shared between all detectors. The detector performs its operation, and returns the result directly to the matcher.
        #   Once the features of the two tiles were computed, the matcher does the matching, and returns the result to a single result queue.
        # TODO - it might be better to use other techniques for sharing data between processes (e.g., shared memory, moving to threads-only, etc.)

        # Set up the detectors and matchers input queue
        self._detectors_in_queue = mp.Queue(maxsize=detector_params.get("queue_max_size", 0))
        self._matchers_in_queue = mp.Queue(maxsize=matcher_params.get("queue_max_size", 0))
        self._detectors_result_queues = [mp.Queue(maxsize=matcher_params.get("queue_max_size", 0)) for i in range(matcher_threads)] # each matcher will receive the detectors results in its own queue
        self._matchers_out_queue = mp.Queue(maxsize=matcher_params.get("queue_max_size", 0)) # Used by the manager to collect the matchers results

        # Set up the pool of detectors
        #self._detectors = [ThreadWrapper(DetectorWorker, (self._processes_factory, self._detectors_in_queue, self._detectors_result_queues)) for i in range(detector_threads)]
        #self._matchers = [ThreadWrapper(MatcherWorker, (self._processes_factory, self._matchers_in_queue, self._matchers_out_queue, self._detectors_result_queues[i], self._detectors_in_queue, i)) for i in range(matcher_threads)]
        self._detectors = [ProcessWrapper(DetectorWorker, (self._processes_factory, self._detectors_in_queue, self._detectors_result_queues)) for i in range(detector_threads)]
        self._matchers = [ProcessWrapper(MatcherWorker, (self._processes_factory, self._matchers_in_queue, self._matchers_out_queue, self._detectors_result_queues[i], self._detectors_in_queue, i)) for i in range(matcher_threads)]
        
#         # Set up the pools of dethreads (each thread will have its own queue to pass jobs around)
#         # Set up the queues
#         #readers_in_queue = queue.Queue(maxsize=reader_params.get("queue_max_size", 0)]))
#         detectors_in_queue = queue.Queue(maxsize=detector_params.get("queue_max_size", 0)]))
#         matchers_in_queue = queue.Queue(maxsize=matcher_params.get("queue_max_size", 0)]))
# 
#         # Set up the threads
# #         self._readers = [TaskRunner(name="reader_%d" % (_ + 1),
# #                                     input_queue=readers_in_queue,
# #                                     output_queue=detectors_in_queue)
# #                            for _ in range(reader_threads)]
#  
#         self._detectors = [TaskRunner(name="detector_%d" % (_ + 1),
#                                       input_queue=detectors_in_queue,
#                                       output_queue=matchers_in_queue,
#                                       init_fn=lambda: FeaturesDetector(conf['detector_type'], **detector_params))
#                            for _ in range(detector_threads)]
# 
#         self._matchers = [TaskRunner(name="matcher_%d" % (_ + 1),
#                                       input_queue=matchers_in_queue,
#                                       init_fn=lambda: FeaturesMatcher(FeaturesDetector(conf['detector_type'], **detector_params), **matcher_params))
#                            for _ in range(detector_threads)]
# 
#         #self._detector = FeaturesDetector(conf['detector_type'], **detector_params)
#         #self._matcher = FeaturesMatcher(self._detector, **matcher_params)
        optimizer_params = conf.get('optimizer_params', {})
        self._optimizer = OptimizerRigid2D(**optimizer_params)



    def close(self):

        print("Closing all matchers")
        for q in self._matchers_in_queue:
            q.put(None)
        for m in self._matchers:
            m.join()
        print("Closing all detectors")
        for i in range(len(self._detectors)):
            self._detectors_in_queue.put(None)
        for d in self._detectors:
            d.join()

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

    def _compute_tile_features(self, tile, bboxes=None):
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
            

    def stitch_section(self, section):
        '''
        Receives a single section (assumes no transformations), stitches all its tiles, and updaates the section tiles' transformations.
        '''

        logger.report_event("stitch_section starting.", log_level=logging.INFO)
        # Compute features

        logger.report_event("Starting feature computation and matching...", log_level=logging.INFO)
        st_time = time.time()
        match_jobs = []
        extend_delta = 50 # TODO - should be a parameter
        for tile1, tile2 in Stitcher._find_overlapping_tiles_gen(section):
            # Add a matching job
            # job = (match_idx, tile1, tile2)
            job = (len(match_jobs), tile1, tile2)
            match_jobs.append((tile1, tile2))
            print('Submitting matching job')
            self._matchers_in_queue.put(job)


        # fetch all the results from the matchers
        match_results_map = {}
        logger.report_event("Collecting matches results", log_level=logging.INFO)
        while len(match_results_map) < len(match_jobs):
            match_idx, filtered_matches = self._matchers_out_queue.get()
            tile1, tile2 = match_jobs[match_idx]
            match_results_map[(tile1.img_fname, tile2.img_fname)] = filtered_matches

        logger.report_event("Starting rigid optimization", log_level=logging.INFO)
        die



        overlap_features = [[None]*len(overlap_bboxes), [None]*len(overlap_bboxes)] # The first sub-list will correspond to tiles_lst1, and the second to tiles_lst2
        for t_basename, bbox_idxs in per_tile_bboxes_idxs.items():
            assert(len(bbox_idxs) > 0)
 


        logger.report_event("Computing features on overlapping areas...", log_level=logging.INFO)
        st_time = time.time()
        overlap_features = [[None]*len(overlap_bboxes), [None]*len(overlap_bboxes)] # The first sub-list will correspond to tiles_lst1, and the second to tiles_lst2
        extend_delta = 50 # TODO - should be a parameter
        for t_basename, bbox_idxs in per_tile_bboxes_idxs.items():
            assert(len(bbox_idxs) > 0)
            # Read the tile (by another thread)
           # self._readers_in_queue.put(

            
                
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
        logger.report_event("stitch_section ended.", log_level=logging.INFO)





    @staticmethod
    def align_img_files(imgs_dir, conf, processes_num):
        # Read the files
        _, imgs = StackAligner.read_imgs(imgs_dir)

        aligner = StackAligner(conf, processes_num)
        return aligner.align_imgs(imgs)


def test_detector(section_dir, conf_fname, workers_num, files_num):
    conf = Stitcher.load_conf_from_file(conf_fname)
    img_fnames = glob.glob(os.path.join(section_dir, '000*', '*.bmp'))[:files_num]

    processes_factory = ProcessesFactory(conf)
    detector_params = conf.get('detector_params', {})
    matcher_params = conf.get('matcher_params', {})
    detector_in_queue = queue.Queue(maxsize=detector_params.get("queue_max_size", 0))
    detector_result_queue = queue.Queue(maxsize=matcher_params.get("queue_max_size", 0))

    # Set up the pool of detectors
    #detector_worker = ThreadWrapper(DetectorWorker, (processes_factory, detector_in_queue))
    detector_workers = [ThreadWrapper(DetectorWorker, (processes_factory, detector_in_queue, [detector_result_queue])) for i in range(workers_num)]

    for img_fname in img_fnames:
        # job = (matcher's result queue idx, tile fname, local area)
        job1 = (0, img_fname, np.array([150, 100]), None)
        print('Submitting job to detector_in_queue')
        detector_in_queue.put(job1)

    for i in range(len(img_fnames)):
        print('Fetching results from detector_result_queue')
        res = detector_result_queue.get()
        print("Detector result:", res[0], len(res[2][0]))
    
    print("Closing all detectors")
    for i in range(workers_num):
        detector_in_queue.put(None)
    for i in range(workers_num):
        detector_workers[i].join()


if __name__ == '__main__':
    section_dir = '/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1/full_image_coordinates.txt'
    section_num = 10
    conf_fname = '../../conf/conf_example.yaml'
    processes_num = 8
    out_fname = './output_stitched_sec{}.json'.format(section_num)
# 
#     section = Section.create_from_full_image_coordinates(section_dir, section_num)
#     conf = Stitcher.load_conf_from_file(conf_fname)
#     stitcher = Stitcher(conf, processes_num)
#     stitcher.stitch_section(section) # will stitch and update the section
# 
#     # Save the transforms to file
#     import json
#     print('Writing output to: {}'.format(out_fname))
#     section.save_as_json(out_fname)
# #     img_fnames, imgs = StackAligner.read_imgs(imgs_dir)
# #     for img_fname, img, transform in zip(img_fnames, imgs, transforms):
# #         # assumption: the output image shape will be the same as the input image
# #         out_fname = os.path.join(out_path, os.path.basename(img_fname))
# #         img_transformed = cv2.warpAffine(img, transform[:2,:], (img.shape[1], img.shape[0]), flags=cv2.INTER_AREA)
# #         cv2.imwrite(out_fname, img_transformed)

# Testing
#    test_detector('/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1', conf_fname, 8, 500)

    logger.start_process('main', 'stitcher.py', [section_dir, conf_fname])
    section = Section.create_from_full_image_coordinates(section_dir, section_num)
    conf = Stitcher.load_conf_from_file(conf_fname)
    stitcher = Stitcher(conf)
    stitcher.stitch_section(section) # will stitch and update the section tiles' transformations

    # TODO - output the section


    logger.end_process('main ending', rh_logger.ExitCode(0))

