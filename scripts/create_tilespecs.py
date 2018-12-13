import argparse
import sys
from rh_logger.api import logger
import logging
import rh_logger
import os
import re
import glob
import common
from mb_aligner.dal.section import Section


def sec_dir_to_wafer_section(sec_dir):
    wafer_folder = sec_dir.split(os.sep)[-3]
    section_folder = sec_dir.split(os.sep)[-2]

    m = re.match('.*[W|w]([0-9])+.*', wafer_folder)
    if m is None:
        raise Exception("Couldn't find wafer number from section directory {} (wafer dir is: {})".format(sec_dir, wafer_folder))
    wafer_num = int(m.group(1))

    m = re.match('.*_S([0-9]+)R1+.*', section_folder)
    if m is None:
        raise Exception("Couldn't find section number from section directory {} (section dir is: {})".format(sec_dir, section_folder))
    sec_num = int(m.group(1))

    return wafer_num, sec_num


def get_layer_num(sec_num, initial_layer_num):
    layer_num = sec_num + initial_layer_num - 1
    return layer_num


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Given a wafer's folder, searches for the recent sections, and creates a per-section tilespec file.")
    parser.add_argument("--wafer_folder", metavar="wafer_folder", required=True,
                        help="a folder of a single wafer containing workflow folders")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory name, where each section folder will have a json tilespec there")
    parser.add_argument("-i", "--initial_layer_num", metavar="initial_layer_num", type=int,
                        help="The layer# of the first section in the list. (default: 1)",
                        default=1)
    
    return parser.parse_args(args)


def create_tilespecs(args):

    # parse the workflows directory
    sections_map = common.parse_workflows_folder(args.wafer_folder)

    logger.report_event("Finished parsing sections", log_level=logging.INFO)

    sorted_sec_keys = sorted(list(sections_map.keys()))
    if min(sorted_sec_keys) != 1:
        logger.report_event("Minimal section # found: {}".format(min(sorted_sec_keys)), log_level=logging.WARN)
    
    logger.report_event("Found {} sections in {}".format(len(sections_map), args.wafer_folder), log_level=logging.INFO)
    if len(sorted_sec_keys) != max(sorted_sec_keys):
        logger.report_event("There are {} sections, but maximal section # found: {}".format(len(sections_map), max(sorted_sec_keys)), log_level=logging.WARN)
        missing_sections = [i for i in range(1, max(sorted_sec_keys)) if i not in sections_map]
        logger.report_event("Missing sections: {}".format(missing_sections), log_level=logging.WARN)
    
    logger.report_event("Outputing sections to tilespecs directory: {}".format(args.output_dir), log_level=logging.INFO)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for sec_num in sorted_sec_keys:
        # extract wafer and section# from directory name
        if isinstance(sections_map[sec_num], list):
            wafer_num, sec_num = sec_dir_to_wafer_section(sections_map[sec_num][0])
        else:
            wafer_num, sec_num = sec_dir_to_wafer_section(sections_map[sec_num])
        out_ts_fname = os.path.join(args.output_dir, 'W{}_Sec{}_montaged.json'.format(str(wafer_num).zfill(2), str(sec_num).zfill(3)))
        if os.path.exists(out_ts_fname):
            logger.report_event("Already found tilespec: {}, skipping".format(os.path.basename(out_ts_fname)), log_level=logging.INFO)
            continue

        layer_num = get_layer_num(sec_num, args.initial_layer_num)
        if isinstance(sections_map[sec_num], list):
            # TODO - not implemented yet
            section = Section.create_from_mfovs_image_coordinates(sections_map[sec_num], layer_num)
        else:
            section = Section.create_from_full_image_coordinates(sections_map[sec_num], layer_num)
        section.save_as_json(out_ts_fname)

if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', 'create_tilespecs.py', [args])
    create_tilespecs(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


