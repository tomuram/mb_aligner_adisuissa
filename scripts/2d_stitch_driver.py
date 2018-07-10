import argparse
import sys
import os
import glob
import json
from mb_aligner.stitching.stitcher import Stitcher
from mb_aligner.dal.section import Section
from rh_logger.api import logger
import logging
import rh_logger

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Runs the stitching algorithm on the given tilespecs input directory")
    parser.add_argument("--ts_dir", metavar="ts_dir", required="True",
                        help="the tilespecs to be stitched directory")
    parser.add_argument("-o", "--output_dir", metavar="output_dir",
                        help="The output directory that will hold the stitched tilespecs (default: ./output_dir)",
                        default="./output_dir")
    parser.add_argument("-c", "--conf_fname", metavar="conf_name",
                        help="The configuration file (yaml). (default: None)",
                        default=None)
    return parser.parse_args(args)

def run_stitcher(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    conf = None
    if args.conf_fname is not None:
        conf = Stitcher.load_conf_from_file(args.conf_fname)
    stitcher = Stitcher(conf)

    # read the inpput tilespecs
    in_ts_fnames = sorted(glob.glob(os.path.join(args.ts_dir, "*.json")))

    for in_ts_fname in in_ts_fnames:
        out_ts_fname = os.path.join(args.output_dir, os.path.basename(in_ts_fname))
        if os.path.exists(out_ts_fname):
            continue

        print("Stitching {}".format(in_ts_fname))
        section = Section.create_from_tilespec(in_ts_fname)
        stitcher.stitch_section(section) 

        # Save the tilespec
        section.save_as_json(out_ts_fname)
#         out_tilespec = section.tilespec
#         import json
#         with open(out_ts_fname, 'wt') as out_f:
#             json.dump(out_tilespec, out_f, sort_keys=True, indent=4)

    del stitcher
        

if __name__ == '__main__':
    args = parse_args()

    logger.start_process('main', '2d_stitcher_driver.py', [args])
    run_stitcher(args)
    logger.end_process('main ending', rh_logger.ExitCode(0))


