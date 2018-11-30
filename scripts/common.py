import subprocess
import numpy as np

def read_bboxes_grep(ts_fname):
    def parse_bbox_lines(bbox_lines):
        str = ''.join(bbox_lines)
        str = str[str.find('[') + 1:str.find(']')]
        bbox = [float(x) for x in str.split(',')]
        return bbox

    cmd = "grep -A 5 \"bbox\" {}".format(ts_fname)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Parse all bounding boxes in the given json file
    per_tile_bboxes = []
    cur_bbox_lines = []
    for line in iter(p.stdout.readline, ''):
        line = line.decode("utf-8")
        if line == '':
            break
        if line.startswith("--"):
            cur_bbox = parse_bbox_lines(cur_bbox_lines)
            per_tile_bboxes.append(cur_bbox)
            cur_bbox_lines = []
        else:
            cur_bbox_lines.append(line)
    if len(cur_bbox_lines) > 0:
        cur_bbox = parse_bbox_lines(cur_bbox_lines)
        per_tile_bboxes.append(cur_bbox)

    if len(per_tile_bboxes) == 0:
        return None
    per_tile_bboxes = np.array(per_tile_bboxes)
    entire_image_bbox = [np.min(per_tile_bboxes[:, 0]), np.max(per_tile_bboxes[:, 1]),
                         np.min(per_tile_bboxes[:, 2]), np.max(per_tile_bboxes[:, 3])]
    return entire_image_bbox

def read_bboxes_grep_pool(all_files, pool):

    all_bboxes = pool.map(read_bboxes_grep, all_files)
    all_bboxes = np.array(all_bboxes)
    entire_image_bbox = [np.min(all_bboxes[:, 0]), np.max(all_bboxes[:, 1]), np.min(all_bboxes[:, 2]), np.max(all_bboxes[:, 3])]

    return entire_image_bbox

