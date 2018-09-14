from collections import defaultdict
import os
import json
from mb_aligner.dal.tile import Tile
from mb_aligner.dal.mfov import Mfov
import csv
import numpy as np
import cv2
import subprocess

class Section(object):
    """
    Represents a single section (at least one mfov) in the system
    """

    def __init__(self, mfovs_dict, **kwargs):
        self._mfovs_dict = mfovs_dict

        # Initialize default values
        self._layer = kwargs.get("layer", None)

        self._bbox = kwargs.get("bbox", None)
        # initialize values using kwargs
        #elif self._mfovs_dict is not None and len(self._mfovs_dict) > 0:
        #    self._layer = self._mfovs_dict.values()[0].layer

        self._wafer_section = kwargs.get("wafer_section", (None, None))

        self._canonical_section_name = None
            

    @classmethod
    def create_from_tilespec(cls, tilespec, **kwargs):
        """
        Creates a section from a given tilespec
        """
        per_mfov_tiles = defaultdict(list)
        min_xys = []
        max_xys = []
        #min_xy = np.array([np.finfo('float32').max, np.finfo('float32').max])
        #max_xy = np.array([np.finfo('float32').min, np.finfo('float32').min])
        for tile_ts in tilespec:
            per_mfov_tiles[tile_ts["mfov"]].append(Tile.create_from_tilespec(tile_ts))
            min_xys.append(tile_ts['bbox'][::2]) 
            max_xys.append(tile_ts['bbox'][1::2]) 

        min_xy = np.min(min_xys, axis=0)
        max_xy = np.max(max_xys, axis=0)
        bbox = [min_xy[0], max_xy[0], min_xy[1], max_xy[1]]

        layer = int(tilespec[0]["layer"])
        all_mfovs = {mfov_idx:Mfov(mfov_tiles_list) for mfov_idx, mfov_tiles_list in per_mfov_tiles.items()}
        return Section(all_mfovs, layer=layer, bbox=bbox, **kwargs)

    @classmethod
    def _parse_coordinates_file(cls, input_file):
        # Read the relevant mfovs tiles locations
        images_dict = {}
        images = []
        x = []
        y = []
        # Instead of just opening the file, opening the sorted file, so the tiles will be arranged
        sorted_lines = subprocess.check_output('cat "{}" | sort'.format(input_file), shell=True)
        assert(len(sorted_lines) > 0)
        sorted_lines = sorted_lines.decode('ascii').split('\r\n')
        for line in sorted_lines:
            line_data = line.split('\t')
            img_fname = line_data[0].replace('\\', '/')
            # Make sure that the mfov appears in the relevant mfovs
            if not (img_fname.split('/')[0]).isdigit():
                # skip the row
                continue
            img_sec_mfov_beam = '_'.join(img_fname.split('/')[-1].split('_')[:3])
            # Make sure that no duplicates appear
            if img_sec_mfov_beam not in images_dict.keys():
                images.append(img_fname)
                images_dict[img_sec_mfov_beam] = len(images) - 1
                cur_x = float(line_data[1])
                cur_y = float(line_data[2])
                x.append(cur_x)
                y.append(cur_y)
            else:
                # Either the image is duplicated, or a newer version was taken,
                # so make sure that the newer version is used
                prev_img_idx = images_dict[img_sec_mfov_beam]
                prev_img = images[prev_img_idx]
                prev_img_date = prev_img.split('/')[-1].split('_')[-1]
                curr_img_date = img_fname.split('/')[-1].split('_')[-1]
                if curr_img_date > prev_img_date:
                    images[prev_img_idx] = img_fname
                    images_dict[img_sec_mfov_beam] = img_fname
                    cur_x = float(line_data[1])
                    cur_y = float(line_data[2])
                    x[prev_img_idx] = cur_x
                    y[prev_img_idx] = cur_y

        return images, np.array(x), np.array(y)

    @classmethod
    def create_from_full_image_coordinates(cls, full_image_coordinates_fname, layer, tile_size=None, **kwargs):
        """
        Creates a section from a given full_image_coordinates filename
        """
        images, x_locs, y_locs = Section._parse_coordinates_file(full_image_coordinates_fname)
        assert(len(images) > 0)
        section_folder = os.path.dirname(full_image_coordinates_fname)

        # Update tile_size if needed
        if tile_size is None:
            # read the first image
            img_fname = os.path.join(section_folder, images[0])
            img = cv2.imread(img_fname, 0)
            tile_size = img.shape

        # normalize the locations of all the tiles (reset to (0, 0))
        x_locs -= np.min(x_locs)
        y_locs -= np.min(y_locs)


        # Create all the tiles
        per_mfov_tiles = defaultdict(list)
        for tile_fname, tile_x, tile_y, in zip(images, x_locs, y_locs):
            tile_fname = os.path.join(section_folder, tile_fname)
            # fetch mfov_idx, and tile_idx
            split_data = os.path.basename(tile_fname).split('_')
            mfov_idx = int(split_data[1])
            tile_idx = int(split_data[2])
            print('adding mfov_idx %d, tile_idx %d' % (mfov_idx, tile_idx))
            tile = Tile.create_from_input(tile_fname, tile_size, (tile_x, tile_y), layer, mfov_idx, tile_idx)
            per_mfov_tiles[mfov_idx].append(tile)
            
        all_mfovs = {mfov_idx:Mfov(mfov_tiles_list) for mfov_idx, mfov_tiles_list in per_mfov_tiles.items()}

        # Just take any tile's width and height to compute the max_x,y values
        max_x = np.max(x_locs) + tile.width
        max_y = np.max(y_locs) + tile.width
        bbox = [0, max_x, 0, max_y]

        return Section(all_mfovs, layer=layer, bbox=bbox, **kwargs)



    @property
    def layer(self):
        """
        Returns the section layer number
        """
        return self._layer

    @property
    def tilespec(self):
        """
        Returns a tilespec representation of the section
        """
        ret = []
        # Order the mfovs by the mfov index
        sorted_mfov_idxs = sorted(self._mfovs_dict.keys())
        for mfov_idx in sorted_mfov_idxs:
            ret.extend(self._mfovs_dict[mfov_idx].tilespec)
        return ret

    @property
    def bbox(self):
        """
        Returns the bounding box [min_x, max_x, min_y, max_y] of the section
        """
        # TODO - update the section bounding box when the tile's bounding boxes are updated
        return self._bbox

    @property
    def wafer_section(self):
        """
        Returns a tuple of (wafer, section) of the original acquired section (set by the microscope software).
        Note that if the user didn't supply it, it is set to (None, None).
        """
        return self._wafer_section
 
    @property
    def canonical_section_name(self):
        """
        Returns a canonical output name for the section (no suffix).
        The canonical name will be:
        [layer]_W[wafer]_Sec[section], where layer is 4 digits, wafer is 2 digits, and section is 3 digits.
        """
        if self._canonical_section_name is None:
            assert(self._layer is not None)

            wafer, section = self._wafer_section
            self._canonical_section_name = '{}_W{}_Sec{}'.format(str(self._layer).zfill(4), str(wafer).zfill(2), str(section).zfill(3))

        return self._canonical_section_name
 
    @property
    def canonical_section_name_no_layer(self):
        """
        Returns a canonical output name for the section (no suffix) w/o the layer prefix.
        The result name will be:
        W[wafer]_Sec[section], where wafer is 2 digits, and section is 3 digits.
        """
        return self.canonical_section_name[5:]

    def save_as_json(self, out_fname):
        """
        Saves the section as a tilespec
        """
        with open(out_fname, 'w') as out_f:
            json.dump(self.tilespec, out_f, sort_keys=True, indent=4)

    def get_mfov(self, mfov_idx):
        '''
        Returns the mfov of the given mfov_idx
        '''
        return self._mfovs_dict[mfov_idx]

    def mfovs(self):
        '''
        A generator that iterates over all the mfovs in the section
        '''
        mfov_keys = sorted(self._mfovs_dict.keys())
        for mfov_idx in mfov_keys:
            yield self._mfovs_dict[mfov_idx]

    def tiles(self):
        '''
        A generator that iterates over all the tiles in the section
        '''
        for mfov in self.mfovs():
            for tile in mfov.tiles():
                yield tile

       

if __name__ == '__main__':
    section = Section.create_from_full_image_coordinates('/n/home10/adisuis/Harvard/git/rh_aligner/tests/ECS_test9_cropped/images/010_S10R1/full_image_coordinates.txt', 5)

    for mfov in section.mfovs():
        print("Mfov idx: %d" % mfov.mfov_index)
    for tile in section.tiles():
        print("Tile idx: %d (mfov %d)" % (tile.tile_index, tile.mfov_index))

