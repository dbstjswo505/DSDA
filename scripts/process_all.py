# -*- coding: utf-8 -*-

"""
This code was originally developed for research purposes by IDLab at Ghent University - imec, Belgium.
Its performance may not be optimized for specific applications.

For information on its use, applications and associated permission for use, please contact Baptist Vandersmissen (baptist.vandersmissen@ugent.be).

Detailed information on the activities of Ghent University IDLab can be found at http://idlab.technology/.

Copyright (c) Ghent University - imec 2018-2023.

This code can be used to read and process all the files in the data set.
It will store the range-doppler maps together with the raw and thresholded microdoppler signature in the same hdf5 file.

Example:    The following line will compute and save the according 'range_doppler', 'microdoppler', and 'microdoppler_thresholded'
            datasets in all hdf5 files in the <root dir>.

            python3 process_all.py --input <path_to_dataset>
"""

import multiprocessing as mp
import argparse
import glob
import os

from scripts.process import process_file


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Person Identification Dataset Script')
    parser.add_argument('--path', default='./data/idrad', type=str)
    args = parser.parse_args()

    filenames = glob.glob(os.path.join(args.path, '*/*.hdf5'))
    pool = mp.Pool(12)
    pool.map(process_file, filenames)
