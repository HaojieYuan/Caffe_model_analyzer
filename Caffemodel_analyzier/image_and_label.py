#!/usr/bin/env python

"""image_and_label.py: get image path and label from list file."""

__author__ = "HaojieYuan"
__email__ = "haojie.d.yuan@gmail.com"
__version__ = "0.1"

import create_labellist

def get(list_file):

    """return image_path list and image_label list,
    Take input list file same as create_imageset.
    Each line should look like 'image_path label\\n' """

    image_list = []
    label_list = []
    
    _list = open(list_file)
    for line in _list:
        tmp = line.split(' ')
        image_list.append(tmp[0])
        label_list.append(int(tmp[1]))


    return image_list, label_list

def build_list(list_config):

    # Create list file
    _file = open(list_config['file_path'], "w+")
    _file.close()

    list_builder = create_labellist.builder(list_config)
    create_labellist.create_labellist(list_builder)

    return list_config['file_path']