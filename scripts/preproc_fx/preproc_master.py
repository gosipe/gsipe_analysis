# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:48:47 2023

@author: Graybird
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import read_xml
def preproc_master(fn):
    dir_xml = os.path.dirname(fn)
    fn_xml = os.path.basename(fn)
    analysis_dir = os.path.join(dir_xml, 'analysis')
    if not os.path.isdir(analysis_dir):
        os.makedirs(analysis_dir)
    plots_dir = os.path.join(dir_xml, 'plots')
    if not os.path.isdir(plots_dir):
        os.makedirs(plots_dir)
    os.chdir(dir_xml)

    # Read XML file and save session info
    sess_info = read_xml(fn_xml)
    fn_new = os.path.splitext(fn_xml)[0]
    anID = fn_new.split('_an')[1].split('_')[0]
    master = {
        'sess_info': sess_info,
        'sess_info.an': anID,
        'sess_info.dir_data': dir_xml
    }
    if 'opto' in fn_xml:
        if 'spon' in fn_xml:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
        elif 'mov' in fn_xml:
            master['sess_info.type'] = 'mov'
            master['sess_info.mov'] = {
                'mov_on': 14,
                'mov_off': 3,
                'mov_num': 7,
                'mov_dur': 2,
                'trial_num': 64
            }
        elif 'grat' in fn_xml:
            master['sess_info.type'] = 'grat'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
        else:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 32
            }
    else:
        if 'spon' in fn_xml:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }
        elif 'mov' in fn_xml:
            master['sess_info.type'] = 'mov'
            master['sess_info.mov'] = {
                'mov_on': 14,
                'mov_off': 3,
                'mov_num': 7,
                'mov_dur': 2,
                'trial_num': 32
            }
        elif 'grat' in fn_xml:
            master['sess_info.type'] = 'grat'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }
        else:
            master['sess_info.type'] = 'spon'
            master['sess_info.grat'] = {
                'grat_on': 2,
                'grat_off': 3,
                'grat_num': 8,
                'trial_num': 16
            }

    # Save masterfile
    np.save(os.path.join(analysis_dir, fn_new + '_master.npy'), master)
