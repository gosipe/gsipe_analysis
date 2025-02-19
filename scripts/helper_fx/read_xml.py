# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:00:09 2023

@author: Graybird
"""

import xml.etree.ElementTree as ET

def read_xml(fn_xml):
    sess_info = {}
    struct_xml = ET.parse(fn_xml).getroot()
    
    sess_info['name'] = fn_xml
    sess_info['date'] = struct_xml.find('PVScan').get('date')
    
    for pv_state_value in struct_xml.findall('.//PVStateValue'):
        key = pv_state_value.get('key')
        value = pv_state_value.get('value')
        
        if key == 'framePeriod':
            sess_info['frame_period'] = float(value)
        elif key == 'dwellTime':
            sess_info['dwellTime'] = float(value)
        elif key == 'linesPerFrame':
            sess_info['lines_per_frame'] = float(value)
        elif key == 'pixelsPerLine':
            sess_info['pixels_per_line'] = float(value)
        elif key == 'objectiveLens':
            sess_info['objective'] = value
        elif key == 'objectiveLensNA':
            sess_info['objective_NA'] = value
        elif key == 'opticalZoom':
            sess_info['optical_zoom'] = float(value)
        elif key == 'rastersPerFrame':
            sess_info['rasters_per_frame'] = float(value)
    
    sequence = struct_xml.find('.//Sequence')
    if sequence is not None and sequence.find('Frame') is not None:
        frames = sequence.findall('Frame')
        sess_info['num_frame'] = len(frames)
        sess_info['frame_time'] = []
        for frame in frames:
            relative_time = float(frame.get('relativeTime'))
            sess_info['frame_time'].append(relative_time)
        
        frame_time1 = float(frames[0].get('absoluteTime'))
        frame_time2 = float(frames[1].get('absoluteTime'))
        sess_info['frame_rate'] = 1 / (frame_time2 - frame_time1)
        last_frame = sess_info['frame_time'][-1]
        sess_info['dur_sess'] = round(last_frame)
    
    elif sequence is not None and sequence[0].find('Frame') is not None:
        frames = sequence[0].findall('Frame')
        sess_info['num_frame'] = len(sequence)
        z_piezo_start = float(struct_xml.find('.//SubindexedValue[@id="20_3_2"]').get('value'))
        z_piezo_second = float(sequence[0].find('.//SubindexedValue[@id="2_3_2"]').get('value'))
        sess_info['zpiezo'] = {
            'zstep': z_piezo_second - z_piezo_start,
            'znum': len(frames)
        }
        sess_info['frame_time'] = []
        for frame in frames:
            relative_time = float(frame.get('relativeTime'))
            sess_info['frame_time'].append(relative_time)
        
        frame_time1 = float(frames[0].get('absoluteTime'))
        frame_time2 = float(sequence[1].find('Frame').get('absoluteTime'))
        sess_info['frame_rate'] = 1 / (frame_time2 - frame_time1)
        last_frame = sess_info['frame_time'][-1]
        sess_info['dur_sess'] = round(last_frame)
    
    return sess_info
