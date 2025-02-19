# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:07:58 2023

@author: Graybird
"""

def line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the intersection point of two lines
    x_intersect = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / ((x1-x2) * (y3-y4) - (y1-y2) * (x3-x4))
    y_intersect = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / ((x1-x2) * (y3-y4) - (y1-y2) * (x3-x4))
    return x_intersect, y_intersect
