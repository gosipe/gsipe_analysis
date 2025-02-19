# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:13:35 2023

@author: Graybird
"""

def custom_color():
    bit_depth = 8

    color = {}

    # Grayscale
    color['gray1'] = [25, 25, 25]
    color['gray2'] = [51, 51, 51]
    color['gray3'] = [102, 102, 102]
    color['gray4'] = [153, 153, 153]
    color['gray5'] = [204, 204, 204]
    color['gray'] = [102, 102, 102]
    color['black'] = [0, 0, 0]

    # Regular Colors
    # Reds
    color['red'] = [138, 29, 37]
    color['red1'] = [73, 16, 20]
    color['red2'] = [94, 20, 25]
    color['red3'] = [116, 25, 31]
    color['red4'] = [138, 29, 37]
    color['red5'] = [162, 65, 71]

    # Oranges
    color['orange'] = [255, 142, 13]
    color['orange1'] = [128, 71, 7]
    color['orange2'] = [191, 107, 10]
    color['orange3'] = [255, 142, 13]
    color['orange4'] = [255, 165, 61]
    color['orange5'] = [255, 187, 110]

    # Yellows
    color['yellow'] = [255, 235, 13]
    color['yellow1'] = [153, 141, 8]
    color['yellow2'] = [204, 188, 10]
    color['yellow3'] = [255, 235, 13]
    color['yellow4'] = [255, 240, 74]
    color['yellow5'] = [255, 244, 122]

    # Greens
    color['green'] = [8, 143, 53]
    color['green1'] = [4, 72, 27]
    color['green2'] = [6, 107, 40]
    color['green3'] = [8, 143, 53]
    color['green4'] = [70, 171, 104]
    color['green5'] = [119, 193, 144]

    # Teals
    color['teal'] = [9, 143, 125]
    color['teal1'] = [5, 72, 63]
    color['teal2'] = [7, 107, 84]
    color['teal3'] = [9, 143, 125]
    color['teal4'] = [71, 171, 158]
    color['teal5'] = [120, 193, 184]

    # Blues
    color['blue'] = [0, 90, 194]
    color['blue1'] = [0, 45, 97]
    color['blue2'] = [0, 68, 146]
    color['blue3'] = [0, 90, 194]
    color['blue4'] = [64, 131, 209]
    color['blue5'] = [115, 164, 221]

    # Purple
    color['purple'] = [86, 35, 194]
    color['purple1'] = [43, 18, 97]
    color['purple2'] = [65, 26, 146]
    color['purple3'] = [86, 35, 194]
    color['purple4'] = [128, 90, 209]
    color['purple5'] = [162, 134, 221]

    # Magenta
    color['magenta'] = [179, 7, 116]
    color['magenta1'] = [90, 4, 58]
    color['magenta2'] = [134, 5, 87]
    color['magenta3'] = [179, 7, 116]
    color['magenta4'] = [198, 69, 151]
    color['magenta5'] = [213, 119, 179]

    # Pinks
    color['pink'] = [255, 107, 117]
    color['pink1'] = [128, 54, 59]
    color['pink2'] = [191, 80, 88]
    color['pink3'] = [255, 107, 117]
    color['pink4'] = [255, 144, 152]
    color['pink5'] = [255, 174, 179]

    # Sage
    color['sage'] = [64, 99, 67]
    color['sage1'] = [32, 50, 34]
    color['sage2'] = [48, 74, 50]
    color['sage3'] = [64, 99, 67]
    color['sage4'] = [112, 138, 114]
    color['sage5'] = [150, 169, 152]

    # Slates
    color['slate'] = [109, 130, 153]
    color['slate1'] = [55, 65, 77]
    color['slate2'] = [82, 98, 115]
    color['slate3'] = [109, 130, 153]
    color['slate4'] = [146, 161, 179]
    color['slate5'] = [175, 186, 199]

    # Brown
    color['brown'] = [120, 106, 81]
    color['brown1'] = [60, 53, 41]
    color['brown2'] = [90, 80, 61]
    color['brown3'] = [120, 106, 81]
    color['brown4'] = [154, 143, 125]
    color['brown5'] = [181, 173, 159]

    # Make field mapping
    for key in color:
        c = color[key]
        c = [comp / (2 ** bit_depth - 1) for comp in c]
        color[key] = c

    return color
