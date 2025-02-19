# Module with custom graphical functions

def custom_axes(a=None, f=None):
    import matplotlib.pyplot as plt
    if a is None:
        a = plt.gca()
    if f is None:
        f = plt.gcf()

    a.tick_params(length=0.02)
    a.set_facecolor('white')
    a.set_linewidth(1.5)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.tick_params(direction='in', labelsize=12)
    a.title.set_fontsize(12)
    a.title.set_fontweight('normal')

    # Uncomment the following lines if you want to customize the legend
    # lgd = a.legend()
    # lgd.set_box(False)
    # lgd.get_frame().set_visible(False)
    # lgd.loc = 'lower outside'
    # lgd.orientation = 'horizontal'

    plt.box(False)

def custom_box(d, color='black'):
    import matplotlib.pyplot as plt
    c = custom_color()

    if color == 'black':
        d['boxprops'] = {'facecolor': c.black, 'alpha': 0.2}
        d['whiskerprops'] = {'color': c.black, 'linestyle': '-'}
        d['boxwidth'] = 0.33
        d['capprops'] = {'color': c.black}
        d['medianprops'] = {'color': c.black}
        d['flierprops'] = {'color': c.black}
        d['widths'] = 0.5
    elif color == 'green':
        d['boxprops'] = {'facecolor': c.green5, 'alpha': 0.5}
        d['whiskerprops'] = {'color': c.green5, 'linestyle': '-'}
        d['boxwidth'] = 0.33
        d['capprops'] = {'color': c.green5}
        d['medianprops'] = {'color': c.green5}
        d['flierprops'] = {'color': c.green5}
        d['widths'] = 0.5

    return d

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

def custom_map(mapType='magenta'):
    import numpy as np
    from scipy.interpolate import interp1d
    if mapType == 'sunburst':
        vec = np.array([0, 15, 25, 35, 55, 75, 90, 100])
        hex = ['#000000', '#212728', '#24303D', '#475D75', '#9B3026', '#C16B3A', '#E8E184', '#E0DDBE']
    elif mapType == 'marine':
        vec = np.array([0, 15, 25, 35, 55, 75, 90, 100])
        hex = ['#1A212C', '#2C3E4B', '#34495E', '#00596E', '#1D7872', '#498179', '#71B095', '#F2F2F2']
    elif mapType == 'aqua':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#1C4D48', '#358F87', '#FFFFFF']
    elif mapType == 'green':
        vec = np.array([0, 40, 60, 80, 100])
        hex = ['#000000', '#04481B', '#088F35', '#46AB68', '#FFFFFF']
    elif mapType == 'magenta':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#860557', '#C64597', '#FFFFFF']
    elif mapType == 'teal':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#076B5E', '#47AB9E', '#FFFFFF']
    elif mapType == 'orange':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#BF6B0A', '#FFA53D', '#FFFFFF']
    elif mapType == 'red':
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#74191F', '#A24147', '#FFFFFF']
    elif mapType == 'gray':
        vec = np.array([0, 20, 40, 50, 65, 80, 100])
        hex = ['#000000', '#191919', '#333333', '#666666', '#999999', '#CCCCCC', '#FFFFFF']
    elif mapType == 'pink':
        vec = np.array([0, 50, 75, 100])
        hex = ['#000000', '#BF5058', '#FF6B75', '#FFFFFF']
    elif mapType == 'earth':
        vec = np.array([0, 35, 65, 100])
        hex = ['#191919', '#5A503D', '#FFA53D', '#FFFFFF']
    elif mapType == 'midnight':
        vec = np.array([0, 30, 60, 100])
        hex = ['#191919', '#37414D', '#92A1B3', '#FFF47A']
    elif mapType == 'invert':
        vec = np.array([0, 20, 40, 50, 65, 80, 100])
        hex = ['#FFFFFF', '#CCCCCC', '#999999', '#666666', '#333333', '#191919', '#000000']
    else:
        vec = np.array([0, 40, 60, 100])
        hex = ['#000000', '#860557', '#C64597', '#FFFFFF']

    raw = np.array([int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in hex) / 255.0
    N = 256
    customMap = np.flip(interp1d(vec, raw, axis=0, bounds_error=False, fill_value=(raw[0], raw[-1]))(np.linspace(100, 0, N)), axis=0)

    return customMap

def custom_plot(xvec, ydata, color=None, width=1, style='-'):
    import matplotlib.pyplot as plt
    if len(xvec) != len(ydata):
        raise ValueError("Cannot plot data because x and y vectors are different lengths")

    def_color = 'k'
    def_width = 1
    def_style = '-'

    if color is None:
        color = def_color

    if width is None:
        width = def_width

    if style is None:
        style = def_style

    p = plt.plot(xvec, ydata, linewidth=width, color=color, linestyle=style)
    plt.hold(True)

    return p

def custom_figure(w=8.5, h=5.5):
    import matplotlib.pyplot as plt
    fig_hand = plt.figure(figsize=(w, h))
    left_color = [0, 0, 0]
    right_color = [0, 0, 0]
    fig_hand.set_facecolor('w')
    fig_hand.set_default('axes.prop_cycle', plt.cycler(color=[left_color, right_color]))
    plt.axis('scaled')
    
    return fig_hand


