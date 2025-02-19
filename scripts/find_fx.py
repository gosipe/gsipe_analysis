# Module of functions to find events based on specific types of data


def find_astro(astro_data, astro_fract, freq_astro, min_prom, diff_thresh, fract_thresh):
    import numpy as np
    from scipy.signal import find_peaks
    astro_windows = []
    diff_vec = np.concatenate((np.diff(astro_data), [0]))
    astro_amp, astro_loc, _, astro_prom = find_peaks(astro_data, min_prominence=min_prom)

    for i in range(len(astro_amp)):
        b1 = np.where((diff_vec[:astro_loc[i]] <= diff_thresh) & (astro_data[:astro_loc[i]] <= astro_amp[i] - astro_prom[i] * 0.5))[0]
        b1 = b1[-1] if len(b1) > 0 else []
        b2 = astro_loc[i] + np.where((diff_vec[astro_loc[i]:] >= -diff_thresh) & (astro_data[astro_loc[i]:] <= astro_amp[i] - astro_prom[i] * 0.5))[0]
        b2 = b2[0] if len(b2) > 0 else len(astro_data)
        if not np.isnan(b1) and not np.isnan(b2):
            astro_windows.append([b1, b2])

    if len(astro_windows) > 0:
        _, _, aIndx = np.unique(astro_windows[:, 1], return_index=True, return_inverse=True)
        notdups = np.ones(len(astro_windows), dtype=bool)
        for j in range(len(aIndx) - 1):
            if aIndx[j] == aIndx[j + 1]:
                notdups[j + 1] = False
        wins = astro_windows[notdups]
    else:
        wins = np.nan

    if not np.isnan(wins):
        astro_fract_ct = []
        for i in range(wins.shape[0]):
            astro_act = np.max(astro_fract[:, wins[i, 0]:wins[i, 1]], axis=1)
            astro_prop_act = np.sum(astro_act, axis=0) / astro_act.shape[0]
            astro_fract_ct.append(astro_prop_act)
        astro_fract_ct = np.array(astro_fract_ct)
        pass_fract = np.where(astro_fract_ct >= fract_thresh)[0]
        lace_wins = wins[pass_fract]
    else:
        lace_wins = np.nan

    # Find the event statistics
    astro_amp = []
    astro_loc = []
    astro_prom = []
    if not np.isnan(lace_wins):
        for winEvent in range(lace_wins.shape[0]):
            amp, loc, _, prom = find_peaks(astro_data[lace_wins[winEvent, 0]:lace_wins[winEvent, 1]])
            astroMax = np.argmax(amp)
            astro_amp.append(amp[astroMax])
            astro_loc.append(loc[astroMax] + lace_wins[winEvent, 0] - 1)
            astro_prom.append(prom[astroMax])
        astro_dur = (lace_wins[:, 1] - lace_wins[:, 0]) / freq_astro
        astro_num = len(astro_amp)
        astro_freq = astro_num / (len(astro_data) / freq_astro)
        astro_win = lace_wins
    elif np.isnan(lace_wins).any() or lace_wins.size == 0:
        astro_amp = np.nan
        astro_loc = np.nan
        astro_win = np.nan
        astro_freq = np.nan
        astro_dur = np.nan
        astro_prom = np.nan
        astro_num = 0

    return astro_num, astro_freq, astro_amp, astro_prom, astro_loc, astro_dur, astro_win

def find_event(event_data, freq_data, min_amp, deriv_thresh):
    import numpy as np
    from scipy.signal import find_peaks
    # Setup and maintenance
    event_windows = []

    # Calculate derivative of high envelope and add zero
    event_dt = np.diff(event_data)
    event_dt = np.concatenate((event_dt, [0]))

    # Find peaks of envH to specify the overall events
    event_amp, event_loc = find_peaks(event_data, height=min_amp)

    # Find the beginning and end of the event using threshold and derivative
    for i in range(len(event_amp)):
        b1 = np.where((event_dt[:event_loc[i]] <= deriv_thresh) & (event_data[:event_loc[i]] <= 0))[0]
        b2 = event_loc[i] + np.where((event_dt[event_loc[i]:] >= deriv_thresh) & (event_data[event_loc[i]:] <= 0))[0][0]
        if b2 >= len(event_data):
            b2 = len(event_data) - 1
        if len(b1) > 0 and len(b2) > 0:
            event_windows.append([b1[-1], b2])

    # Find unique windows
    if len(event_windows) > 0:
        event_windows = np.array(event_windows)
        _, unique_indices = np.unique(event_windows[:, 1], return_index=True)
        wins = event_windows[unique_indices]
    else:
        wins = np.array([])

    if wins.size > 0:
        event_num = wins.shape[0]
        event_win = wins
        event_dur = (wins[:, 1] - wins[:, 0]) / freq_data
    else:
        event_num = 0
        event_win = np.array([])
        event_dur = np.array([])

    return event_num, event_win, event_dur

def find_pupil(pupil_data, freq_pupil, min_prom, diff_thresh):
    import numpy as np
    from scipy.signal import find_peaks
    pupil_windows = []

    pupil_smooth = np.transpose(np.convolve(pupil_data, np.ones(freq_pupil*2), mode='same') / (freq_pupil*2))
    pupil_diff = np.concatenate((np.diff(pupil_smooth), [0]))

    # Find peaks of data vector to specify the overall events
    p_amps, p_locs, _, p_proms = find_peaks(pupil_smooth, prominence=min_prom)

    # Find the beginning and end of the event using threshold and derivative
    for i in range(len(p_amps)):
        b1 = np.where((pupil_diff[:p_locs[i]] <= diff_thresh) & (pupil_data[:p_locs[i]] <= p_amps[i] - p_proms[i]*0.5))[0]
        b2 = p_locs[i] + np.where((pupil_diff[p_locs[i]:] >= diff_thresh) & (pupil_data[p_locs[i]:] <= p_amps[i] - p_proms[i]*0.5))[0][0]
        if b2 >= len(pupil_data):
            b2 = len(pupil_data) - 1
        if len(b1) > 0 and len(b2) > 0:
            pupil_windows.append([b1[-1], b2])

    # Find unique windows
    if len(pupil_windows) > 0:
        pupil_windows = np.array(pupil_windows)
        _, unique_indices = np.unique(pupil_windows[:, 1], return_index=True)
        wins = pupil_windows[unique_indices]
    else:
        wins = np.array([])

    # Find the event statistics
    pupil_amp = []
    pupil_loc = []
    pupil_peak = []
    pupil_prom = []
    if np.isnan(wins):
        pupil_amp = np.array([])
        pupil_loc = np.array([])
        pupil_peak = np.array([])
        pupil_dur = np.array([])
        pupil_num = 0
        pupil_win = np.array([])
        pupil_prom = np.array([])
    elif not np.isnan(wins):
        val_win_index = np.ones(wins.shape[0], dtype=bool)
        for win_event in range(wins.shape[0]):
            amp, loc, _, prom = find_peaks(pupil_data[wins[win_event, 0]:wins[win_event, 1]])
            if len(amp) > 0:
                eveMax = np.argmax(amp)
                pupil_amp.append(amp[eveMax])
                pupil_loc.append(loc[eveMax] + wins[win_event, 0])
                pupil_peak.append(len(amp))
                pupil_prom.append(prom[eveMax])
            elif len(amp) == 0:
                val_win_index[win_event] = False
        wins = wins[val_win_index]

        pupil_dur = (wins[:, 1] - wins[:, 0]) / freq_pupil
        pupil_num = len(pupil_amp)
        pupil_win = wins

    return pupil_num, pupil_amp, pupil_prom, pupil_loc, pupil_peak, pupil_win, pupil_dur

def find_wheel(wheel_data, freq_wheel, min_height, min_dur, min_int):
    import numpy as np
    from scipy.signal import find_peaks, convolve
    wheel_event = wheel_data >= min_height  # find values above min_height
    event_thresh = convolve(wheel_event, np.ones(min_dur * freq_wheel), mode='same')  # Take a moving average forward in time over min_dur
    event_thresh = event_thresh >= min_dur  # Values of min_dur indicate a period of at least min_dur above min_height
    event_indices = np.where(event_thresh)[0]  # Find position of events

    if event_indices.size > 0:
        wheel_win = np.zeros((event_indices.size, 2))
        wheel_win[0, 0] = event_indices[0]  # first event index is the start of the first event
        event_diff_indices = np.where(np.diff(event_indices) >= (min_int + min_dur) * freq_wheel)[0]
        wheel_win[1:, 0] = event_indices[event_diff_indices + 1]  # find distance between events => min_int + min_dur
        wheel_win[:-1, 1] = event_indices[event_diff_indices] + min_dur * freq_wheel
        wheel_win[-1, 1] = event_indices[-1] + min_dur * freq_wheel  # The end of the last event period

        if wheel_win[-1, 1] > len(wheel_data):
            wheel_win[-1, 1] = len(wheel_data)  # If end of last window is beyond the end of recording

        wheel_dur = wheel_win[:, 1] - wheel_win[:, 0]  # duration of windows
        wheel_num = wheel_win.shape[0]  # Number of locomotion events

        wheel_amp = np.zeros(wheel_num)
        wheel_loc = np.zeros(wheel_num)
        for i in range(wheel_num):
            wheel_amp[i] = np.max(wheel_data[int(wheel_win[i, 0]):int(wheel_win[i, 1])])  # Maximum wheel amplitude in each window
            wheel_loc[i] = np.argmax(wheel_data[int(wheel_win[i, 0]):int(wheel_win[i, 1])])

    else:
        wheel_num = 0
        wheel_win = np.array([np.nan])
        wheel_dur = np.array([np.nan])
        wheel_loc = np.array([np.nan])
        wheel_amp = np.array([np.nan])

    return wheel_num, wheel_amp, wheel_loc, wheel_win, wheel_dur

