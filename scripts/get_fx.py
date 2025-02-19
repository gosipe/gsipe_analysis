# Module for custom 'get' functions

def get_dff(f, type='Method2'):
    import numpy as np
    from scipy.stats import kde, zscore, percentileofscore
    # Determines if multiple ROIs are being analyzed
    if f.shape[0] > 1:
        num_roi = f.shape[0]
    else:
        num_roi = 1

    # Preallocate matrices
    dff = np.zeros((num_roi, f.shape[1]))
    f_base = np.zeros(num_roi)

    # Switch method and perform DF/F computation
    if type == 'Method1':
        f_prct = 5  # set percentile for calculating the base threshold
        for roi in range(num_roi):
            f_base[roi] = np.percentile(f[roi, :], f_prct)
            if f_base[roi] <= 1:
                f[roi, :] = f[roi, :] + np.abs(f_base[roi])
                f_base[roi] = np.abs(f_base[roi])
            dff[roi, :] = (f[roi, :] - f_base[roi]) / f_base[roi]
        final_dff = dff
    elif type == 'Method2':
        for roi in range(num_roi):
            ksd = kde.gaussian_kde(f[roi, :])
            xi = np.linspace(np.min(f[roi, :]), np.max(f[roi, :]), 100)
            ksd_values = ksd(xi)
            max_idx = np.argmax(ksd_values)
            f_0 = xi[max_idx]
            dff[roi, :] = (f[roi, :] - f_0) / f_0
        final_dff = dff
    elif type == 'Method3':
        for roi in range(num_roi):
            dff[roi, :] = zscore(f[roi, :], axis=1)
        final_dff = dff

    return final_dff

def get_dff_opto(f, real_frames, type='Method2'):
    import numpy as np
    from scipy.stats import kde, zscore, percentileofscore
    # Determines if multiple ROIs are being analyzed
    if f.shape[0] > 1:
        num_roi = f.shape[0]
    else:
        num_roi = 1

    # Preallocate matrices
    dff = np.zeros((num_roi, f.shape[1]))
    f_base = np.zeros(num_roi)

    # Switch method and perform DF/F computation
    if type == 'Method1':
        f_prct = 5  # set percentile for calculating the base threshold
        for roi in range(num_roi):
            f_base[roi] = np.percentile(f[roi, real_frames], f_prct)
            if f_base[roi] <= 1:
                f[roi, :] = f[roi, :] + np.abs(f_base[roi])
                f_base[roi] = np.abs(f_base[roi])
            dff[roi, :] = (f[roi, :] - f_base[roi]) / f_base[roi]
        final_dff = dff
    elif type == 'Method2':
        for roi in range(num_roi):
            ksd = kde.gaussian_kde(f[roi, real_frames])
            xi = np.linspace(np.min(f[roi, real_frames]), np.max(f[roi, real_frames]), 100)
            ksd_values = ksd(xi)
            max_idx = np.argmax(ksd_values)
            f_0 = xi[max_idx]
            dff[roi, :] = (f[roi, :] - f_0) / f_0
        final_dff = dff
    elif type == 'Method3':
        for roi in range(num_roi):
            dff[roi, :] = zscore(f[roi, :], axis=1)
        final_dff = dff

    return final_dff

def get_event_dff(dff, event_loc, event_win, freq_samp):
    import numpy as np
    from scipy.stats import sem
    """
    Function that calculates activity around events (rows x columns)

    Args:
    - dff: matrix of activity (ROIs x frames)
    - event_loc: frame location of event (number events x location)
    - event_win: time (s) before and after event to analyze
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_dff: dictionary containing event-related information
      - act: activity for each event (ROIs x event frames x event number)
      - event_avg: average activity across events (ROIs x event frames)
      - roi_avg: average activity across ROIs (event frames)
      - sterr: standard error of the mean across ROIs (event frames)
      - diff: difference in activity between before and after each event (ROIs x event number)
      - sort_idx: sorted indices of ROIs based on the difference in activity

    """

    if len(event_win) == 1:
        event_win = [event_win, event_win]

    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames

    frame_side = np.round(event_win * freq_samp).astype(int)  # Calculate number of frames on each side of the event
    event_frames = np.sum(frame_side) + 1  # Calculate number of frames before and after the event

    # Remove events where the windows are outside the frame bounds
    valid_loc = event_loc[
        ~((event_loc - frame_side[0] <= 0) | (event_loc + frame_side[1] > num_frame))
    ]

    event_num = len(valid_loc)  # Calculate remaining number of events
    event_act = np.zeros((num_roi, event_frames, event_num))  # Pre-allocate activity variable

    # Extract the relevant activity from each ROI for each event
    for i, loc in enumerate(valid_loc):
        event_act[:, :, i] = dff[:, loc - frame_side[0]:loc + frame_side[1] + 1]

    event_dff = {}  # Create a dictionary to store the results

    event_dff['act'] = event_act  # Save all data
    event_dff['event_avg'] = np.mean(event_act, axis=2)
    event_dff['roi_avg'] = np.mean(event_act, axis=0)
    event_dff['sterr'] = sem(np.squeeze(np.mean(event_act, axis=0)), axis=0)

    # Find difference between before and after
    event_diff = np.zeros((num_roi, event_num))

    for i in range(event_num):
        event_diff[:, i] = np.mean(event_act[:, :frame_side[0], i], axis=1) - np.mean(event_act[:, frame_side[1]:, i], axis=1)

    # Sort by highest difference
    mean_diff = np.mean(event_diff, axis=1)
    sort_idx = np.argsort(mean_diff)[::-1]
    roi_mean = np.mean(event_act, axis=2)[sort_idx]

    event_dff['diff'] = event_diff
    event_dff['sort_idx'] = sort_idx

    return event_dff

def get_event_gain(neuro_trial, event_loc, event_win, freq_samp):
    import numpy as np
    from scipy.stats import sem
    """
    Function that takes in calcium data sorted as trials x time (rows x columns)
    including the event locations as trial/time pairs, and the size of window to analyze

    Args:
    - neuro_trial: calcium data sorted as trials x time (rows x columns)
    - event_loc: event locations as trial/time pairs
    - event_win: size of the window to analyze
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_gain: dictionary containing event-related information
      - all_gain: gain traces for each event (ROIs x window frames x event number)
      - all_dff: dF/F traces for each event (ROIs x window frames x event number)
      - gain: dictionary containing gain information
        - neuro_mean: mean gain trace across all events (window frames)
        - event_mean: mean gain trace for each event (event frames)
        - stats: standard error of the mean gain trace (window frames)
      - dff: dictionary containing dF/F information
        - neuro_mean: mean dF/F trace across all events (window frames)
        - event_mean: mean dF/F trace for each event (event frames)

    """

    max_frames = neuro_trial.shape[1] * neuro_trial.shape[2]
    num_roi = neuro_trial.shape[0]
    trial_frame = neuro_trial.shape[1]
    num_trials = neuro_trial.shape[2]

    event_num = event_loc.shape[0]
    valid_event = np.ones(event_num, dtype=bool)

    win_frame = 1 + event_win * 2 * freq_samp
    foo_all = np.zeros((num_roi, win_frame))

    for e in range(event_num):
        if event_loc[e] - event_win * freq_samp <= 0 or event_loc[e] + event_win * freq_samp > max_frames:
            valid_event[e] = False

    event_loc = event_loc[valid_event]
    event_num = event_loc.shape[0]

    all_gain = np.zeros((num_roi, win_frame, event_num))
    all_dff = np.zeros((num_roi, win_frame, event_num))

    for r in range(num_roi):
        foo_roi = np.transpose(np.squeeze(neuro_trial[r, :, :]))
        foo_reshape = np.reshape(foo_roi.T, (-1,), order='F')
        _, trace_gain = get_gain(foo_roi)

        for e in range(event_num):
            all_gain[r, :, e] = trace_gain[event_loc[e] - event_win * freq_samp:event_loc[e] + event_win * freq_samp + 1]
            all_dff[r, :, e] = foo_reshape[event_loc[e] - event_win * freq_samp:event_loc[e] + event_win * freq_samp + 1]

    event_gain = {}  # Create a dictionary to store the results

    event_gain['all_gain'] = all_gain
    event_gain['all_dff'] = all_dff

    event_gain['gain'] = {}
    event_gain['dff'] = {}

    event_gain['gain']['neuro_mean'] = np.transpose(np.squeeze(np.mean(all_gain, axis=2)))
    event_gain['dff']['neuro_mean'] = np.transpose(np.squeeze(np.mean(all_dff, axis=2)))
    event_gain['gain']['event_mean'] = np.squeeze(np.mean(all_gain, axis=2))
    event_gain['dff']['event_mean'] = np.squeeze(np.mean(all_dff, axis=2))
    event_gain['gain']['stats'] = sem(event_gain['gain']['neuro_mean'], axis=0)

    return event_gain
    
def get_event_pwcorr(dff, event_loc, event_win, corr_bin, freq_samp, partial_data=None):
    import numpy as np
    from scipy.stats import sem
    import get_pw_corr
    """
    Function that calculates pair-wise correlations between ROIs around events with the possibility of taking into account correlation with the event

    Args:
    - dff: matrix of activity (ROIs x frames)
    - event_loc: frame location of events (number of events x location)
    - event_win: time (s) before and after the event to analyze
    - corr_bin: number of frames to compare for each correlation timepoint
    - freq_samp: frequency of collected frames (Hz)
    - partial_data: partial data to correct with partial pair-wise correlations (optional)

    Returns:
    - event_pwcorr: dictionary containing event-related pairwise correlation information
      - data: pairwise correlation values for each event (event number x event frames)
      - stats: standard error of the mean pairwise correlation values (event frames)

    """

    if partial_data is None:
        partial_flag = False
    elif not np.isnan(partial_data):
        partial_flag = True
    else:
        partial_flag = False

    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames
    frame_side = event_win * freq_samp  # Calculate number of frames on each side of the event
    event_frames = frame_side * 2 + 1  # Calculate number of frames before and after the event

    # Remove events where the windows are outside the frame bounds
    valid_loc = event_loc[~(event_loc - event_win * freq_samp - corr_bin <= 0) & ~(event_loc + event_win * freq_samp + corr_bin >= num_frame)]
    event_num = valid_loc.shape[0]  # Calculate the remaining number of events

    # Pre-allocate activity variable (event number x event frames)
    event_pw = np.zeros((event_num, event_frames))
    slide = np.arange(event_frames)

    if partial_flag:
        itx = 0
        for e in range(event_num):
            start_idx = valid_loc[itx] - frame_side
            for i in range(event_frames):
                corrwin = np.arange(start_idx - corr_bin + slide[i], start_idx + corr_bin + slide[i] + 1)
                _, _, _, avg_corr = get_pw_corr(dff[:, corrwin], partial_data[0, corrwin])
                event_pw[itx, i] = avg_corr['mean']
            itx += 1
    else:
        itx = 0
        for e in range(event_num):
            start_idx = valid_loc[itx] - frame_side
            for i in range(event_frames):
                corrwin = np.arange(start_idx - corr_bin + slide[i], start_idx + corr_bin + slide[i] + 1)
                _, _, _, avg_corr = get_pw_corr(dff[:, corrwin])
                event_pw[itx, i] = avg_corr['mean']
            itx += 1

    event_pwcorr = {}  # Create a dictionary to store the results

    event_pwcorr['data'] = event_pw
    event_pwcorr['stats'] = sem(event_pwcorr['data'], axis=0)

    return event_pwcorr

def get_event_tc(master, idx_roi, event_loc, event_win, win_buffer, freq_samp):
    import numpy as np
    from scipy.stats import sem
    import sterr
    
    """
    Function that compares changes in tuned responses around events

    Args:
    - master: master data structure
    - idx_roi: indices of ROIs to analyze
    - event_loc: frame location of events (number of events x location)
    - event_win: time (s) before and after the event to analyze
    - win_buffer: buffer time (s) to exclude around the event
    - freq_samp: frequency of collected frames (Hz)

    Returns:
    - event_tc: dictionary containing event-related tuning curve information
      - idx: matrix indicating the trial and grating conditions for each event (trial number x grat number x condition)
      - tc: tuning curve responses for each ROI, grating, and condition (ROI x grat number x condition)
      - stats: standard error of the mean tuning curve responses for each condition (condition)

    """

    neuro_grat = master.analysis.neuro.grat
    grat_on = master.sess_info.grat.grat_on
    grat_off = master.sess_info.grat.grat_off
    grat_num = master.sess_info.grat.grat_num
    trial_num = master.sess_info.grat.trial_num

    buffer = win_buffer * freq_samp
    ori_matrix = neuro_grat.ori_matrix  # Matrix of frames when gratings are presented, rows = trials, columns=orientations
    num_roi = len(idx_roi)

    # Pre-allocate idx_ori for determining pupil/grating overlap
    idx_ori = {
        'pre': np.zeros((trial_num, grat_num), dtype=bool),
        'post': np.zeros((trial_num, grat_num), dtype=bool),
        'next': np.zeros((trial_num, grat_num), dtype=bool),
        'base': np.zeros((trial_num, grat_num), dtype=bool)
    }

    # Cycle through events and find which grating presentations they overlap with
    for e in range(event_loc.shape[0]):
        foo_pre = np.arange(event_loc[e] - buffer - event_win * freq_samp, event_loc[e] - buffer)
        foo_post = np.arange(event_loc[e] + buffer, event_loc[e] + buffer + event_win * freq_samp)
        # Cycle through grating number and trials
        for g in range(grat_num):
            for t in range(trial_num):
                foo_stim = np.arange(ori_matrix[t, g], ori_matrix[t, g] + grat_on * freq_samp)
                pre_overlap = np.intersect1d(foo_pre, foo_stim)
                post_overlap = np.intersect1d(foo_post, foo_stim)
                # Cutoff for frame overlap is >=0.5s (i.e., freq_samp/2)
                if len(pre_overlap) >= freq_samp / 2:
                    idx_ori['pre'][t, g] = True
                if len(post_overlap) >= freq_samp / 2:
                    idx_ori['post'][t, g] = True

    serial_ori = idx_ori['post'].T.flatten()
    serial_ori[-1] = 0
    serial_shift = np.roll(serial_ori, 1)
    idx_ori['next'] = serial_shift.reshape((grat_num, trial_num)).T
    idx_ori['base'] = ~(idx_ori['pre'] + idx_ori['post'])

    idx_matrix = np.zeros((trial_num, grat_num, 4), dtype=bool)
    idx_matrix[:, :, 0] = idx_ori['base']
    idx_matrix[:, :, 1] = idx_ori['pre']
    idx_matrix[:, :, 2] = idx_ori['post']
    idx_matrix[:, :, 3] = idx_ori['next']
    
    tc_matrix = np.zeros((num_roi, grat_num, 4))
    for r in range(num_roi):
        roi_tc = neuro_grat.roi[idx_roi[r]].resp.mean_r.diff
        base_tc = roi_tc.copy()
        base_tc[~idx_ori['base']] = np.nan
        base_mean = np.nanmean(base_tc, axis=0)
        delta_base_tc = roi_tc - base_mean
        foo_shift = neuro_grat.roi[idx_roi[r]].tc.mean_r.shift_val
        if foo_shift < 0:
            shift_idx = np.roll(idx_matrix, abs(foo_shift), axis=1)
            shift_tc = np.roll(delta_base_tc, abs(foo_shift), axis=1)
        elif foo_shift > 0:
            shift_idx = np.roll(idx_matrix, (grat_num - foo_shift), axis=1)
            shift_tc = np.roll(delta_base_tc, (grat_num - foo_shift), axis=1)
        else:
            shift_idx = idx_matrix.copy()
            shift_tc = delta_base_tc.copy()
        for j in range(4):
            for g in range(grat_num):
                tc_matrix[r, g, j] = np.nanmean(shift_tc[shift_idx[:, g, j], g], axis=0)
    
    tc_matrix = np.concatenate((tc_matrix[:, -1:, :], tc_matrix), axis=1)
    
    base_stats = sterr(np.squeeze(tc_matrix[:, :, 0]), axis=1)
    pre_stats = sterr(np.squeeze(tc_matrix[:, :, 1]), axis=1)
    post_stats = sterr(np.squeeze(tc_matrix[:, :, 2]), axis=1)
    next_stats = sterr(np.squeeze(tc_matrix[:, :, 3]), axis=1)
    
    event_tc = {}
    event_tc['idx'] = idx_matrix
    event_tc['tc'] = tc_matrix
    event_tc['stats'] = {
        'base': base_stats,
        'pre': pre_stats,
        'post': post_stats,
        'next': next_stats
    }

    return event_tc
                          
def get_gain(trial_data):
    import numpy as np
    """
    Function that takes in calcium data sorted as trials x time (rows x columns),
    including the event locations as trial/time pairs, and the size of the window to analyze.

    Args:
    - trial_data: calcium data sorted as trials x time (2D array)

    Returns:
    - trial_gain: calcium data scaled and centered based on the mean of each trial (2D array)
    - trace_gain: flattened and reshaped version of trial_gain (1D array)

    """

    trial_scale = trial_data * 100
    trial_min = np.min(np.min(trial_scale))

    if 0 <= trial_min < 1:
        dff_offset = trial_min + 1
        trial_offset = trial_scale + dff_offset
    elif trial_min < 0:
        dff_offset = np.abs(trial_min) + 1
        trial_offset = trial_scale + dff_offset
    else:
        trial_offset = trial_scale
        mean_offset = trial_min

    trial_num, trial_time = trial_offset.shape
    trial_gain = np.zeros((trial_num, trial_time))

    trial_mean = np.mean(trial_offset, axis=0)
    for trial in range(trial_num):
        trial_gain[trial, :] = (trial_offset[trial, :] - trial_mean) / trial_mean

    trace_gain = trial_gain.T.flatten()

    return trial_gain, trace_gain

def get_interpol(f, dur, new_rate):
    import numpy as np
    from scipy.interpolate import interp1d
    
    """
    Interpolates the data matrix or vector to a new frame rate.

    Args:
    - f: data matrix or vector to interpolate (2D array or 1D array)
    - dur: total duration of the time series in seconds (float)
    - new_rate: the new interpolated frame rate (float)

    Returns:
    - int_f: interpolated data (2D array)
    - x_vec: corresponding time vector in seconds for int_f (1D array)

    """

    print('Interpolating DFF...')

    num_roi = f.shape[0]  # Number of ROIs (in rows)
    old_frame = f.shape[1]  # Number of frames (in columns)
    new_frame = int(new_rate * dur)  # The new time vector from interpolated frame rate

    # Define the old vector and the new vector
    old_vec = np.linspace(0, dur, old_frame)
    new_vec = np.linspace(0, dur, new_frame)
    int_f = np.zeros((num_roi, new_frame))  # Preallocate int_f size

    # Interpolate the old f to int_f based on new_vec
    for i in range(num_roi):
        f_interp = interp1d(old_vec, f[i, :])
        int_f[i, :] = f_interp(new_vec)

    # Create the new time vector in seconds
    x_vec = np.arange(1 / new_rate, (new_frame + 1) / new_rate, 1 / new_rate)

    print('...Done!')

    return int_f, x_vec

def get_opto_dff(master, act_win):
    import numpy as np
    from scipy.stats import sem
    """
    Calculates activity around optogenetic stimulation.

    Args:
    - master: data master object (or dictionary) containing the required fields
    - act_win: window size in seconds after stimulation to analyze (float)

    Returns:
    - opto_dff: dictionary containing the optogenetic stimulation data

    """

    idx_roi = master.analysis.neuro.grat.osi >= 0.4
    dff = master.data.neuro.dff[idx_roi, :]
    stim_end = master.sess_info.opto.stim_end
    base_end = master.sess_info.opto.base_end
    stim1_end = stim_end[:, 0]
    stim2_end = stim_end[:, 1]

    freq_neuro = master.data.neuro.info.freq_neuro
    num_roi = dff.shape[0]  # Find number of ROIs
    num_frame = dff.shape[1]  # Find number of frames
    frame_win = int(act_win * freq_neuro)  # Calculate number of frames after stimulation to analyze

    # Remove events where the windows are outside the frame bounds
    valid_stim1 = stim1_end[~(stim1_end + frame_win > num_frame)]
    valid_stim2 = stim2_end[~(stim2_end + frame_win > num_frame)]
    valid_base = base_end[~(base_end + frame_win > num_frame)]

    # Process responses to base trials
    base_event_num = valid_base.shape[0]  # Calculate remaining number of events
    base_dff = np.zeros((num_roi, frame_win, base_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(base_event_num):
        base_dff[:, :, i] = dff[:, valid_base[i] + 1 : valid_base[i] + frame_win]
    opto_dff = {"base": {"act": base_dff, "avg": np.mean(base_dff, axis=2), "sterr": sem(np.mean(base_dff, axis=2), axis=1)}}

    # Process responses to first stim type
    stim1_event_num = valid_stim1.shape[0]  # Calculate remaining number of events
    stim1_dff = np.zeros((num_roi, frame_win, stim1_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(stim1_event_num):
        stim1_dff[:, :, i] = dff[:, valid_stim1[i] + 1 : valid_stim1[i] + frame_win]
    opto_dff["stim"] = [{"act": stim1_dff, "avg": np.mean(stim1_dff, axis=2), "sterr": sem(np.mean(stim1_dff, axis=2), axis=1)}]

    # Process responses to second stim type
    stim2_event_num = valid_stim2.shape[0]  # Calculate remaining number of events
    stim2_dff = np.zeros((num_roi, frame_win, stim2_event_num))  # Pre-allocate activity variable (ROIs x event frames x event number)
    # Extract the relevant activity from each ROI for each event
    for i in range(stim2_event_num):
        stim2_dff[:, :, i] = dff[:, valid_stim2[i] + 1 : valid_stim2[i] + frame_win]
        opto_dff["stim"].append({"act": stim2_dff, "avg": np.mean(stim2_dff, axis=2), "sterr": sem(np.mean(stim2_dff, axis=2), axis=1)})
    
    return opto_dff
    
def get_opto_gain(foo_roi):
    import numpy as np
    """
    Calculates gain in calcium data for optogenetic stimulation.

    Args:
    - foo_roi: dictionary containing the calcium data for base and stimulation trials

    Returns:
    - stim1_gain: gain values for stimulation 1 trials (numpy array)
    - stim2_gain: gain values for stimulation 2 trials (numpy array)

    """

    stim_num = foo_roi["stim"][0]["trial"].shape[0]

    foo_base = foo_roi["base"]["trial"] * 100
    foo_stim1 = foo_roi["stim"][0]["trial"] * 100
    foo_stim2 = foo_roi["stim"][1]["trial"] * 100

    base_min = np.min(np.min(foo_base))
    if base_min < 1 and base_min >= 0:
        dff_offset = base_min + 1
        base_offset = foo_base + dff_offset
    elif base_min < 0:
        dff_offset = abs(base_min) + 1
        base_offset = foo_base + dff_offset
    else:
        base_offset = foo_base

    stim1_min = np.min(np.min(foo_stim1))
    if stim1_min < 1 and stim1_min >= 0:
        dff_offset = stim1_min + 1
        stim1_offset = foo_stim1 + dff_offset
    elif stim1_min < 0:
        dff_offset = abs(stim1_min) + 1
        stim1_offset = foo_stim1 + dff_offset
    else:
        stim1_offset = foo_stim1

    stim2_min = np.min(np.min(foo_stim2))
    if stim2_min < 1 and stim2_min >= 0:
        dff_offset = stim2_min + 1
        stim2_offset = foo_stim2 + dff_offset
    elif stim2_min < 0:
        dff_offset = abs(stim2_min) + 1
        stim2_offset = foo_stim2 + dff_offset
    else:
        stim2_offset = foo_stim1

    stim1_gain = np.zeros((8, 200))
    stim2_gain = np.zeros((8, 200))
    base_mean = np.mean(base_offset, axis=0)
    for trial in range(stim_num):
        stim1_gain[trial, :] = (stim1_offset[trial, :] - base_mean) / base_mean
        stim2_gain[trial, :] = (stim2_offset[trial, :] - base_mean) / base_mean

    return stim1_gain, stim2_gain

def get_opto_tc(master):
    import numpy as np
    """
    Compares changes in tuned responses around optogenetic stimulations.

    Args:
    - master: dictionary containing grating responses and analysis parameters

    Returns:
    - opto_tc: dictionary containing the neuronal population mean, standard error,
               index matrix, and tuning curve matrix

    """

    neuro_grat = master["analysis"]["neuro"]["grat"]
    opto_params = master["sess_info"]["opto"]
    freq_neuro = master["data"]["neuro"]["info"]["freq_neuro"]
    grat_on = master["sess_info"]["grat"]["grat_on"]
    grat_off = master["sess_info"]["grat"]["grat_off"]
    grat_num = master["sess_info"]["grat"]["grat_num"]
    trial_num = master["sess_info"]["grat"]["trial_num"]

    ori_matrix = neuro_grat["ori_matrix"]

    stim_grat = opto_params["optostimtrial"]
    stim_freq = opto_params["optoTrialFreq"]
    stim_type = opto_params["stimType"]
    stim_type_num = len(stim_type)

    idx_ori = {
        "pre": np.zeros((trial_num, grat_num), dtype=bool),
        "post": np.zeros((trial_num, grat_num), dtype=bool)
    }
    idx_roi = np.arange(master["analysis"]["neuro"]["grat"]["info"]["num_roi"])
    num_roi = len(idx_roi)

    idx_matrix = np.zeros((trial_num, grat_num, 5))
    trial_num_ct = 1
    stim_num_ct = 1
    for t in range(trial_num):
        stim_trial = trial_num_ct % stim_freq
        if stim_trial:
            if stim_num_ct == 1:
                idx_matrix[t, stim_grat - 1, 1] = 1
                foo_next = stim_grat
                if foo_next > grat_num:
                    idx_matrix[t + 1, 0, 2] = 1
                else:
                    idx_matrix[t, stim_grat, 2] = 1
            elif stim_num_ct == 2:
                idx_matrix[t, stim_grat - 1, 3] = 1
                foo_next = stim_grat
                if foo_next > grat_num:
                    idx_matrix[t + 1, 0, 4] = 1
                else:
                    idx_matrix[t, stim_grat, 4] = 1
            stim_num_ct += 1
            if stim_num_ct > stim_type_num:
                stim_num_ct = 1
        trial_num_ct += 1
    idx_matrix[:, :, 0] = ~(idx_matrix[:, :, 1] + idx_matrix[:, :, 2] + idx_matrix[:, :, 3] + idx_matrix[:, :, 4])

    tc_matrix = np.zeros((num_roi, grat_num, 5))
    non_base = (idx_matrix[:, :, 1] + idx_matrix[:, :, 2] + idx_matrix[:, :, 3] + idx_matrix[:, :, 4])
    tc_matrix[:, :, 0] = np.nan

    for r in range(num_roi):
        foo_roi = neuro_grat["roi"][idx_roi[r]]["resp"]["mean_r"]["shift"]
        foo_mean = np.mean(foo_roi, axis=0)
        foo_roi = foo_roi - foo_mean
        foo_shift = neuro_grat["roi"][idx_roi[r]]["tc"]["mean_r"]["shift_val"]
        if foo_shift < 0:
            shift_idx= np.roll(idx_matrix, abs(foo_shift), axis=1)
        elif foo_shift > 0:
            shift_idx = np.roll(idx_matrix, (grat_num - foo_shift), axis=1)
        else:
            shift_idx = idx_matrix
        for j in range(5):
            for g in range(grat_num):
                tc_matrix[r, g, j] = np.nanmean(foo_roi[np.where(shift_idx[:, g, j]), g])
                tc_matrix = np.concatenate((tc_matrix[:, -1, :], tc_matrix), axis=1)
        opto_tc = {}
        opto_tc["base"] = {"stats": sterr(np.squeeze(tc_matrix[:, :, 0]), axis=1)}
        opto_tc["stim"] = [    {"stim": {"stats": sterr(np.squeeze(tc_matrix[:, :, 1]), axis=1)}},
            {"next": {"stats": sterr(np.squeeze(tc_matrix[:, :, 2]), axis=1)}}
        ]
        opto_tc["stim"].append(
            {"stim": {"stats": sterr(np.squeeze(tc_matrix[:, :, 3]), axis=1)}},
            {"next": {"stats": sterr(np.squeeze(tc_matrix[:, :, 4]), axis=1)}}
        )
        opto_tc["num_roi"] = num_roi
        opto_tc["idx"] = idx_matrix
        opto_tc["tc"] = tc_matrix
        
        return opto_tc
    
def get_osi(tune_curve):
    import numpy as np
    """
    Computes orientation selectivity index (OSI), direction selectivity index (DSI),
    preferred orientation (PO), and preferred direction (PD) from a tuning curve.

    Args:
    - tune_curve: array containing the tuning curve

    Returns:
    - osi: orientation selectivity index
    - dsi: direction selectivity index
    - po: preferred orientation
    - pd: preferred direction

    """

    dir_num = len(tune_curve)

    angle_vec = np.arange(0, 360, 360/dir_num)
    angle_rad = np.deg2rad(angle_vec)

    tc_norm = np.interp(tune_curve, (tune_curve.min(), tune_curve.max()), (0, 1))

    osi = np.abs(np.sum(tc_norm * np.exp(2j * angle_rad)) / np.sum(tc_norm))
    pref_ori = 0.5 * np.angle(np.sum(tc_norm * np.exp(2j * angle_rad)) / np.sum(tc_norm))
    pref_ori = np.rad2deg(pref_ori)

    if pref_ori < 0:
        po = pref_ori + 360
    else:
        po = pref_ori

    dsi = np.abs(np.sum(tc_norm * np.exp(1j * angle_rad)) / np.sum(tc_norm))
    pref_dir = np.angle(np.sum(tc_norm * np.exp(1j * angle_rad)) / np.sum(tc_norm))
    pref_dir = np.rad2deg(pref_dir)

    if pref_dir < 0:
        pd = pref_dir + 360
    else:
        pd = pref_dir

    return osi, dsi, po, pd

def get_pw_corr(roi_data, partial_data=None):
    import numpy as np
    from scipy.spatial import distance
    """
    Computes the average pairwise correlation between all neurons across a window.

    Args:
    - roi_data: array containing the ROI data sorted as neurons x time
    - partial_data: optional array containing partial data sorted as neurons x time

    Returns:
    - pw_corr: pairwise correlation matrix
    - pw_corr_ind: pairwise correlation matrix sorted by average correlation
    - ind_pw: indices of pairwise correlation matrix sorted by average correlation
    - avg_pw_corr: average pairwise correlation

    """

    if partial_data is None:
        roi_num = roi_data.shape[0]
        roi_time = roi_data.shape[1]

        roi_norm = np.interp(roi_data, (roi_data.min(), roi_data.max()), (0, 1))

        pw_corr = np.corrcoef(roi_norm, rowvar=False)

        col_mean = np.mean(pw_corr, axis=1)
        ind_pw = np.argsort(col_mean)
        pw_corr_ind = pw_corr[ind_pw][:, ind_pw]

        top_vals = pw_corr[np.triu_indices(roi_num, k=1)]

        avg_pw_corr = np.std(top_vals)

    else:
        roi_num = roi_data.shape[0]
        roi_time = roi_data.shape[1]

        roi_norm = np.interp(roi_data, (roi_data.min(), roi_data.max()), (0, 1))
        partial_norm = np.interp(partial_data, (partial_data.min(), partial_data.max()), (0, 1))

        pw_corr = distance.pdist(np.vstack((roi_norm, partial_norm)).T, metric='correlation')
        pw_corr = distance.squareform(pw_corr)

        col_mean = np.mean(pw_corr, axis=1)
        ind_pw = np.argsort(col_mean)
        pw_corr_ind = pw_corr[ind_pw][:, ind_pw]

        top_vals = pw_corr[np.triu_indices(roi_num, k=1)]

        avg_pw_corr = np.std(top_vals)

    return pw_corr, pw_corr_ind, ind_pw, avg_pw_corr

def get_quartiles(data_vec):
    import numpy as np
    """
    Computes quartiles and indices of data vector.

    Args:
    - data_vec: 1D array of data

    Returns:
    - quartiles: dictionary containing quartile values and indices

    """

    data_vec = np.squeeze(data_vec)
    if data_vec.ndim == 1:
        data_vec = np.transpose(data_vec)

    q1 = np.percentile(data_vec, 25)
    q2 = np.percentile(data_vec, 50)
    q3 = np.percentile(data_vec, 75)
    q4 = np.percentile(data_vec, 100)

    quartiles = {}
    quartiles['vals'] = np.array([q1, q2, q3, q4])

    idx_q1 = np.where(data_vec < q1)[0]
    idx_q2 = np.where((data_vec >= q1) & (data_vec < q2))[0]
    idx_q3 = np.where((data_vec >= q2) & (data_vec < q3))[0]
    idx_q4 = np.where(data_vec >= q3)[0]

    quartiles['idx_q1'] = idx_q1
    quartiles['idx_q2'] = idx_q2
    quartiles['idx_q3'] = idx_q3
    quartiles['idx_q4'] = idx_q4

    return quartiles

def get_tc(vis_data, grat_on, grat_off, neuro_freq, pval_cutoff):
    import numpy as np
    from scipy.stats import ttest_ind
    from scipy.stats import circshift
    """
    Computes the tuning curves and responses for a given visual stimulus.

    Args:
    - vis_data: 3D array of visual data (num_frames x num_trial x num_grat)
    - grat_on: duration of grating on period in seconds
    - grat_off: duration of grating off period in seconds
    - neuro_freq: neuronal frequency in Hz
    - pval_cutoff: p-value threshold for t-test

    Returns:
    - tc: dictionary containing tuning curve information
    - resp: dictionary containing response information

    """

    num_grat = vis_data.shape[2]
    num_trial = vis_data.shape[1]

    off_last_sec = grat_off - 1
    on_last_sec = grat_on - 1
    off_frames = np.arange(off_last_sec * neuro_freq, neuro_freq * grat_off)
    on_frames = np.arange(
        neuro_freq * grat_off + on_last_sec * neuro_freq,
        neuro_freq * grat_off + neuro_freq * grat_on
    )

    delta_deg = 360 / num_grat
    ori_vec = np.arange(0, 360 - 360 / delta_deg, delta_deg)

    resp = {}
    resp['mean_r'] = {}
    resp['mean_r']['on'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['off'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['diff'] = np.zeros((num_trial, num_grat))
    resp['max_r'] = {}
    resp['max_r']['on'] = np.zeros((num_trial, num_grat))
    resp['max_r']['off'] = np.zeros((num_trial, num_grat))
    resp['max_r']['diff'] = np.zeros((num_trial, num_grat))
    resp['mean_r']['test'] = {}
    resp['mean_r']['test']['idx'] = np.zeros((1, num_grat))
    resp['mean_r']['test']['pval'] = np.zeros((1, num_grat))
    resp['max_r']['test'] = {}
    resp['max_r']['test']['idx'] = np.zeros((1, num_grat))
    resp['max_r']['test']['pval'] = np.zeros((1, num_grat))

    for ori in range(num_grat):
        resp['max_r']['off'][:, ori] = np.squeeze(np.max(vis_data[off_frames, :, ori], axis=0))
        resp['max_r']['on'][:, ori] = np.squeeze(np.max(vis_data[on_frames, :, ori], axis=0))
        resp['max_r']['diff'][:, ori] = resp['max_r']['on'][:, ori] - resp['max_r']['off'][:, ori]
        _, pval = ttest_ind(resp['max_r']['on'][:, ori], resp['max_r']['off'][:, ori])
        resp['max_r']['test']['idx'][0, ori] = pval < pval_cutoff
        resp['max_r']['test']['pval'][0, ori] = pval

        resp['mean_r']['off'][:, ori] = np.squeeze(np.mean(vis_data[off_frames, :, ori], axis=0))
        resp['mean_r']['on'][:, ori] = np.squeeze(np.mean(vis_data[on_frames, :, ori], axis=0))
        resp['mean_r']['diff'][:, ori] = resp['mean_r']['on'][:, ori] - resp['mean_r']['off'][:, ori]
        Args_, pval = ttest_ind(resp['mean_r']['on'][:, ori], resp['mean_r']['off'][:, ori])
        resp['mean_r']['test']['idx'][0, ori] = pval < pval_cutoff
        resp['mean_r']['test']['pval'][0, ori] = pval
        resp['mean_r']['norm'] = (resp['mean_r']['diff'] - np.min(resp['mean_r']['diff'], axis=1, keepdims=True)) / (
        np.max(resp['mean_r']['diff'], axis=1, keepdims=True) - np.min(resp['mean_r']['diff'], axis=1, keepdims=True)
        )
        resp['max_r']['norm'] = (resp['max_r']['diff'] - np.min(resp['max_r']['diff'], axis=1, keepdims=True)) / (
                np.max(resp['max_r']['diff'], axis=1, keepdims=True) - np.min(resp['max_r']['diff'], axis=1, keepdims=True)
        )
        
        resp['max_r']['diff_stat'] = sterr(resp['max_r']['diff'], axis=1)
        resp['mean_r']['diff_stat'] = sterr(resp['mean_r']['diff'], axis=1)
        resp['max_r']['norm_stat'] = sterr(resp['max_r']['norm'], axis=1)
        resp['mean_r']['norm_stat'] = sterr(resp['mean_r']['norm'], axis=1)
        
        tc = {}
        tc['max_r'] = {}
        tc['mean_r'] = {}
        tc['max_r']['diff'] = resp['max_r']['diff_stat']['mean']
        tc['mean_r']['diff'] = resp['mean_r']['diff_stat']['mean']
        tc['max_r']['norm'] = resp['max_r']['norm_stat']['mean']
        tc['mean_r']['norm'] = resp['mean_r']['norm_stat']['mean']
        
        loc_max_max = np.argmax(tc['max_r']['diff'])
        tc['max_r']['pref_grat'] = ori_vec[loc_max_max]
        loc_mean_max = np.argmax(tc['mean_r']['diff'])
        tc['mean_r']['pref_grat'] = ori_vec[loc_mean_max]
        
        tc['mean_r']['diff_norm'] = normalize(tc['mean_r']['diff'], axis=1)
        tc['max_r']['diff_norm'] = normalize(tc['max_r']['diff'], axis=1)
        
        mid_ori = num_grat // 2
        
        tc['max_r']['shift_val'] = loc_max_max - mid_ori
        tc['mean_r']['shift_val'] = loc_mean_max - mid_ori
        
        if tc['max_r']['shift_val'] < 0:
            tc['max_r']['diff_shift'] = circshift(tc['max_r']['diff'], abs(tc['max_r']['shift_val']), axis=1)
            tc['max_r']['norm_shift'] = circshift(tc['max_r']['norm'], abs(tc['max_r']['shift_val']), axis=1)
            resp['max_r']['diff_shift'] = circshift(resp['max_r']['diff'], abs(tc['max_r']['shift_val']), axis=1)
            resp['max_r']['norm_shift'] = circshift(resp['max_r']['norm'], abs(tc['max_r']['shift_val']), axis=1)
        elif tc['max_r']['shift_val'] > 0:
            tc['max_r']['diff_shift'] = circshift(tc['max_r']['diff'], num_grat - tc['max_r']['shift_val'], axis=1)
            tc['max_r']['norm_shift'] = circshift(tc['max_r']['norm'], num_grat - tc['max_r']['shift_val'], axis=1)
            resp['max_r']['diff_shift'] = circshift(resp['max_r']['diff'], num_grat - tc['max_r']['shift_val'], axis=1)
            resp['max_r']['norm_shift'] = circshift(resp['max_r']['norm'], num_grat - tc['max_r']['shift_val'], axis=1)
        else:
            tc['max_r']['diff_shift'] = tc['max_r']['diff']
            tc['max_r']['norm_shift'] = tc['max_r']['norm']
            resp['max_r']['diff_shift'] = resp['max_r']['diff']
            resp['max_r']['norm_shift'] = resp['max_r']['norm']
        if tc['mean_r']['shift_val'] < 0:
            tc['mean_r']['diff_shift'] = circshift(tc['mean_r']['diff'], abs(tc['mean_r']['shift_val']), axis=1)
            tc['mean_r']['norm_shift'] = circshift(tc['mean_r']['norm'], abs(tc['mean_r']['shift_val']), axis=1)
            resp['mean_r']['diff_shift'] = circshift(resp['mean_r']['diff'], abs(tc['mean_r']['shift_val']), axis=1)
            resp['mean_r']['norm_shift'] = circshift(resp['mean_r']['norm'], abs(tc['mean_r']['shift_val']), axis=1)
        elif tc['mean_r']['shift_val'] > 0:
            tc['mean_r']['diff_shift'] = circshift(tc['mean_r']['diff'], num_grat - tc['mean_r']['shift_val'], axis=1)
            tc['mean_r']['norm_shift'] = circshift(tc['mean_r']['norm'], num_grat - tc['mean_r']['shift_val'], axis=1)
            resp['mean_r']['diff_shift'] = circshift(resp['mean_r']['diff'], num_grat - tc['mean_r']['shift_val'], axis=1)
            resp['mean_r']['norm_shift'] = circshift(resp['mean_r']['norm'], num_grat - tc['mean_r']['shift_val'], axis=1)
        else:
            tc['mean_r']['diff_shift'] = tc['mean_r']['diff']
            tc['mean_r']['norm_shift'] = tc['mean_r']['norm']
            resp['mean_r']['diff_shift'] = resp['mean_r']['diff']
            resp['mean_r']['norm_shift'] = resp['mean_r']['norm']
        
        tc['info']['angle_vec'] = ori_vec
        
        angles_deg = np.linspace(0, 360 - delta_deg, 20000)
        angles_rads = angles_deg * (np.pi / 180)
        
        coeff_set, good_fit = VMFit(tc['max_r']['diff_norm'], tc['max_r']['pref_grat'])
        tc['max_r']['vm_fit'] = good_fit
        tc['max_r']['vm_coeff'] = coeff_set
        tc['max_r']['vm_fx'] = VonMisesFunction(coeff_set, angles_rads)
        
        coeff_set, good_fit = VMFit(tc['mean_r']['diff_norm'], tc['mean_r']['pref_grat'])
        tc['mean_r']['vm_fit'] = good_fit
        tc['mean_r']['vm_fx'] = coeff_set
        tc['mean_r']['vm_fx'] = VonMisesFunction(coeff_set, angles_rads)
        
        return tc, resp


