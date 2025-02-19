
import numpy as np
import get_osi
import get_tc
from scipy.stats import ttest_ind

def analyze_grat(master):
    osi_thresh = 0.2
    pval_cutoff = 0.01
    grat_on = master['sess_info']['grat']['grat_on']
    grat_off = master['sess_info']['grat']['grat_off']
    grat_num = master['sess_info']['grat']['grat_num']
    trial_num = master['sess_info']['grat']['trial_num']

    neuro_data = master['data']['neuro']['dff']
    freq_neuro = master['data']['neuro']['info']['freq_neuro']
    num_roi = neuro_data.shape[0]

    trial_dur = grat_num * (grat_off + grat_on)
    trial_frames = int(trial_dur * freq_neuro)

    neuro_trial = np.reshape(neuro_data, (num_roi, trial_frames, trial_num))
    neuro_grat = {'neuro_trial': neuro_trial}

    for i in range(num_roi):
        foo = np.squeeze(neuro_trial[i, :, :])
        foo2 = np.reshape(foo, ((grat_off + grat_on) * freq_neuro, grat_num, trial_num))
        neuro_grat['roi'][i] = {}
        neuro_grat['roi'][i]['id'] = i
        neuro_grat['roi'][i]['grat_all'] = []
        for grat_num in range(grat_num):
            neuro_grat['roi'][i]['grat_dir'][grat_num] = {}
            neuro_grat['roi'][i]['grat_dir'][grat_num]['grat_resp'] = np.squeeze(foo2[:, grat_num, :])
            neuro_grat['roi'][i]['grat_all'] = np.concatenate((neuro_grat['roi'][i]['grat_all'], np.squeeze(foo2[:, grat_num, :])), axis=2)
        neuro_grat['roi'][i]['mean_grat'] = np.mean(foo2, axis=2)

    idx_vis = np.zeros(num_roi, dtype=bool)
    for i in range(num_roi):
        for ori_dir in range(grat_num):
            off_resp = np.mean(neuro_grat['roi'][i]['grat_dir'][ori_dir]['grat_resp'][1*freq_neuro:grat_off*freq_neuro, :], axis=0)
            on_resp = np.mean(neuro_grat['roi'][i]['grat_dir'][ori_dir]['grat_resp'][grat_off*freq_neuro:, :], axis=0)
            _, pval = ttest_ind(on_resp, off_resp, alternative='greater')
            neuro_grat['roi'][i]['grat_dir'][ori_dir]['grat_sig'] = pval < pval_cutoff
            neuro_grat['roi'][i]['grat_dir'][ori_dir]['grat_p'] = pval
        if any([neuro_grat['roi'][i]['grat_dir'][grat_dir]['grat_sig'] for grat_dir in range(grat_num)]):
            idx_vis[i] = True
    neuro_grat['num'] = {}
    neuro_grat['num']['vis'] = np.sum(idx_vis)
    foo_idx = np.arange(num_roi)
    neuro_grat['idx'] = {}
    neuro_grat['idx']['vis'] = foo_idx[idx_vis]

    for j in range(num_roi):
        tc, resp = get_tc(neuro_grat['roi'][j]['grat_all'], grat_on, grat_off, freq_neuro, pval_cutoff)
        neuro_grat['roi'][j]['tc'] = tc
        neuro_grat['roi'][j]['resp'] = resp
        idx_osi = np.zeros(num_roi, dtype=bool)
        for i in range(num_roi):
            osi, dsi, po, pd = get_osi(neuro_grat['roi'][i]['tc']['mean_r']['reg'])
            neuro_grat['roi'][i]['osi'] = osi
            neuro_grat['roi'][i]['dsi'] = dsi
            neuro_grat['roi'][i]['po'] = po
            neuro_grat['roi'][i]['pd'] = pd
            if osi > osi_thresh:
                idx_osi[i] = True
        neuro_grat['num']['osi'] = np.sum(idx_osi)
        neuro_grat['idx']['osi'] = foo_idx[idx_osi]
        neuro_grat['osi'] = [neuro_grat['roi'][i]['osi'] for i in range(num_roi)]
        
        neuro_grat['info'] = {}
        neuro_grat['info']['osi_thresh'] = osi_thresh
        neuro_grat['info']['pval_thresh'] = pval_cutoff
        neuro_grat['info']['num_roi'] = num_roi
        
        ori_start = int(freq_neuro * grat_off)
        ori_interval = int(freq_neuro * (grat_on + grat_off))
        ori_trial = np.arange(ori_start, ori_interval * grat_num, ori_interval)
        ori_matrix = np.tile(ori_trial, (trial_num, 1))
        trial_int = np.arange(trial_num) * (ori_interval * grat_num)
        ori_matrix = ori_matrix + np.tile(trial_int[:, np.newaxis], (1, grat_num))
        neuro_grat['ori_matrix'] = ori_matrix
        
        return neuro_grat
