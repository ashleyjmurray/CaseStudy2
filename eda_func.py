import neurokit2 as nk
import pandas as pd
import numpy as np
from biosppy.signals import bvp
from random import sample
import math
from sklearn import *
import numpy as np
import pandas as pd
import os
from functools import reduce
from tqdm import tqdm
import gc

def get_eda_features(eda, sample_rate=700, windex = (0, -1)):
    if sample_rate == 4:
        sample_rate = 8
        eda = eda.dropna().reset_index(drop = True)
        temp_eda = np.zeros((len(eda)*2))
        for i in range(len(eda)-1):
            temp_eda[i*2] = eda[i]
            temp_eda[i*2+1] = (eda[i] + eda[i+1])/2
        eda = temp_eda
    else:
        eda = eda.dropna().reset_index(drop = True)
    eda = eda[windex[0]:windex[1]]
    t = (np.arange(0,len(eda)+1)*(1/sample_rate))[windex[0]:windex[1]]
    signals, info = nk.eda_process(eda, sampling_rate=sample_rate)
    cleaned = signals["EDA_Clean"]
    eda_features = nk.eda_phasic(nk.standardize(eda), sampling_rate=sample_rate)
    scr = eda_features["EDA_Phasic"]
    scl = eda_features["EDA_Tonic"]
    scl_corrcoeff = np.corrcoef(scl, t)[0,1]
    num_scr_segments = len(np.nan_to_num(info["SCR_Onsets"]))
    sum_startle_magnitudes = sum(np.nan_to_num(info["SCR_Amplitude"]))
    sum_response_time = sum(np.nan_to_num(info["SCR_RiseTime"]))
    peak_integrals = []
    for onset_idx, peak_idx in zip(info["SCR_Onsets"], info["SCR_Peaks"]):
        if np.isnan(onset_idx) or np.isnan(peak_idx):
            continue
        onset_idx, peak_idx = int(onset_idx), int(peak_idx)
        cur_y = cleaned[onset_idx:peak_idx+1]
        cur_integral = simps(cur_y, dx = 1/sample_rate)
        peak_integrals.append(cur_integral)
    sum_response_areas = np.sum(peak_integrals)
    mean_eda = np.mean(eda)
    std_eda = np.std(eda)
    min_eda = np.min(eda)
    max_eda = np.max(eda)
    slope_eda = (eda[len(eda)-1] - eda[0])/len(eda)
    range_eda = max_eda - min_eda
    mean_scl = np.mean(scl)
    std_scl = np.std(scl)
    std_scr = np.std(scr)
    return pd.DataFrame({"mean_eda":[mean_eda], "std_eda":[std_eda], "min_eda":[min_eda], "max_eda":[max_eda], "slope_eda":[slope_eda],
                       "range_eda":[range_eda], "mean_scl":[mean_scl], "std_scl":[std_scl], "std_scr":[std_scr], "scl_corr":[scl_corrcoeff],
                       "num_scr_seg":[num_scr_segments], "sum_startle_mag":[sum_startle_magnitudes],"sum_response_time":[sum_response_time],
                       "sum_response_areas":[sum_response_areas]})