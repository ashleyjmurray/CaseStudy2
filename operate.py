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
 
def process_emg(df):
    df = df.dropna()
    signal, info = nk.emg_process(df['EMG'], sampling_rate=700)
    return info

def process_ecg(df):
    df = df.dropna()
    processed_data, info = nk.bio_process(ecg=df["ECG"], sampling_rate=700)
    results = nk.bio_analyze(processed_data, sampling_rate=700)
    return results

def process_bvp(df):
    signal = df['BVP']
    signal = signal.dropna()
    out = bvp.bvp(signal=signal, sampling_rate=700., show=False)
    heart_rate = list(out[4])
    return heart_rate

from scipy.integrate import *
 
def get_eda_features(eda, sample_rate=700, windex = (0, -1)):
    eda = eda.dropna()
    eda = eda[windex[0]:windex[1]]
    t = (np.arange(0,len(eda)+1)*(1/sample_rate))[windex[0]:windex[1]]
    signals, info = nk.eda_process(eda, sampling_rate=sample_rate)
    cleaned = signals["EDA_Clean"]
    eda_features = nk.eda_phasic(nk.standardize(eda), sampling_rate=700)
    scr = eda_features["EDA_Phasic"]
    scl = eda_features["EDA_Tonic"]
    scl_corrcoeff = np.corrcoef(scl, t)[0,1]
    num_scr_segments = len(info["SCR_Onsets"])
    sum_startle_magnitudes = sum(info["SCR_Amplitude"])
    sum_response_time = sum(info["SCR_RiseTime"])
    peak_integrals = []
    for onset_idx, peak_idx in zip(info["SCR_Onsets"], info["SCR_Peaks"]):
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

def create_windows(df):
    ti = 0
    indices = []
    while(True):
        ti = ti + 60
        x = df[df['time'] == ti]
        if x.empty:
            break
        indices.append(x.index[0])
    samples = []
    for x in range(len(indices)-1):
        s = df.loc[indices[x]:indices[x+1]]
        samples.append(s)
    x = sample(samples, math.ceil(.2*len(samples)))
    return x
    
df = pd.read_csv("data_w_nas.csv")
print("read it in")
d = dict((*df.groupby('subject'),))
print("grouped it")

list_of_df = []
for key, value in d.items():
    y = create_windows(value)
    list_of_df.append(y)

print(list_of_df)
print("created windows")

#example = list_of_df[0][0]

y = []
for o in list_of_df:
    for i in o:
        ecg_data = process_ecg(i)
        print("ecg data", ecg_data)
        ecg_data['subject'] = i['subject']
        y.append(ecg_data)
        bvp_data = process_bvp(i)
        print("bvp data", bvp_data)
        bvp_data['subject'] = i['subject']
        y.append(bvp_data)
        eda_data = get_eda_features(i['EDA'])
        print("eda data", eda_data)
        eda_data['subject'] = i['subject']
        emg_data = process_emg(i)
        print("emgdata", emg_data)
        emg_data['subject'] = i['subject']
final_df = pd.concat(y)
final_df.to_csv("final.csv")

print("done")
