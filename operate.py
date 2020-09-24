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
    df = df['EMG']
    df = df.dropna()
    signal, info = nk.emg_process(df, sampling_rate=700)
    return info

def process_ecg(df):
    df = df['ECG']
    df = df.dropna()
    processed_data, info = nk.bio_process(ecg=df, sampling_rate=700)
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

def create_windows(df, initial_time):
    ti = initial_time
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
    #x = sample(samples, math.ceil(.2*len(samples)))
    return samples
    
df = pd.read_csv("data_w_nas.csv")
df = df[df.label ! = 1.0]
df = df[df.label ! = 5.0]
df = df[df.label ! = 6.0]
df = df[df.label ! = 7.0]
df = df[df.label ! = 4.0]
df = df[df['label'].notna()]

#split by subject, then split by label
labels = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
windows = []
for i in labels:
   sub = df[df['subject'] == i]
   sub_stress = sub[sub['label'] == 2.0]
   sub_amusement = sub[sub['label'] == 3.0]
   sub_stress_temp = sub_stress['time'].head(1)
   stress_initial_time = sub_stress_temp.values[0]
   sub_amusement_temp = sub_amusement['time'].head(1)
   amusement_initial_time = sub_amusement_temp.values[0]
   temp = create_windows(sub_stress, stress_initial_time)
   for t in temp:
    windows.append(t)
   temp2 = create_windows(sub_amusement, amusement_initial_time)
   for t in temp2:
    windows.append(t)

print("created windows")

#example = list_of_df[0][0]

y = []
for i in o:
   ecg_data = process_ecg(i)
   print("ecg data", ecg_data)
   ecg_data['subject'] = i['subject'].head(1).values[0]
   y.append(ecg_data)
   bvp_data = process_bvp(i)
   print("bvp data", bvp_data)
   bvp_data['subject'] = i['subject'].head(1).values[0]
   y.append(bvp_data)
   eda_data = get_eda_features(i['EDA_x'])
   print("eda data", eda_data)
   eda_data['subject'] = i['subject'].head(1).values[0]
   emg_data = process_emg(i)
   print("emgdata", emg_data)
   emg_data['subject'] = i['subject'].head(1).values[0]
final_df = pd.concat(y)
final_df.to_csv("final.csv")

print("done")
