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

def resp_features(rsp):
    cleaned = nk.rsp_process(rsp.dropna().reset_index(drop = True))
    mean_rsp_rate = np.mean(cleaned[0]["RSP_Rate"])
    peak_idx = cleaned[1]["RSP_Peaks"]
    trough_idx = cleaned[1]["RSP_Troughs"]
    for i, (p_idx, t_idx) in enumerate(zip(peak_idx, trough_idx)):
        if i == 0 or i == len(peak_idx)-1:
            continue
        inhale_time = (p_idx - t_idx)*(1/700)
        exhale_time = (cleaned[1]["RSP_Troughs"][i+1]-p_idx)*(1/700)
    mean_inhale_duration = np.mean(inhale_time)
    std_inhale_duration = np.std(inhale_time)
    mean_exhale_duration = np.mean(exhale_time)
    std_exhale_duration = np.std(exhale_time)
    ie_ratio = np.sum(inhale_time)/np.sum(exhale_time)
    stretch = np.max(rsp) - np.min(rsp)
    return pd.DataFrame({"resp_rate":mean_rsp_rate, "mean_inhale_duration":mean_inhale_duration,"std_inhale_duration":std_inhale_duration,
                        "mean_exhale_duration":[mean_exhale_duration], "std_exhale_duration":[std_exhale_duration], "ie_ratio":[ie_ratio],
                        "resp_stretch":[stretch]})
 
def process_emg(df):
    features = pd.DataFrame(columns=['emg_mean', 'emg_standard_deviation', 'emg_num_peaks', "emg_tenth_quantile", "emg_nintieth_quantile", "emg_range"])
    emg = df['EMG']
    peaks, _ = find_peaks(emg)
    num_peaks = len(peaks)
    mean = emg.mean()
    standard_deviation = emg.std()
    tenth_quantile = emg.quantile(.1)
    nintieth_quantile = emg.quantile(.9)
    range_e = emg.max() - emg.min()
    features.loc[len(features)] = [mean, standard_deviation, num_peaks, tenth_quantile, nintieth_quantile, range_e]
    return features

def process_temp(df, wrist):
    features = pd.DataFrame(columns=['temp_mean', 'temp_standard_deviation', 'temp_tenth_quantile', 'temp_nintieth_quantile', 'temp_range'])
    if wrist:
        temp = df['wr_Temp']
    else:
        temp = df['Temp']
    mean = temp.mean()
    standard_deviation = temp.std()
    tenth_quantile = temp.quantile(.1)
    nintieth_quantile = temp.quantile(.9)
    range_t = temp.max() - temp.min()
    features.loc[len(features)] = [mean, standard_deviation, tenth_quantile, nintieth_quantile, range_t]
    return features

def process_ecg(df):
    df = df['ECG']
    df = df.dropna()
    processed_data, info = nk.bio_process(ecg=df, sampling_rate=700)
    results = nk.bio_analyze(processed_data, sampling_rate=700)
    return results

#def process_bvp(df):
#    signal = df['BVP']
#    signal = signal.dropna()
#    out = bvp.bvp(signal=signal, sampling_rate=700., show=False)
#    heart_rate = list(out[4])
#    return heart_rate

from scipy.integrate import simps

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
    return samples
    
df = pd.read_csv("amusement_and_stress.csv")
labels = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
windows = []
for i in labels:
    sub_stress = df[(df['subject'] == i) & (df['label'] == 2.0)]
    sub_amusement = df[(df['subject'] == i) & (df['label'] == 3.0)]
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

y = []
for i in range(len(windows)):
    #df = pd.DataFrame()
    ecg_data = process_ecg(windows[i])
    ecg_data['subject'] = windows[i]['subject'].head(1).values[0]
    ecg_data['window_trial'] = i
    ecg_data['label'] = windows[i]['label'].head(1).values[0]
    y.append(ecg_data)
    
    eda_data = get_eda_features(windows[i]['EDA_x'])
    eda_data['subject'] = windows[i]['subject'].head(1).values[0]
    eda_data["window_trial"] = i
    eda_data['label'] = windows[i]['label'].head(1).values[0]
    eda_data['wrist_or_chest'] = 'chest'
    y.append(eda_data)
    
    eda_data_wr = get_eda_features(windows[i]['EDA_y'], sample_rate=4)
    eda_data_wr['subject'] = windows[i]['subject'].head(1).values[0]
    eda_data_wr["window_trial"] = i
    eda_data_wr['label'] = windows[i]['label'].head(1).values[0]
    eda_data_wr['wrist_or_chest'] = 'wrist'
    y.append(eda_data_wr)
    
    resp_data = resp_features(windows[i]['Resp'])
    resp_data['subject'] = windows[i]['subject'].head(1).values[0]
    resp_data['window_trial'] = i
    y.append(resp_data)
    
    temp_features_wr = process_temp(windows[i], True)
    temp_features_wr['subject'] = windows[i]['subject'].head(1).values[0]
    temp_features_wr['window_trial'] = i
    temp_features_wr['wrist_or_chest'] = 'wrist'
    y.append(temp_features_wr)
    
    temp_features = process_temp(windows[i], False)
    temp_features['subject'] = windows[i]['subject'].head(1).values[0]
    temp_features['window_trial'] = i
    temp_features['wrist_or_chest'] = 'chest'
    y.append(temp_features)
    
final_df = pd.concat(y)
final_df.to_csv("final.csv")
