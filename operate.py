#import all of the packages
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
from scipy.integrate import simps

#function that calculates the resp features, utilizing the neurokit2 package
#returns a dataframe object that has the resp rate, mean inhale duration, standard deviation of the inhale duration,
#mean exhale duration, standard deviation of the exhale duration, ie ration, and the resp stretch
def resp_features(rsp):
    cleaned = nk.rsp_process(rsp.dropna().reset_index(drop = True))
    mean_rsp_rate = np.mean(cleaned[0]["RSP_Rate"])
    peak_idx = cleaned[1]["RSP_Peaks"]
    trough_idx = cleaned[1]["RSP_Troughs"]
    inhale_time = 0
    exhale_time = 0
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
                        
#function that calculates the emg features. it returns a dataframe containing the mean emg, standard deviation of
#the emg, the number of peaks in the emg, the tenth quantile of the emg, the nintieth quantile of the emg, and
#the range of the emg.
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

#this function processes the temperature and returns the features concerning temperature.
#it returns a dataframe containing the mean temperature, the standard deviation of the temperature, the tenth quantile,
#the nintieth quantile, and the range of the temperature values.
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

#function returns the processes ecg. it utilizes the neurokit2 package to process the results to calculate
#features associated with the heart rate variability. returns a dataframe object.
def process_ecg(df):
    df = df['ECG']
    df = df.dropna()
    processed_data, info = nk.bio_process(ecg=df, sampling_rate=700)
    results = nk.bio_analyze(processed_data, sampling_rate=700)
    return results

#function that returns the features associated with eda. this also utilizes the neurokit2 package to process the
#raw eda signals. The sample rate input depends on whether the eda signals are from the wrist or the chest device.
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

#function that creates the overlapping windows per each subject, and for amusement and stress separately.
#the windows are spaced out approximately per 60 seconds. there are some windows that end up smaller than the others,
#just by the fact that there are not even increments of 60 seconds for both stress and amusement conditions for each of
#the subjects.
def create_windows(df, initial_time):
    indices = []
    samples = []
    length = math.floor(len(df) / 70000)
    counter = 0
    while(length != 0):
        counter = counter + 70000
        indices.append(counter)
        length = length - 1
        
    samples.append(df.loc[0:indices[0]]) #from the first row of the dataframe to the first 60 seconds
    diff = int(indices[1]-indices[0])
    for x in range(len(indices)-1):
        s = df.loc[indices[x]:indices[x+1]]
        samples.append(s)
        temp = int(indices[x+1] - indices[x])
        temp_2 = int(temp/2)
        if temp_2 + diff <= len(df):
            ss = df.loc[temp_2:temp_2+diff]
            samples.append(ss)
        else:
            ss = df.loc[temp_2:len(df)-1]
            samples.append(ss)
        final_s = df.loc[indices[len(indices)-1]:len(df)] #final
        samples.append(final_s)
    return samples
    
#read in the csv file containing only the amusement and stress conditions. this calls the function to create the
#windows.
df = pd.read_csv("amusement_and_stress.csv")
labels = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
windows = []
for i in labels:
    sub_stress = df[(df['subject'] == i) & (df['label'] == 2.0)]
    sub_stress = sub_stress.reset_index()
    
    sub_amusement = df[(df['subject'] == i) & (df['label'] == 3.0)]
    sub_amusement = sub_amusement.reset_index()
    
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

#create the dataframe that will be the final output ocntaining all of the features for the particular window.
y = pd.DataFrame(columns = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD','HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS',
'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d',
'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn',
'HRV_SampEn','eda_mean_chest',
'eda_std_chest',
'eda_min_chest',
'eda_max_chest',
'eda_slope_chest',
'eda_range_chest',
'eda_mean_scl_chest',
'eda_std_scl_chest',
'eda_std_scr_chest',
'eda_scl_corr_chest',
'eda_num_scr_seg_chest',
'eda_sum_startle_mag_chest',
'eda_sum_response_time_chest',
'eda_sum_response_areas_chest',
'eda_mean_wr',
'eda_std_wr',
'eda_min_wr',
'eda_max_wr',
'eda_slope_wr',
'eda_range_wr',
'eda_mean_scl_wr',
'eda_std_scl_wr',
'eda_std_scr_wr',
'eda_scl_corr_wr',
'eda_num_scr_seg_wr',
'eda_sum_startle_mag_wr',
'eda_sum_response_time_wr',
'eda_sum_response_areas_wr',
'resp_rate',
  'mean_inhale_duration',
  'std_inhale_duration',
  'mean_exhale_duration',
  'std_exhale_duration',
  'ie_ratio',
  'resp_stretch',
  'temp_wr_mean',
  'temp_wr_standard_deviation',
  'temp_wr_tenth_quantile',
  'temp_wr_nintieth_quantile',
  'temp_wr_range',
  'temp_chest_mean',
  'temp_chest_standard_deviation',
  'temp_chest_tenth_quantile',
  'temp_chest_nintieth_quantile',
  'temp_chest_range',
  'acc_x_mean',
  'acc_y_mean',
  'acc_z_mean',
  'acc_square_root',
  'subject', 'label'])



#for each of the windows, calculate the features for each of the raw signals. 
for i in range(len(windows)):
    
    subject = windows[i]['subject'].head(1).values[0]
    label = windows[i]['label'].head(1).values[0]
    ecg_data = process_ecg(windows[i])
    #ecg_data['subject'] = windows[i]['subject'].head(1).values[0]
    #ecg_data['window_trial'] = i
    #ecg_data['label'] = windows[i]['label'].head(1).values[0]
    
    ECG_Rate_Mean = ecg_data.ECG_Rate_Mean.values[0]
    HRV_RMSSD = ecg_data.HRV_RMSSD.values[0]
    HRV_MeanNN= ecg_data.HRV_MeanNN.values[0]
    HRV_SDNN= ecg_data.HRV_SDNN.values[0]
    HRV_SDNN= ecg_data.HRV_SDNN.values[0]
    HRV_CVNN= ecg_data.HRV_CVNN.values[0]
    HRV_CVSD= ecg_data.HRV_CVSD.values[0]
    HRV_MedianNN= ecg_data.HRV_MedianNN.values[0]
    HRV_MadNN= ecg_data.HRV_MadNN.values[0]
    HRV_MCVNN= ecg_data.HRV_MCVNN.values[0]
    HRV_IQRNN= ecg_data.HRV_IQRNN.values[0]
    HRV_pNN50= ecg_data.HRV_pNN50.values[0]
    HRV_pNN20= ecg_data.HRV_pNN20.values[0]
    HRV_TINN= ecg_data.HRV_TINN.values[0]
    HRV_HTI= ecg_data.HRV_HTI.values[0]
    HRV_ULF= ecg_data.HRV_ULF.values[0]
    HRV_VLF= ecg_data.HRV_VLF.values[0]
    HRV_LF= ecg_data.HRV_LF.values[0]
    HRV_HF= ecg_data.HRV_HF.values[0]
    HRV_VHF= ecg_data.HRV_VHF.values[0]
    HRV_LFHF= ecg_data.HRV_LFHF.values[0]
    HRV_LFn= ecg_data.HRV_LFn.values[0]
    HRV_HFn= ecg_data.HRV_HFn.values[0]
    HRV_LnHF= ecg_data.HRV_LnHF.values[0]
    HRV_SD1= ecg_data.HRV_SD1.values[0]
    HRV_SD2= ecg_data.HRV_SD2.values[0]
    HRV_SD1SD2= ecg_data.HRV_SD1SD2.values[0]
    HRV_S= ecg_data.HRV_S.values[0]
    HRV_CSI= ecg_data.HRV_CSI.values[0]
    HRV_CVI= ecg_data.HRV_CVI.values[0]
    HRV_CSI_Modified= ecg_data.HRV_CSI_Modified.values[0]
    HRV_PIP= ecg_data.HRV_PIP.values[0]
    HRV_IALS= ecg_data.HRV_IALS.values[0]
    HRV_PSS= ecg_data.HRV_PSS.values[0]
    HRV_PAS= ecg_data.HRV_PAS.values[0]
    HRV_GI= ecg_data.HRV_GI.values[0]
    HRV_SI= ecg_data.HRV_SI.values[0]
    HRV_AI= ecg_data.HRV_AI.values[0]
    HRV_PI= ecg_data.HRV_PI.values[0]
    HRV_C1d= ecg_data.HRV_C1d.values[0]
    HRV_C1a= ecg_data.HRV_C1a.values[0]
    HRV_SD1d= ecg_data.HRV_SD1d.values[0]
    HRV_SD1a= ecg_data.HRV_SD1a.values[0]
    HRV_C2d= ecg_data.HRV_C2d.values[0]
    HRV_C2a= ecg_data.HRV_C2a.values[0]
    HRV_SD2d= ecg_data.HRV_SD2d.values[0]
    HRV_SD2a= ecg_data.HRV_SD2a.values[0]
    HRV_Cd= ecg_data.HRV_Cd.values[0]
    HRV_Ca= ecg_data.HRV_Ca.values[0]
    HRV_SDNNd= ecg_data.HRV_SDNNd.values[0]
    HRV_SDNNa= ecg_data.HRV_SDNNa.values[0]
    HRV_ApEn= ecg_data.HRV_ApEn.values[0]
    HRV_SampEn= ecg_data.HRV_SampEn.values[0]
    HRV_SDSD = ecg_data.HRV_SDSD.values[0]
    
    #EDA Features ~~
    
    eda_data = get_eda_features(windows[i]['EDA_x'])
    #eda_data['subject'] = windows[i]['subject'].head(1).values[0]
    #eda_data["window_trial"] = i
    #eda_data['label'] = windows[i]['label'].head(1).values[0]
    #eda_data['wrist_or_chest'] = 'chest'
    
    eda_mean_chest = eda_data.mean_eda.values[0]
    eda_std_chest = eda_data.std_eda.values[0]
    eda_min_chest = eda_data.min_eda.values[0]
    eda_max_chest = eda_data.max_eda.values[0]
    eda_slope_chest = eda_data.slope_eda.values[0]
    eda_range_chest = eda_data.range_eda.values[0]
    eda_mean_scl_chest = eda_data.mean_scl.values[0]
    eda_std_scl_chest = eda_data.std_scl.values[0]
    eda_std_scr_chest = eda_data.std_scr.values[0]
    eda_scl_corr_chest = eda_data.scl_corr.values[0]
    eda_num_scr_seg_chest = eda_data.num_scr_seg.values[0]
    eda_sum_startle_mag_chest = eda_data.sum_startle_mag.values[0]
    eda_sum_response_time_chest = eda_data.sum_response_time.values[0]
    eda_sum_response_areas_chest = eda_data.sum_response_time.values[0]
    
    
    #y.append(eda_data)
    
    eda_data_wr = get_eda_features(windows[i]['EDA_y'], sample_rate=4)
    #eda_data_wr['subject'] = windows[i]['subject'].head(1).values[0]
    #eda_data_wr["window_trial"] = i
    #eda_data_wr['label'] = windows[i]['label'].head(1).values[0]
    #eda_data_wr['wrist_or_chest'] = 'wrist'
    #y.append(eda_data_wr)
    eda_mean_wr = eda_data_wr.mean_eda.values[0]
    eda_std_wr = eda_data_wr.std_eda.values[0]
    eda_min_wr = eda_data_wr.min_eda.values[0]
    eda_max_wr = eda_data_wr.max_eda.values[0]
    eda_slope_wr = eda_data_wr.slope_eda.values[0]
    eda_range_wr = eda_data_wr.range_eda.values[0]
    eda_mean_scl_wr = eda_data_wr.mean_scl.values[0]
    eda_std_scl_wr = eda_data_wr.std_scl.values[0]
    eda_std_scr_wr = eda_data_wr.std_scr.values[0]
    eda_scl_corr_wr = eda_data_wr.scl_corr.values[0]
    eda_num_scr_seg_wr = eda_data_wr.num_scr_seg.values[0]
    eda_sum_startle_mag_wr = eda_data_wr.sum_startle_mag.values[0]
    eda_sum_response_time_wr = eda_data_wr.sum_response_time.values[0]
    eda_sum_response_areas_wr = eda_data_wr.sum_response_time.values[0]
    
    
    # Resp Features ~~
    resp_data = resp_features(windows[i]['Resp'])
    resp_data['subject'] = windows[i]['subject'].head(1).values[0]
    resp_data['window_trial'] = i
    
    resp_rate = resp_data.resp_rate.values[0]
    mean_inhale_duration = resp_data.mean_inhale_duration.values[0]
    std_inhale_duration = resp_data.std_inhale_duration.values[0]
    mean_exhale_duration = resp_data.mean_exhale_duration.values[0]
    std_exhale_duration = resp_data.std_exhale_duration.values[0]
    ie_ratio = resp_data.ie_ratio.values[0]
    resp_stretch = resp_data.resp_stretch.values[0]
    
    #y.append(resp_data)
    
    #Temp Features ~
    temp_features_wr = process_temp(windows[i], True)
    #temp_features_wr['subject'] = windows[i]['subject'].head(1).values[0]
    #temp_features_wr['window_trial'] = i
    #temp_features_wr['wrist_or_chest'] = 'wrist'
    
    temp_wr_mean = temp_features_wr.temp_mean.values[0]
    temp_wr_standard_deviation = temp_features_wr.temp_standard_deviation.values[0]
    temp_wr_tenth_quantile = temp_features_wr.temp_tenth_quantile.values[0]
    temp_wr_nintieth_quantile = temp_features_wr.temp_nintieth_quantile.values[0]
    temp_wr_range = temp_features_wr.temp_range.values[0]
    
    temp_features = process_temp(windows[i], False)
    #temp_features['subject'] = windows[i]['subject'].head(1).values[0]
    #temp_features['window_trial'] = i
    #temp_features['wrist_or_chest'] = 'chest'
    #y.append(temp_features)
    
    temp_chest_mean = temp_features.temp_mean.values[0]
    temp_chest_standard_deviation = temp_features.temp_standard_deviation.values[0]
    temp_chest_tenth_quantile = temp_features.temp_tenth_quantile.values[0]
    temp_chest_nintieth_quantile = temp_features.temp_nintieth_quantile.values[0]
    temp_chest_range = temp_features.temp_range.values[0]
    
    import math
    acc_x_mean = windows[i].ACC_x.mean()
    acc_y_mean = windows[i].ACC_y.mean()
    acc_z_mean = windows[i].ACC_z.mean()
    temp = math.pow(acc_x_mean, 2) + math.pow(acc_y_mean, 2) + math.pow(acc_z_mean, 2)
    acc_square_root = math.sqrt(temp)
    
    
    y.loc[len(y)] = [ECG_Rate_Mean, HRV_RMSSD, HRV_MeanNN, HRV_SDNN, HRV_SDSD,HRV_CVNN, HRV_CVSD, HRV_MedianNN, HRV_MadNN, HRV_MCVNN, HRV_IQRNN, HRV_pNN50, HRV_pNN20, HRV_TINN, HRV_HTI, HRV_ULF,HRV_VLF, HRV_LF, HRV_HF, HRV_VHF, HRV_LFHF, HRV_LFn,HRV_HFn, HRV_LnHF, HRV_SD1, HRV_SD2, HRV_SD1SD2, HRV_S,HRV_CSI, HRV_CVI,HRV_CSI_Modified, HRV_PIP, HRV_IALS,HRV_PSS, HRV_PAS, HRV_GI, HRV_SI, HRV_AI, HRV_PI, HRV_C1d,HRV_C1a, HRV_SD1d, HRV_SD1a, HRV_C2d, HRV_C2a, HRV_SD2d,HRV_SD2a, HRV_Cd, HRV_Ca, HRV_SDNNd, HRV_SDNNa, HRV_ApEn,HRV_SampEn,eda_mean_chest,eda_std_chest,eda_min_chest,eda_max_chest,eda_slope_chest,eda_range_chest,eda_mean_scl_chest, eda_std_scl_chest,eda_std_scr_chest,eda_scl_corr_chest,eda_num_scr_seg_chest,eda_sum_startle_mag_chest,eda_sum_response_time_chest,eda_sum_response_areas_chest,eda_mean_wr,eda_std_wr,eda_min_wr, eda_max_wr,eda_slope_wr,eda_range_wr,eda_mean_scl_wr,eda_std_scl_wr,eda_std_scr_wr,eda_scl_corr_wr,eda_num_scr_seg_wr,eda_sum_startle_mag_wr,eda_sum_response_time_wr,eda_sum_response_areas_wr, resp_rate,mean_inhale_duration,std_inhale_duration,mean_exhale_duration,std_exhale_duration,ie_ratio,resp_stretch,temp_wr_mean,temp_wr_standard_deviation,temp_wr_tenth_quantile,temp_wr_nintieth_quantile, temp_wr_range,temp_chest_mean,temp_chest_standard_deviation,temp_chest_tenth_quantile,temp_chest_nintieth_quantile,temp_chest_range, acc_x_mean, acc_y_mean, acc_z_mean, acc_square_root, subject, label]
    

y.to_csv("final.csv")
