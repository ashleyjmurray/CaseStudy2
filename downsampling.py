import numpy as np
import pandas as pd
import os
from functools import reduce
import gc
from tqdm import tqdm

subject_names = next(os.walk("./WESAD"))[1]
all_data = pd.DataFrame()
for subjname in tqdm(subject_names):
    subject_pickle = pd.read_pickle(f"./WESAD/{subjname}/{subjname}.pkl")
    labels = subject_pickle["label"]
    chest = subject_pickle["signal"]["chest"]
    wrist = subject_pickle["signal"]["wrist"]

    # Overall reference to the chest data
    max_obs = chest["ECG"].shape[0]
    t_array = np.arange(0, max_obs)*(1/700)

    # Making time masks for all other metrics from the wrist sensor
    bvp_obs = wrist["BVP"].shape[0]
    bvp_mask = np.arange(0,bvp_obs)*(1/64)

    acc_obs = wrist["ACC"].shape[0]
    acc_mask = np.arange(0,acc_obs)*(1/32)

    eda_temp_obs = wrist["EDA"].shape[0]
    eda_temp_mask = np.arange(0,eda_temp_obs)*(1/4)

    # Dataframe of all the data from the chest censor at 700hz
    chest_df = pd.DataFrame({"ACC_x":chest["ACC"][:,0].reshape(-1), "ACC_y":chest["ACC"][:,1].reshape(-1), "ACC_z":chest["ACC"][:,2].reshape(-1), "ECG":chest["ECG"].reshape(-1), 
                         "EMG":chest["EMG"].reshape(-1), "EDA":chest["EDA"].reshape(-1), "Temp":chest["Temp"].reshape(-1), "Resp":chest["Resp"].reshape(-1)})
    chest_df["time"] = t_array
    chest_df["label"] = labels

    wrist_acc_df = pd.DataFrame({"time":acc_mask, "wr_ACC_x":wrist["ACC"][:,0].reshape(-1), "wr_ACC_y":wrist["ACC"][:,1].reshape(-1), 
                                "wr_ACC_z":wrist["ACC"][:,2].reshape(-1)})

    bvp_df = pd.DataFrame({"time":bvp_mask, "BVP":wrist["BVP"].reshape(-1)})

    eda_temp_df = pd.DataFrame({"time":eda_temp_mask, "EDA":wrist["EDA"].reshape(-1), "wr_Temp":wrist["TEMP"].reshape(-1)})

    data_frames = [chest_df, wrist_acc_df, bvp_df, eda_temp_df]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],
                                                how='outer'), data_frames)

    downsampled = df_merged.dropna()
    downsampled["subject"] = subjname
    all_data = pd.concat([all_data, downsampled])
    del downsampled
    gc.collect()

all_data.to_csv("downsampled_data.csv", index = False)