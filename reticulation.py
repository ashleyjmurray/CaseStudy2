import neurokit2 as nk
import pandas as pd
import numpy as np
from biosppy.signals import bvp

def process_ecg(csv_file):
    subject = pd.read_csv(csv_file)
    processed_data, info = nk.bio_process(ecg=subject["ECG"], sampling_rate=700)
    results = nk.bio_analyze(processed_data, sampling_rate=700)
    return results
    
def process_bvp(csv_file):
	subject = pd.read_csv(csv_file)
	signal = s2['BVP']
	out = bvp.bvp(signal=signal, sampling_rate=700., show=False)
	heart_rate = list(out[4])
	return heart_rate
