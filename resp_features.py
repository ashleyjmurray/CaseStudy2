# As an example: resp_features(df["Resp"])
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