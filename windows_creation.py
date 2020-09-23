import pandas as pd
import numpy as np
from random import sample
import math

#returns a list of dataframes

def create_windows(df):
	ti = 0
	indices = []
	while(True):
		ti = ti + 30
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