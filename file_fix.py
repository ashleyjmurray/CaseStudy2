import csv
import datetime
import math
import time
import collections
from collections import OrderedDict
import os.path
import pandas as pd
import glob

def readFile(file):
    dict = OrderedDict()

    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        i =0;
        for row in reader:
            if(i==0):
                timestamp=row[0]
                #print(timestamp)
                timestamp=float(timestamp)-3600*4 #Time Zone Correction - will need to change depending on time zone!
                #print(timestamp)
            elif(i==1):
                hertz = float(row[0])
            elif(i==2):
                dict[timestamp]=row[0]
            else:
                timestamp = timestamp + 1.0/hertz
                dict[timestamp]=row[0]
            i = i+1.0
    return dict

def formatfile(file, idd, typed):
    EDA = {}
    EDA = readFile(file = file)
    EDA =  {datetime.datetime.utcfromtimestamp(k).strftime('%Y-%m-%d %H:%M:%S.%f'): v for k, v in EDA.items()}
    EDAdf = pd.DataFrame.from_dict(EDA, orient='index', columns=['EDA'])
    EDAdf['EDA'] = EDAdf['EDA'].astype(float)
    
    EDAdf['Datetime'] =EDAdf.index
    EDAdf['Datetime'] = pd.to_datetime(EDAdf['Datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
    EDAdf  = EDAdf.set_index('Datetime')
    
    out_filename = (filesource + '/' + idd + '/' + typed + '.csv')
    EDAdf.to_csv(out_filename, mode='a', header=False)
    print('Done')



def processAcceleration(x,y,z):
    x = float(x)
    y = float(y)
    z = float(z) 
    return {'x':x,'y':y,'z':z}

def readAccFile(file):
    dict = OrderedDict()
    
    with open(file, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0;
        for row in reader:
            if(i == 0):
                timestamp = float(row[0])-3600*4 #Time Zone Correction
            elif(i == 1):    
                hertz=float(row[0])
            elif(i == 2):
                dict[timestamp]= processAcceleration(row[0],row[1],row[2])
            else:
                timestamp = timestamp + 1.0/hertz 
                dict[timestamp] = processAcceleration(row[0],row[1],row[2])
            i = i + 1
        return dict
    
def formatAccfile(file, idd, typed):
    EDA = {}
    EDA = readAccFile(file = file)
    EDA =  {datetime.datetime.utcfromtimestamp(k).strftime('%Y-%m-%d %H:%M:%S.%f'): v for k, v in EDA.items()}
    EDAdf = pd.DataFrame.from_dict(EDA, orient='index', columns=['x', 'y', 'z'])
    
    EDAdf['x'] = EDAdf['x'].astype(float)
    EDAdf['y'] = EDAdf['y'].astype(float)
    EDAdf['z'] = EDAdf['z'].astype(float)
    
    EDAdf['Datetime'] =EDAdf.index
    EDAdf['Datetime'] = pd.to_datetime(EDAdf['Datetime'], format='%Y-%m-%dT%H:%M:%S.%f')
    EDAdf  = EDAdf.set_index('Datetime')
    
    out_filename = (filesource + '/' + idd + typed + '.csv')
    EDAdf.to_csv(out_filename, mode='a', header=False)
    print('Done')

def importIBI(file, idd, typed):
    IBI = pd.read_csv(file, header=None)
    timestampstart = float(IBI[0][0])-3600*4
    IBI[0] = (IBI[0][1:len(IBI)]).astype(float)+timestampstart
    IBI = IBI.drop([0])
    IBI[0] = IBI[0].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.%f'))
    IBI  = IBI.set_index(0)
    
    out_filename = (filesource + '/' + idd + typed + '.csv')
    IBI.to_csv(out_filename, mode='a', header=False)
    print('Done')

filesource = '/WESAD'
x = filesource + '/' + idd + '/' + idd + '_E4_Data/' 

ids = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
for idd in ids:
    for i in ['EDA', 'TEMP', 'HR', 'BVP']:
        temp =  x + idd + i + '.csv'
        formatfile(temp, idd, i)
    formatAccfile(x + idd + 'ACC.csv', idd, 'ACC')
    importIBI(x + idd + 'IBI.csv', idd, 'IBI')






