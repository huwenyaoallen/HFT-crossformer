import pandas as pd
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Data Preprocessing')

parser.add_argument('--raw_folder', type=str, default='./full_dataset/home/ubuntu/quant/tick_data/final_lob_data/min30_all_labeled_sample/', help='raw data path')
parser.add_argument('--save_folder', type=str, default='./30min_datasets_full/data', help='save data path')
parser.add_argument('--save_timestamps_path', type=str, default='./30min_datasets_full/timestamps.npy', help='save timestamps path')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

all_timestamps = set()

for file in os.listdir(args.raw_folder):
    par = pd.read_parquet(os.path.join(args.raw_folder, file))
    print('Processing file: {}'.format(file))
    par = par.sort_values(by='time')

    #drop SellPrice1, LogReturn1, SellPrice2, SellPrice3, LogReturn3
    par = par.drop(columns=['SellPrice1', 'LogReturn1', 'SellPrice2', 'SellPrice3', 'LogReturn3'])
    #read SecurityID and check if they are all the same
    security_id = par['SecurityID'].unique()
    if len(security_id) != 1:
        raise ValueError('SecurityID is not unique')
    par = par.drop(columns=['SecurityID'])

    #drop rows where LogReturn2 is NaN or Inf
    par = par.dropna(subset=['LogReturn2'])
    par = par[par['LogReturn2'] != float('inf')]
    par = par[par['LogReturn2'] != float('-inf')]
    
    #fill nan and inf in other columns (besides time) with last valid observation
    par = par.replace(float('inf'), np.nan)
    par = par.replace(float('-inf'), np.nan)
    par = par.fillna(method='ffill')

    #check if there are still NaN values
    if par.isnull().values.any():
        raise ValueError('There are still NaN values')
    if par.isin([float('inf')]).values.any():
        raise ValueError('There are still Inf values')
    if par.isin([float('-inf')]).values.any():
        raise ValueError('There are still -Inf values')

    #update timestamps
    all_timestamps.update(par['time'].values)

    #save in csv
    par.to_csv(os.path.join(args.save_folder, 'S{}.csv'.format(security_id[0])), index=False)

#save timestamps
timestamps = list(all_timestamps)
#sort timestamps
timestamps.sort()
np.save(args.save_timestamps_path, timestamps)

