import os
import sys

import pandas as pd



    # pv_columns = ['time', 'radiation_horizontal', 'temperature_outdoor', 'radiation_incline', 'temperature_module', 
    #               '6_sisuldong_cumulative_power', '6_sisuldong_hourly_power',
    #               'daily_load']
        

if __name__ == '__main__':
    root_path = 'dataset/GIST'
    vis_path = f'visualize/GIST/raw/'
    os.makedirs(vis_path, exist_ok=True)

    for pp in ['sisuldong']:
        df = pd.read_csv(os.path.join(root_path, f'{pp}.csv'))
        print('='*30)
        print(pp, 'null', df.isnull().sum().sum())

        ## get mean & std of each variable
        _mean, _std = [''], ['']
        _mean.extend(df.loc[:, df.columns[1:]].mean().tolist())
        _std.extend(df.loc[:, df.columns[1:]].std().tolist())
        print()
    


