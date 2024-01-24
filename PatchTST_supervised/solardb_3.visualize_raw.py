import os
import numpy as np
import pandas as pd
import warnings
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')




if __name__ == '__main__':
    root_path = './dataset/SolarDB/pre-process'
    save_path = './dataset/SolarDB/pre-process'

    palette = ['', 'limegreen', 'orange', 'dodgerblue', 'red']
    
    for i in range(1, 17):
        if i != 8: continue
        pp = f'pp{i}'
        print('='*30)
        vis_path = f'visualize/SolarDB/raw/{pp}'
        os.makedirs(vis_path, exist_ok=True)

        df = pd.read_csv(os.path.join(root_path, f'{pp}_year.csv'))
        print(pp, 'null', df.isnull().sum().sum())

        start_date      = df.loc[0, 'timestamp'].split(' ')[0]
        year,month,day  = list(map(int, start_date.split('-')))
        date            = datetime.datetime(year, month, day)

        finish_date     = df.loc[len(df)-1, 'timestamp'].split(' ')[0]
        year,month,day  = list(map(int, finish_date.split('-')))
        finish_date     = datetime.datetime(year, month, day)

        while True:
            last_date   = date + datetime.timedelta(days=7)
            _first_date = f'{date.year}-{str(date.month).zfill(2)}-{str(date.day).zfill(2)}'
            _last_date  = f'{last_date.year}-{str(last_date.month).zfill(2)}-{str(last_date.day).zfill(2)}'
            
            first_idx   = df[(f'{_first_date} 00:00:00' <= df['timestamp'])].index[0]
            last_idx    = df[(f'{_last_date} 00:00:00' > df['timestamp'])].index[-1]

            _hour       = df.loc[first_idx:last_idx, 'timestamp'].apply(lambda x: x.split(':')[0][5:])      # if ' 00' in x or ' 12' in x else ' '

            ## visualize per column
            for j, column in enumerate(df.columns):
                if column == 'timestamp': continue

                _values = df.loc[first_idx:last_idx, column]

                plt.clf()
                fig = plt.figure(figsize=(20,9))
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(f'{_first_date} ~ {_last_date}   {column}')
                ax.grid(True, linestyle=':', axis='x')

                ax.plot(_hour, _values, label=column, color=palette[j])
                ax.set_xticks([k for k in range(0, 7*24+1,12)])
                ax.set_xlabel(f'mean: {round(df.loc[first_idx:last_idx, column].mean(), 2)}\nstd: {round(df.loc[first_idx:last_idx, column].std(), 2)}')
                
                plt.legend()
                plt.savefig(f'{vis_path}/{_first_date}-{_last_date}_{column}.png', bbox_inches='tight', pad_inches=0.3)

                
            ## visualize all
            plt.clf()
            fig = plt.figure(figsize=(20,9))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f'{_first_date} ~ {_last_date}   All variables normalized')
            ax.grid(True, linestyle=':', axis='x')

            for j, column in enumerate(df.columns):
                if column == 'timestamp': continue

                _values = df.loc[first_idx:last_idx, column]

                ax.plot(_hour, (_values - _values.min()) / (_values.max() - _values.min()), label=column, color=palette[j])
                ax.set_xticks([k for k in range(0, 7*24+1,12)])

            plt.legend()
            plt.savefig(f'{vis_path}/{_first_date}-{_last_date}_all.png', bbox_inches='tight', pad_inches=0.3)

            date += relativedelta(months=1)
            if date > finish_date: break











