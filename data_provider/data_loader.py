import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import copy
import pickle

warnings.filterwarnings('ignore')


class Dataset_DKASC_single(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='79-Site_DKA-M6_A-Phase.csv', target='Active_Power',
                 scale=True, timeenc=0, freq='h', domain='source'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.domain = domain

        self.root_path = root_path
        self.data_path = data_path

        self.DATASET_SPLIT_YEAR = {
            '79-Site_DKA-M6_A-Phase.csv'    : [2009, 2020,  2021, 2022,  2023, 2023],   # 7.0kW, CdTe, Fixed, First-Solar
            '91-Site_DKA-M9_B-Phase.csv'    : [2014, 2020,  2021, 2021,  2022, 2022],   # 10.5kW, mono-Si, Tracker: Dual              # 1A
            '87-Site_DKA-M9_A+C-Phases.csv' : [2010, 2020,  2021, 2021,  2022, 2022],   # 23.4kW, mono-Si, Tracker: Dual              # 1B
            '78-Site_DKA-M11_3-Phase.csv'   : [2010, 2020,  2021, 2021,  2022, 2022],   # 26.5kW, mono-Si, Tracker: Dual              # 2
            # '91-Site_DKA-M9_B-Phase.csv'    : [2020, 2021,  2020, 2021,  2022, 2022],   # 10.5kW, mono-Si, Tracker: Dual              # 1A, other papers
            # '87-Site_DKA-M9_A+C-Phases.csv' : [2014, 2017,  2018, 2018,  2018, 2018],   # 23.4kW, mono-Si, Tracker: Dual              # 1B
            # '78-Site_DKA-M11_3-Phase.csv'   : [2020, 2021,  2020, 2021,  2022, 2022],   # 26.5kW, mono-Si, Tracker: Dual              # 2
            '70-Site_3-BP-Solar.csv'        : [2009, 2020,  2021, 2021,  2022, 2022],   # 5.0kW, poly-Si, Fixed, Roof Mounted         # 3
            '79-Site_7-First-Solar.csv'     : [2009, 2020,  2021, 2021,  2022, 2022],   # 7.0kW, CdTe, Fixed
            '93-Site_8-Kaneka.csv'          : [2009, 2020,  2021, 2021,  2022, 2022],   # 6.0kW, Amorphous Silicon, Fixed
            '85-Site_10-SunPower.csv'       : [2010, 2020,  2021, 2021,  2022, 2022],   # 5.8kW, mono-Si, Fixed
            '55-Site_29-CSUN.csv'           : [2014, 2020,  2021, 2021,  2022, 2022],   # 6.0kW, poly-Si, Fixed
            '73-Site_35-Elkem.csv'          : [2014, 2020,  2021, 2021,  2022, 2022],   # 5.5kW, poly-Si, Fixed
            '72-Site_26-Q-CELLS.csv'        : [2014, 2020,  2021, 2021,  2022, 2022],   # 5.5kW, poly-Si, Fixed
            '56-Site_30-Q-CELLS.csv'        : [2014, 2020,  2021, 2021,  2022, 2022],   # 5.6kW, poly-Si, Fixed
            '213-Site_24-Q-CELLS.csv'       : [2017, 2020,  2021, 2021,  2022, 2022],   # 6.1kW, poly-Si, Fixed
            '59-Site_38-Q-CELLS.csv'        : [2017, 2020,  2021, 2021,  2022, 2022],   # 5.9kW, mono-Si, Fixed
            '212-Site_25-Hanwha-Solar.csv'  : [2017, 2020,  2021, 2021,  2022, 2022],   # 5.8kW, poly-Si, Fixed
        }
        
        # Create scaler for each input_channels
        self.input_channels = ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        self.input_channels = self.input_channels + [self.target]
        for i in self.input_channels:
            setattr(self, f'scaler_{self.domain}_{i}', StandardScaler())
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.__read_data__()
        
        # ###TODO: 저장해서 쓸 수 있어야 하는데 StandardScaler에서 문제 발생한다.
        # # save preprocessed data
        # processed_data = data_path.split('.')[0] + f'_{flag}.pkl'
        # save_path = os.path.join(root_path,
        #                          'preprocessed',
        #                          f'{self.seq_len}_{self.label_len}_{self.pred_len}')
        # os.makedirs(save_path, exist_ok=True)
        # self.pkl_file = os.path.join(save_path, processed_data)
        # if os.path.exists(self.pkl_file):
        #     print(f'Load saved file {self.pkl_file}.')
            
        #     with open(self.pkl_file, 'rb') as f:
        #         save_data = pickle.load(f)
                
        #         self.data_x = save_data['data_x']
        #         self.data_y = save_data['data_y']
        #         self.data_stamp = save_data['data_stamp']
                
        # else:
        #     print(f'Preprocessing {data_path} for {flag}...')
        
        #     self.__read_data__()
            
        #     # load preprocessed data
        #     save_data = {
        #         'data_x': self.data_x,
        #         'data_y': self.data_y,
        #         'data_stamp': self.data_stamp
        #     }
            
        #     with open(self.pkl_file, 'wb') as f:
        #         pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    def __read_data__(self):            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        ## apply resolution
        df_raw['minute'] = df_raw['timestamp'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
        df_raw = df_raw.drop(df_raw[df_raw['minute'] % 60 != 0].index)
        df_raw = df_raw.reset_index()
        
        # Using specific columns
        df_raw['date'] = pd.to_datetime(df_raw['timestamp'], errors='raise')
        '''
        df_raw.columns: ['date', ...(other features)]
        '''
        df_raw = df_raw[['date'] + self.input_channels]

        ## pre-processing
        df_date = pd.DataFrame()
        df_date['date'] = pd.to_datetime(df_raw.date)
        df_date['year'] = df_date.date.apply(lambda row: row.year, 1)
        df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
        df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
        df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
        df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)        
        print(f"Preprocess for {self.data_path} about missing and wrong values.")
        print(f"Missing values: ")

        for i in df_raw.columns[1:]:    # except 'date' which is the first column
            print(i, end='\t')
            print(df_raw[i].isnull().sum(), end='  ')
            df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
            df_raw, df_date = self.remove_null_value(df_raw, df_date, i)
            print('-> ', df_raw[i].isnull().sum())
        print('')
        
        ## check if there is missing value
        assert df_raw['Global_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Diffuse_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Weather_Temperature_Celsius'].isnull().sum() == 0
        assert df_raw['Weather_Relative_Humidity'].isnull().sum() == 0
        assert df_raw['Active_Power'].isnull().sum() == 0
        
        ### get maximum and minimum value of 'Active_Power'
        setattr(self, f'pv_{self.domain}_max', np.max(df_raw['Active_Power'].values))   # save the maximum value of 'Active_Power' as a source/target domain
        setattr(self, f'pv_{self.domain}_min', np.min(df_raw['Active_Power'].values))   # save the minimum value of 'Active_Power' as a source/target domain

        # remove minus temperature
        print('Remove minus temperature.')
        print('Weather_Temperature_Celsius', end='\t')
        print(len(df_raw[df_raw['Weather_Temperature_Celsius'] < 0].index.tolist()), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_minus_temperature(df_raw, df_date)
        print('-> ', len(df_raw[df_raw['Weather_Temperature_Celsius'] < 0].index.tolist()))
        print('')

        # remove minus radiation
        print('Remove minus radiation.')
        for i in ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation']:
            print(i, end='\t')
            print(len(df_raw[df_raw[i] < 0].index.tolist()), end='  ')
            df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
            df_raw, df_date = self.remove_minus_radiation(df_raw, df_date)
            print('-> ', len(df_raw[df_raw[i] < 0].index.tolist()))
        print('')

        ### clip over 100 humidity
        print('Clip over 100 humidity.')
        print(f'Weather_Relative_Humidity', end='\t')
        print(len(df_raw[df_raw['Weather_Relative_Humidity'] > 100].index.tolist()), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.clip_over_hundred_humidity(df_raw, df_date)
        print('-> ', len(df_raw[df_raw['Weather_Relative_Humidity'] > 100].index.tolist()))
        print('')

        # crop data for the year to be used train/val/test
        border1 = df_raw[df_raw['date'] >= f'{self.DATASET_SPLIT_YEAR[self.data_path][2*(self.set_type)]}-01-01 00:00:00'].index[0]
        border2 = df_raw[df_raw['date'] <= f'{self.DATASET_SPLIT_YEAR[self.data_path][2*(self.set_type)+1]}-12-31 23:00:00'].index[-1]+1
        df_stamp = df_raw[['date']][border1:border2]

        if self.timeenc == 0:
            data_stamp = pd.DataFrame()
            data_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            data_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            data_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            data_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_border1 = df_raw[df_raw['date'] >= f'{self.DATASET_SPLIT_YEAR[self.data_path][0]}-01-01 00:00:00'].index.tolist()[0]
            train_border2 = df_raw[df_raw['date'] <= f'{self.DATASET_SPLIT_YEAR[self.data_path][1]}-12-31 23:00:00'].index.tolist()[-1]+1
            train_data = df_data[train_border1:train_border2]
            
            train_data_values = train_data.values
            df_data_values = df_data.values
            transformed_data = []  # List to store each transformed feature
            
            # Transform data
            for idx, val in enumerate(df_raw.columns[1:]):  # except 'date' which is the first column
                train_features = train_data_values[:, idx].reshape(-1, 1)
                getattr(self, f'scaler_{self.domain}_{val}').fit(train_features)
                
                df_data_features = df_data_values[:, idx].reshape(-1, 1)
                transformed_feature = getattr(self, f'scaler_{self.domain}_{val}').transform(df_data_features)
                transformed_data.append(transformed_feature)
            data = np.concatenate(transformed_data, axis=1)
            
        else:
            data = df_data.values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.timeenc == 0:
            # setattr(self, f'data_stamp_{self.domain}', data_stamp[['month', 'day', 'weekday', 'hour']].values)
            self.data_stamp = data_stamp[['month', 'day', 'weekday', 'hour']].values
        elif self.timeenc == 1:
            # setattr(self, f'data_stamp_{self.domain}', data_stamp)
            self.data_stamp = data_stamp
        
    def remove_successive_missing_value(self, pv, df_stamp):
        df_stamp_org = copy.deepcopy(df_stamp)
        missing_idx = pv[pv['Active_Power'].isnull()]['Active_Power'].index.tolist()
        successive, count = True, 1
        front, rear = 0, 0

        for i in range(len(missing_idx)-1, 0, -1):
            front, rear = missing_idx[i-1], missing_idx[i]
            # print(successive, count)
            if successive:
                if front == rear-1:
                    count += 1
                else:
                    if count >= 4:
                        year, month, day = df_stamp_org.iloc[rear]['year'], df_stamp_org.iloc[rear]['month'], df_stamp_org.iloc[rear]['day']
                        # print('[', rear, ']', year, month, day)
                        has_index = len(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                        if has_index:
                            pv = pv.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                            df_stamp = df_stamp.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                    successive = False
                    count = 1
            else:
                if front == rear-1:
                    successive = True
                    count += 1
                else:
                    pass
        
        if successive and count >= 4:
            year, month, day = df_stamp_org.iloc[rear]['year'], df_stamp_org.iloc[rear]['month'], df_stamp_org.iloc[rear]['day']
            has_index = len(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
            if has_index:
                pv = pv.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                df_stamp = df_stamp.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)

        return pv, df_stamp
        
    def remove_null_value(self, pv, df_stamp, column):   # pv = df_raw
        df_stamp_org = copy.deepcopy(df_stamp)
        missing_idx = pv[pv[column].isnull()][column].index.tolist()
        for i in range(len(missing_idx)-1, -1, -1):
            idx = missing_idx[i]
            year, month, day = df_stamp_org.iloc[idx]['year'], df_stamp_org.iloc[idx]['month'], df_stamp_org.iloc[idx]['day']
            has_index = len(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
            if has_index:
                # print('year, month, day: ', year, month, day)
                pv = pv.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                df_stamp = df_stamp.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)

        return pv, df_stamp
    
    def remove_minus_temperature(self, pv, df_stamp):   # pv = df_raw
        df_stamp_org = copy.deepcopy(df_stamp)
        minus_idx = pv[pv['Weather_Temperature_Celsius'] < 0].index.tolist()
        for i in range(len(minus_idx)-1, 1, -1):
            idx = minus_idx[i]
            year, month, day = df_stamp_org.iloc[idx]['year'], df_stamp_org.iloc[idx]['month'], df_stamp_org.iloc[idx]['day']
            has_index = len(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
            if has_index:
                # print('year, month, day: ', year, month, day)
                pv = pv.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                df_stamp = df_stamp.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)

        return pv, df_stamp

    def remove_minus_radiation(self, pv, df_stamp):
        df_stamp_org = copy.deepcopy(df_stamp)
        minus_idx = pv[pv['Global_Horizontal_Radiation'] < 0].index.tolist()
        minus_idx.extend(pv[pv['Diffuse_Horizontal_Radiation'] < 0].index.tolist())
        for i in range(len(minus_idx)-1, -1, -1):
            idx = minus_idx[i]
            year, month, day = df_stamp_org.iloc[idx]['year'], df_stamp_org.iloc[idx]['month'], df_stamp_org.iloc[idx]['day']
            has_index = len(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
            if has_index:
                # print('year, month, day: ', year, month, day)
                pv = pv.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
                df_stamp = df_stamp.drop(df_stamp[(df_stamp['year'] == year) & (df_stamp['month'] == month) & (df_stamp['day'] == day)].index)
            
        return pv, df_stamp

    def clip_over_hundred_humidity(self, pv, df_stamp):
        over_idx = pv[pv['Weather_Relative_Humidity'] > 100].index.tolist()
        pv.loc[over_idx, 'Weather_Relative_Humidity'] = 100
        
        return pv, df_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return (seq_x, seq_y, seq_x_mark, seq_y_mark)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        ''' active power만 예측하고 검증할 때는 이거만 씀 '''
        return getattr(self, f'scaler_{self.domain}_Active_Power').inverse_transform(data)

class Dataset_DKASC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='',
                 target='Active_Power', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path_list = data_path.split(',')
        
        self.DATASET_SPLIT_YEAR = {
            '52-Site_DKA-M16_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 33
            '54-Site_DKA-M15_C-Phase.csv'    : [2011, 2012,  2013, 2014,  2015, 2016],
            '55-Site_DKA-M20_B-Phase.csv'    : [2014, 2020,  2021, 2021,  2022, 2023],    # 29
            '56-Site_DKA-M20_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 30
            '57-Site_DKA-M16_A-Phase.csv'    : [2011, 2012,  2013, 2014,  2015, 2016],
            '58-Site_DKA-M17_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 36
            '59-Site_DKA-M19_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 38
            '60-Site_DKA-M18_A-Phase.csv'    : [2013, 2019,  2020, 2021,  2022, 2023],
            '61-Site_DKA-M15_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 23
            '63-Site_DKA-M17_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 34
            '64-Site_DKA-M17_B-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 37
            '66-Site_DKA-M16_B-Phase.csv'    : [2011, 2020,  2021, 2022,  2022, 2023],
            '67-Site_DKA-M8_A-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 21
            '68-Site_DKA-M8_C-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 20
            '69-Site_DKA-M4_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 17
            '70-Site_DKA-M5_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 3
            '71-Site_DKA-M2_C-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 18
            '72-Site_DKA-M15_B-Phase.csv'    : [2014, 2020,  2021, 2021,  2022, 2023],    # 26
            '73-Site_DKA-M19_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 35
            '74-Site_DKA-M18_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 31
            '77-Site_DKA-M18_B-Phase.csv'    : [2011, 2012,  2013, 2014,  2015, 2016],
            '79-Site_DKA-M6_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 7
            '84-Site_DKA-M5_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 12
            '85-Site_DKA-M7_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 10
            '90-Site_DKA-M3_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 14
            '92-Site_DKA-M6_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 13
            '93-Site_DKA-M4_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 8
            '97-Site_DKA-M10_B+C-Phases.csv' : [2011, 2020,  2021, 2021,  2022, 2023],    # 22
            '98-Site_DKA-M8_B-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 19
            '99-Site_DKA-M4_C-Phase.csv'     : [2009, 2014,  2015, 2016,  2017, 2018],
            '100-Site_DKA-M1_A-Phase.csv'    : [2009, 2020,  2021, 2021,  2022, 2023],    # 16A
            '212-Site_DKA-M15_C-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 25
            '213-Site_DKA-M16_A-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 24
            '214-Site_DKA-M18_B-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 32
            '218-Site_DKA-M4_C-Phase_II.csv' : [2018, 2020,  2021, 2021,  2022, 2023],    # 9A
        }


        self.pv_list = []
        self.x_list = []
        self.y_list = []
        self.ds_list = []
        # self.ap_max_list = []
        # self.ap_min_list = []

        # 각 모드에 맞는 파일 이름 (빠른 훈련을 위해서 데이터 미리 파일로 저장해서 불러옴)
        self.train_file = os.path.join(root_path, f'DKASC_preprocessed_train.{self.save_format}')
        self.val_file = os.path.join(root_path, f'DKASC_preprocessed_val.{self.save_format}')
        self.test_file = os.path.join(root_path, f'DKASC_preprocessed_test.{self.save_format}')
        
        # 파일이 존재하면 불러오고, 없으면 생성
        if os.path.exists(self._get_preprocessed_file()): 
            self.__load_preprocessed_data__() 
        else:
            self.__preprocess_and_save_data__()
            self.__load_preprocessed_data__()
        

        self.__read_data__()


    # 1. 데이터셋 리스트 불러와서 훈련, 검증, 테스트 데이터셋 나누기
    # 2. 전체 훈련 데이터셋들을 불러와서 Scaler Fit
    # 3. 검증, 테스트 데이터셋 scaling
    # 4. 나중에 모델 훈련 시, 불러올 수 있게 전체 훈련, 검증, 테스트 데이터셋 저장
    def __preprocess_and_save_data__(self):
        # 데이터 프레임
        train_data_frames, val_data_frames, test_data_frames = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Data stamp 저장 프레임 (년, 월, 일, 요일, 시간)
        train_ds_frames, val_ds_frames, test_ds_frames = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    
        # DKASC 모든 데이터 불러오기
        if self.data_paths == 'ALL':     
            self.data_paths = self.DATASET_SPLIT_YEAR.keys()   

        self.data_path_list = sorted(self.data_paths)
        for idx, data_path in enumerate(self.data_path_list):
            df_raw = pd.read_csv(os.path.join(self.root_path, data_path))
            
            df_raw['date'] = pd.to_datetime(df_raw['timestamp'], errors='raise')

            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            cols = df_raw.columns.tolist()
            cols.remove('timestamp')
            cols.remove('date')
            cols.remove('Active_Power')
            df_raw = df_raw[['date'] + cols + [self.target]]            

            ## Creae scaler for each feature
            for i in range(len(df_raw.columns)-1):
                setattr(self, f'scaler_{i}', StandardScaler())
            
            ## pre-processing
            df_date = pd.DataFrame()
            df_date['date'] = pd.to_datetime(df_raw.date)
            df_date['year'] = df_date.date.apply(lambda row: row.year, 1)
            df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
            df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
            df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
            df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
            
            ## check for not null
            assert (df_raw.isnull().sum()).sum() == 0
            
            ### get maximum and minimum value of 'Active_Power'
            # self.ap_max_list.append(df_raw['Active_Power'].max())
            # self.ap_min_list.append(df_raw['Active_Power'].min())
            print(data_path, '\tmax: ', round(df_raw['Active_Power'].max(),2), '\tmin: ', round(df_raw['Active_Power'].min(),2))
            for column in df_raw.columns:
                if column == 'date': continue
                print(column, '\tmean:', round(df_raw[column].mean(),2),'\tstd:', round(df_raw[column].std(),2))

            # 1-1. 년도에 따라서 train, val, test set 나누기
            # 1-1-1. 파일명에 해당하는 연도 정보 가져오기
            if data_path not in self.DATASET_SPLIT_YEAR:
                raise ValueError(f"'{data_path}' 파일에 대한 연도 정보가 없습니다.")
            
            split_years = self.DATASET_SPLIT_YEAR[data_path]

            # split_years는 [train_start, train_end, val_start, val_end, test_start, test_end]
            train_start = split_years[0]
            train_end = split_years[1]
            val_start = split_years[2]
            val_end = split_years[3]
            test_start = split_years[4]
            test_end = split_years[5]

            # 각 구간에 해당하는 데이터를 추출
            train_data = df_raw[(df_raw['date'] >= f'{train_start}-01-01') & (df_raw['date'] <= f'{train_end}-12-31')]
            val_data = df_raw[(df_raw['date'] >= f'{val_start}-01-01') & (df_raw['date'] <= f'{val_end}-12-31')]
            test_data = df_raw[(df_raw['date'] >= f'{test_start}-01-01') & (df_raw['date'] <= f'{test_end}-12-31')]


            # 각 데이터셋 time encoding (년, 월, 일, 요일, 시간)
            for df_stamp in [train_data, val_data, test_data]:
                if self.timeenc == 0:
                    data_stamp = pd.DataFrame()
                    data_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                    data_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                    data_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                    data_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                    
                    data_stamp = data_stamp[['month', 'day', 'weekday', 'hour']].values

                elif self.timeenc == 1:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)
            
            train_data_frames = pd.concat([train_data_frames, train_data])
            val_data_frames = pd.concat([val_data_frames, val_data])
            test_data_frames = pd.concat([test_data_frames, test_data])

            train_ds_frames = pd.concat([train_ds_frames, pd.DataFrame(data_stamp)])
            val_ds_frames = pd.concat([val_ds_frames, pd.DataFrame(data_stamp)])
            test_ds_frames = pd.concat([test_ds_frames, pd.DataFrame(data_stamp)])


        self.scaler = StandardScaler()  
        data_frames = [train_data_frames, val_data_frames, test_data_frames]

        for i, df_raw in enumerate(data_frames):
            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:] 
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]


            # 훈련 데이터(전체 Location)에 대해서만 스케일러 Fit
            if self.scale:
                df_data_values = df_data.values

               # Train 데이터에 대해 fit 후 transform
                if i == 0:  # train 데이터에 대해서 fit과 transform을 모두 수행
                    self.scaler.fit(df_data_values)  # fit
                    df_data_values = self.scaler.transform(df_data_values)  # transform
                else:  # val, test는 transform만 수행
                    df_data_values = self.scaler.transform(df_data_values)

                
                df_data = pd.DataFrame(df_data_values, columns=df_data.columns)
            
            # 변경된 df_data를 원래의 df_raw에 업데이트
            df_raw[cols_data if self.features in ['M', 'MS'] else [self.target]] = df_data


            # 리스트 내 해당 데이터프레임을 변경된 값으로 업데이트
            data_frames[i] = df_raw  # 직접 대입하여 변경


        # 훈련, 검증, 테스트 데이터 저장
        train_data_frames.to_pickle(self.train_file)
        print("[INFO] Train data saved.")

        val_data_frames.to_pickle(self.val_file)
        print("[INFO] Validation data saved.")

        test_data_frames.to_pickle(self.test_file)
        print("[INFO] Test data saved.")

            ###### TODO: 의논 후 정리 [이전] 각각의 column에 대해서 스케일러 적용 ######################
            
                # train_data_values = train_data_frames.values
                # df_data_values = df_data.values
            #     transformed_data = []  # List to store each transformed feature
            #     for i in range(df_data_values.shape[1]):
            #         train_features = train_data_values[:, i].reshape(-1, 1)
            #         getattr(self, f'scaler_{i}').fit(train_features)
                    
            #         df_data_features = df_data_values[:, i].reshape(-1, 1)
            #         transformed_feature = getattr(self, f'scaler_{i}').transform(df_data_features)
            #         transformed_data.append(transformed_feature)
            #     data = np.concatenate(transformed_data, axis=1)
                
            # else:
            #     data = df_data.values

            ################################################################

  
    def __getitem__(self, index):
        s_begin = index
        for i, x in enumerate(self.x_list):
            if s_begin < (len(x) - self.seq_len - self.pred_len +1):
                break
            else:
                s_begin -= (len(x) - self.seq_len - self.pred_len +1)
            
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.x_list[i][s_begin:s_end]
        seq_y = self.y_list[i][r_begin:r_end]
        seq_x_mark = self.ds_list[i][s_begin:s_end]
        seq_y_mark = self.ds_list[i][r_begin:r_end]
        # pv_max = self.ap_max_list[i]
        # pv_min = self.ap_min_list[i]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
        return seq_x, seq_y, seq_x_mark, seq_y_mark, pv_max, pv_min

    def __len__(self):
        total_len = 0
        for x in self.x_list:
            total_len += (len(x) - self.seq_len - self.pred_len +1)
        return total_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        ## change the scaler number .. if num of features changes
        return self.scaler_8.inverse_transform(data)

class Dataset_GIST(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='GIST_sisuldong.csv', target='Active_Power',
                 scale=True, timeenc=0, freq='h', domain='target'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.domain = domain

        self.root_path = root_path
        self.data_path = data_path
        
        # Create scaler for each input_channels
        # self.input_channels = ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        self.input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        self.input_channels = self.input_channels + [self.target]
        for i in self.input_channels:
            setattr(self, f'scaler_{self.domain}_{i}', StandardScaler())
            
        self.__read_data__()

        # ###TODO: 저장해서 쓸 수 있어야 하는데 StandardScaler에서 문제 발생한다.
        # # save preprocessed data
        # processed_data = data_path.split('.')[0] + f'_{flag}.pkl'
        # save_path = os.path.join(root_path,
        #                          'preprocessed',
        #                          f'{self.seq_len}_{self.label_len}_{self.pred_len}')
        # os.makedirs(save_path, exist_ok=True)
        # self.pkl_file = os.path.join(save_path, processed_data)
        # if os.path.exists(self.pkl_file):
        #     print(f'Load saved file {self.pkl_file}.')
            
        #     with open(self.pkl_file, 'rb') as f:
        #         save_data = pickle.load(f)
                
        #         self.data_x = save_data['data_x']
        #         self.data_y = save_data['data_y']
        #         self.data_stamp = save_data['data_stamp']                  
        # else:
        #     print(f'Preprocessing {data_path} for {flag}...')

        #     self.__read_data__()
            
        #     # load preprocessed data
        #     save_data = {
        #         'data_x': self.data_x,
        #         'data_y': self.data_y,
        #         'data_stamp': self.data_stamp
        #     }
            
        #     with open(self.pkl_file, 'wb') as f:
        #         pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
        

    def __read_data__(self):            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        df_raw['timestep'] = pd.to_datetime(df_raw['timestep'], errors='raise')

        '''
        df_raw.columns: ['timestep', ...(other features), target feature]
        '''

        columns = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        df_raw = df_raw[['timestep'] + self.input_channels]
        df_raw[columns] = df_raw[columns].apply(pd.to_numeric, errors='coerce')

        # preprocessing
        df_stamp = pd.DataFrame()
        df_stamp['timestep'] = pd.to_datetime(df_raw['timestep'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['timestep'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['timestep'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        ## check if there is missing value
        assert df_raw['Global_Horizontal_Radiation'].isnull().sum() == 0
        # assert df_raw['Diffuse_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Weather_Temperature_Celsius'].isnull().sum() == 0
        assert df_raw['Weather_Relative_Humidity'].isnull().sum() == 0
        assert df_raw['Active_Power'].isnull().sum() == 0
        
        ### get maximum and minimum value of 'Active_Power'
        setattr(self, f'pv_{self.domain}_max', np.max(df_raw['Active_Power'].values))   # save the maximum value of 'Active_Power' as a source/target domain
        setattr(self, f'pv_{self.domain}_min', np.min(df_raw['Active_Power'].values))   # save the minimum value of 'Active_Power' as a source/target domain
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_border1 = border1s[0]
            train_border2 = border2s[0]
            train_data = df_data[train_border1:train_border2]
            
            train_data_values = train_data.values
            df_data_values = df_data.values
            transformed_data = []  # List to store each transformed feature
            # for i in range(df_data_values.shape[1]):
            for idx, val in enumerate(df_raw.columns[1:]):  # except 'date' which is the first column
                train_features = train_data_values[:, idx].reshape(-1, 1)
                getattr(self, f'scaler_{self.domain}_{val}').fit(train_features)

                df_data_features = df_data_values[:, idx].reshape(-1, 1)
                transformed_feature = getattr(self, f'scaler_{self.domain}_{val}').transform(df_data_features)
                transformed_data.append(transformed_feature)
            data = np.concatenate(transformed_data, axis=1)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.timeenc == 0:
            self.data_stamp = df_stamp[border1:border2][['month', 'day', 'weekday', 'hour']].values
        elif self.timeenc == 1:
            self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        ''' active power만 예측하고 검증할 때는 이거만 씀 '''
        return getattr(self, f'scaler_{self.domain}_Active_Power').inverse_transform(data)

####################################################

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)