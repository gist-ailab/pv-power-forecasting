import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

import copy

warnings.filterwarnings('ignore')



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

####################################################
class Dataset_pv_DKASC(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='91-Site_DKA-M9_B-Phase.csv', target='Active_Power',
                 scale=True, timeenc=0, freq='h', domain='source'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            # setattr(self, f'seq_len_{domain}', 24 * 4 * 4)
            # setattr(self, f'label_len_{domain}', 24 * 4)
            # setattr(self, f'pred_len_{domain}', 24 * 4)
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            # setattr(self, f'seq_len_{domain}', size[0])
            # setattr(self, f'label_len_{domain}', size[1])
            # setattr(self, f'pred_len_{domain}', size[2])
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = 'Active_Power'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.domain = domain

        self.root_path = root_path
        self.data_path = data_path

        self.DATASET_SPLIT_YEAR = {
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
        self.__read_data__()

    def __read_data__(self):
        # Creae scaler for each feature
        for i in range(5):
            setattr(self, f'scaler_{self.domain}_{i}', StandardScaler())
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        ## apply resolution
        df_raw['minute'] = df_raw['timestamp'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
        df_raw = df_raw.drop(df_raw[df_raw['minute'] % 60 != 0].index)
        df_raw = df_raw.reset_index()
        
        df_raw['date'] = pd.to_datetime(df_raw['timestamp'], errors='raise')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        df_raw = df_raw[['date'] + cols + [self.target]]

        ## pre-processing
        df_date = pd.DataFrame()
        df_date['date'] = pd.to_datetime(df_raw.date)
        df_date['year'] = df_date.date.apply(lambda row: row.year, 1)
        df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
        df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
        df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
        df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
        
        ### remove missing value of 'Active_Power'
        print(df_raw['Active_Power'].isnull().sum(), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_null_value(df_raw, df_date, 'Active_Power')
        print('-> ', df_raw['Active_Power'].isnull().sum())
        ### remove missing value of 'Weather_Temperature_Celsius'
        print(df_raw['Weather_Temperature_Celsius'].isnull().sum(), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_null_value(df_raw, df_date, 'Weather_Temperature_Celsius')
        print('-> ', df_raw['Weather_Temperature_Celsius'].isnull().sum())
        ### remove missing value of 'Global_Horizontal_Radiation'
        print(df_raw['Global_Horizontal_Radiation'].isnull().sum(), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_null_value(df_raw, df_date, 'Global_Horizontal_Radiation')
        print('-> ', df_raw['Global_Horizontal_Radiation'].isnull().sum())
        ### remove missing value of 'Diffuse_Horizontal_Radiation'
        print(df_raw['Diffuse_Horizontal_Radiation'].isnull().sum(), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_null_value(df_raw, df_date, 'Diffuse_Horizontal_Radiation')
        print('-> ', df_raw['Diffuse_Horizontal_Radiation'].isnull().sum())
        ### remove missing value of 'Weather_Relative_Humidity'
        print(df_raw['Weather_Relative_Humidity'].isnull().sum(), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_null_value(df_raw, df_date, 'Weather_Relative_Humidity')
        print('-> ', df_raw['Weather_Relative_Humidity'].isnull().sum())
        
        assert df_raw['Active_Power'].isnull().sum() == 0
        assert df_raw['Weather_Temperature_Celsius'].isnull().sum() == 0
        assert df_raw['Global_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Diffuse_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Weather_Relative_Humidity'].isnull().sum() == 0
        
        ### get maximum and minimum value of 'Active_Power'
        setattr(self, f'pv_{self.domain}_max', np.max(df_raw['Active_Power'].values))   # save the maximum value of 'Active_Power' as a source/target domain
        setattr(self, f'pv_{self.domain}_min', np.min(df_raw['Active_Power'].values))   # save the minimum value of 'Active_Power' as a source/target domain
        # self.pv_max = np.max(df_raw['Active_Power'].values)
        # self.pv_min = np.min(df_raw['Active_Power'].values)

        ### remove minus temperature
        print(len(df_raw[df_raw['Weather_Temperature_Celsius'] < 0].index.tolist()), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_minus_temperature(df_raw, df_date)
        print('-> ', len(df_raw[df_raw['Weather_Temperature_Celsius'] < 0].index.tolist()))

        ### remove minus radiation
        print(len(df_raw[df_raw['Global_Horizontal_Radiation'] < 0].index.tolist()), end='  ')
        print(len(df_raw[df_raw['Diffuse_Horizontal_Radiation'] < 0].index.tolist()))
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.remove_minus_radiation(df_raw, df_date)
        print('-> ', len(df_raw[df_raw['Global_Horizontal_Radiation'] < 0].index.tolist()))
        print('-> ', len(df_raw[df_raw['Diffuse_Horizontal_Radiation'] < 0].index.tolist()))

        ### clip over 100 humidity
        print(len(df_raw[df_raw['Weather_Relative_Humidity'] > 100].index.tolist()), end='  ')
        df_raw, df_date = df_raw.reset_index(drop=True), df_date.reset_index(drop=True)
        df_raw, df_date = self.clip_over_hundred_humidity(df_raw, df_date)
        print('-> ', len(df_raw[df_raw['Weather_Relative_Humidity'] > 100].index.tolist()))

        # del df_date
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
            for i in range(5):
                train_features = train_data_values[:, i].reshape(-1, 1)
                getattr(self, f'scaler_{self.domain}_{i}').fit(train_features)
                
                df_data_features = df_data_values[:, i].reshape(-1, 1)
                transformed_feature = getattr(self, f'scaler_{self.domain}_{i}').transform(df_data_features)
                transformed_data.append(transformed_feature)
            data = np.concatenate(transformed_data, axis=1)
            
        else:
            data = df_data.values
        
        # setattr(self, f'data_x_{self.domain}', data[border1:border2])
        # setattr(self, f'data_y_{self.domain}', data[border1:border2])
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
        # self.seq_len = getattr(self, f'seq_len_{self.domain}')
        # self.label_len = getattr(self, f'label_len_{self.domain}')
        # self.pred_len = getattr(self, f'pred_len_{self.domain}')
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # seq_x = getattr(self, f'data_x_{self.domain}')[s_begin:s_end]
        # seq_y = getattr(self, f'data_y_{self.domain}')[r_begin:r_end]
        # seq_x_mark = getattr(self, f'data_stamp_{self.domain}')[s_begin:s_end]
        # seq_y_mark = getattr(self, f'data_stamp_{self.domain}')[r_begin:r_end]        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(getattr(self, f'data_x_{self.domain}')) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return getattr(self, f'scaler_{self.domain}_4').inverse_transform(data)
        # return self.scaler_DKASC_4.inverse_transform(data)
    
    
class Dataset_pv_DKASC_multi(Dataset):
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
        self.target = 'Active_Power'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path_list = data_path.split(',')
        
        self.DATASET_SPLIT_YEAR = {
            '1A-91-Site_DKA-M9_B-Phase.csv'     : [2014, 2020,  2021, 2021,  2022, 2023],    # 
            '1B-87-Site_DKA-M9_A+C-Phases.csv'  : [2014, 2020,  2021, 2021,  2022, 2023],    # 
            '2-78-Site_DKA-M11_3-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '3-70-Site_DKA-M5_A-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '4-86-Site_DKA-M2_B-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '5-89-Site_DKA-M3_B-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '6-95-Site_DKA-M3_C-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '7-79-Site_DKA-M6_A-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '8-93-Site_DKA-M4_A-Phase.csv'      : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '9A-218-Site_DKA-M4_C-Phase_II.csv' : [2018, 2020,  2021, 2021,  2022, 2023],    # 
            '10-85-Site_DKA-M7_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '11-106-Site_DKA-M5_C-Phase.csv'    : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '12-84-Site_DKA-M5_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '13-92-Site_DKA-M6_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '14-90-Site_DKA-M3_A-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '16A-100-Site_DKA-M1_A-Phase.csv'   : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '16B-103-Site_DKA-M1_B-Phase.csv'   : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '16C-105-Site_DKA-M1_C-Phase.csv'   : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '16D-81-Site_DKA-M2_A-Phase.csv'    : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '17-69-Site_DKA-M4_B-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '18-71-Site_DKA-M2_C-Phase.csv'     : [2009, 2020,  2021, 2021,  2022, 2023],    # 
            '19-98-Site_DKA-M8_B-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 
            '20-68-Site_DKA-M8_C-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 
            '21-67-Site_DKA-M8_A-Phase.csv'     : [2010, 2020,  2021, 2021,  2022, 2023],    # 
            '22-97-Site_DKA-M10_B+C-Phases.csv' : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '23-61-Site_DKA-M15_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '24-213-Site_DKA-M16_A-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 
            '25-212-Site_DKA-M15_C-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 
            '26-72-Site_DKA-M15_B-Phase.csv'    : [2014, 2020,  2021, 2021,  2022, 2023],    # 
            '29-55-Site_DKA-M20_B-Phase.csv'    : [2014, 2020,  2021, 2021,  2022, 2023],    # 
            '30-56-Site_DKA-M20_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '31-74-Site_DKA-M18_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '32-214-Site_DKA-M18_B-Phase_II.csv': [2017, 2020,  2021, 2021,  2022, 2023],    # 
            '33-52-Site_DKA-M16_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '34-63-Site_DKA-M17_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '35-73-Site_DKA-M19_A-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '36-58-Site_DKA-M17_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '37-64-Site_DKA-M17_B-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
            '38-59-Site_DKA-M19_C-Phase.csv'    : [2011, 2020,  2021, 2021,  2022, 2023],    # 
        }
        self.pv_list = []
        self.x_list = []
        self.y_list = []
        self.ds_list = []
        # self.ap_max_list = []
        # self.ap_min_list = []
        
        self.__read_data__()

    def __read_data__(self):
        # data_path_list = ['25-212-Site_DKA-M15_C-Phase_II.csv', '10-85-Site_DKA-M7_A-Phase.csv']
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

            border1 = df_raw[df_raw['date'] >= f'{self.DATASET_SPLIT_YEAR[data_path][2*(self.set_type)]}-01-01 00:00:00'].index[0]
            border2 = df_raw[df_raw['date'] <= f'{self.DATASET_SPLIT_YEAR[data_path][2*(self.set_type)+1]}-12-31 23:00:00'].index[-1]+1
            
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
                train_border1 = df_raw[df_raw['date'] >= f'{self.DATASET_SPLIT_YEAR[data_path][0]}-01-01 00:00:00'].index.tolist()[0]
                train_border2 = df_raw[df_raw['date'] <= f'{self.DATASET_SPLIT_YEAR[data_path][1]}-12-31 23:00:00'].index.tolist()[-1]+1
                train_data = df_data[train_border1:train_border2]
                
                train_data_values = train_data.values
                df_data_values = df_data.values
                transformed_data = []  # List to store each transformed feature
                for i in range(df_data_values.shape[1]):
                    train_features = train_data_values[:, i].reshape(-1, 1)
                    getattr(self, f'scaler_{i}').fit(train_features)
                    
                    df_data_features = df_data_values[:, i].reshape(-1, 1)
                    transformed_feature = getattr(self, f'scaler_{i}').transform(df_data_features)
                    transformed_data.append(transformed_feature)
                data = np.concatenate(transformed_data, axis=1)
                
            else:
                data = df_data.values

            self.x_list.append(data[border1:border2])
            self.y_list.append(data[border1:border2])
            if self.timeenc == 0:
                self.ds_list.append(data_stamp[['month', 'day', 'weekday', 'hour']].values)
            else:
                self.ds_list.append(data_stamp)
        
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


class CrossDomain_Dataset(Dataset):
    # Dataloader for cross-validation
    def __init__(self, source_root_path, target_root_path,
                 flag='train', size=None, features='S',
                 source_data_path='91-Site_DKA-M9_B-Phase.csv',
                 target_data_path='GIST_sisuldong.csv',
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
        
        self.source_root_path = source_root_path
        self.source_data_path = source_data_path
        self.target_root_path = target_root_path
        self.target_data_path = target_data_path
        
        # TODO: We have to designate the source/target domain for each dataset with Cross_Dataset args, not the way below, like typing directly.
        pv_DKASC = Dataset_pv_DKASC(root_path=source_root_path, flag=flag, size=size,
                                    features=features, data_path=source_data_path, target=target,
                                    scale=scale, timeenc=timeenc, freq=freq, domain='source')
        pv_DKASC.__read_data__()
        self.seq_len_source = pv_DKASC.seq_len
        
        pv_GIST = Dataset_pv_GIST(root_path=target_root_path, flag=flag, size=size,
                                  features=features, data_path=target_data_path, target=target,
                                  scale=scale, timeenc=timeenc, freq=freq, domain='target')
        pv_GIST.__read_data__()
        
        # TODO: source & target dataset에 대한 data loader instance를 만들어서 활용해보자.
        # 일단 지금은 __red_data__ 를 따로 만들어서 사용
        # 아니면 __read_source_data__ 에서만 self 변수명을 달리하고 target은 그대로 사용하여 getitem에서 source, target을 구분하여 사용하는 것도 고려
        # DKASC 쪽의 변수 이름을 수정하고 여기에서는 instance를 생성해서 불러오기만 하면 될 듯 ㅋㅋㅋ
        
    def __getitem__(self, index):
        # BSH 위에서 instance 만든 거로 self가 생성되려나...? instance 통해서 따로 만들어야 할 것 같다.
        
        
        s_begin = index        
        # source domain
        
        s_end_source = s_begin + self.seq_len_source
        r_begin_source = s_end_source - self.label_len_source
        r_end_source = r_begin_source + self.label_len_source + self.pred_len_target

        # target domain


        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_4.inverse_transform(data)
        
   
class Dataset_pv_SolarDB(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='',
                 target='power_ac', scale=True, timeenc=0, freq='h'):
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
        self.target = 'power_ac'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()

    def __read_data__(self):
        # data_path_list = ['25-212-Site_DKA-M15_C-Phase_II.csv', '10-85-Site_DKA-M7_A-Phase.csv']
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        df_raw['date'] = pd.to_datetime(df_raw['timestamp'], errors='raise')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = df_raw.columns.tolist()
        # cols.remove('timestamp')
        # cols.remove('date')
        # cols.remove('power_ac')
        cols = ['temp', 'humidity']
        # cols.remove('sun_irradiance')       ##
        df_raw = df_raw[['date'] + cols + [self.target]]            

        ## Creae scaler for each feature
        self.num_cols = len(df_raw.columns)-1
        for i in range(self.num_cols):
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
        print(self.data_path, '\tmax: ', round(df_raw['power_ac'].max(),2), '\tmin: ', round(df_raw['power_ac'].min(),2))
        
        for column in df_raw.columns:
            if column == 'date': continue
            print(column, '\tmean:', round(df_raw[column].mean(),2),'\tstd:', round(df_raw[column].std(),2))


        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_stamp = df_raw[['date']]

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
            train_border1 = border1s[0]
            train_border2 = border2s[0]
            train_data = df_data[train_border1:train_border2]
            
            train_data_values = train_data.values
            df_data_values = df_data.values
            transformed_data = []  # List to store each transformed feature
            for i in range(self.num_cols):
                train_features = train_data_values[:, i].reshape(-1, 1)
                getattr(self, f'scaler_{i}').fit(train_features)
                
                df_data_features = df_data_values[:, i].reshape(-1, 1)
                transformed_feature = getattr(self, f'scaler_{i}').transform(df_data_features)
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
        ## change the scaler number .. if num of features changes
        return getattr(self, f'scaler_{self.num_cols-1}').inverse_transform(data)


class Dataset_pv_GIST(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='GIST_sisuldong.csv', target='Active_Power',
                 scale=True, timeenc=0, freq='h', domain='target'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            # setattr(self, f'seq_len_{domain}', 24 * 4 * 4)
            # setattr(self, f'label_len_{domain}', 24 * 4)
            # setattr(self, f'pred_len_{domain}', 24 * 4)
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            # setattr(self, f'seq_len_{domain}', size[0])
            # setattr(self, f'label_len_{domain}', size[1])
            # setattr(self, f'pred_len_{domain}', size[2])
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = 'Active_Power'
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.domain = domain

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    def __read_data__(self):
        # Creae scaler for each feature
        for i in range(4):
            setattr(self, f'scaler_{self.domain}_{i}', StandardScaler())
            
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        df_raw['date'] = pd.to_datetime(df_raw['timestamp'], errors='raise')

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = ['Global_Horizontal_Radiation', 'Diffuse_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        cols = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
        df_raw = df_raw[['date'] + cols + [self.target]]

        # df_stamp = df_raw[['date']][border1:border2]
        df_stamp = pd.DataFrame()
        df_stamp['date'] = df_raw.date
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        ## check if there is missing value
        assert df_raw['Active_Power'].isnull().sum() == 0
        assert df_raw['Weather_Temperature_Celsius'].isnull().sum() == 0
        assert df_raw['Global_Horizontal_Radiation'].isnull().sum() == 0
        assert df_raw['Weather_Relative_Humidity'].isnull().sum() == 0
        
        ### get maximum and minimum value of 'Active_Power'
        setattr(self, f'pv_{self.domain}_max', np.max(df_raw['Active_Power'].values))   # save the maximum value of 'Active_Power' as a source/target domain
        setattr(self, f'pv_{self.domain}_min', np.min(df_raw['Active_Power'].values))   # save the minimum value of 'Active_Power' as a source/target domain
        self.pv_max = np.max(df_raw['Active_Power'].values)
        self.pv_min = np.min(df_raw['Active_Power'].values)
        
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
            train_border1 = border1s[0]
            train_border2 = border2s[0]
            train_data = df_data[train_border1:train_border2]
            
            train_data_values = train_data.values
            df_data_values = df_data.values
            transformed_data = []  # List to store each transformed feature
            for i in range(4):
                train_features = train_data_values[:, i].reshape(-1, 1)
                getattr(self, f'scaler_{self.domain}_{i}').fit(train_features)
                
                df_data_features = df_data_values[:, i].reshape(-1, 1)
                transformed_feature = getattr(self, f'scaler_{self.domain}_{i}').transform(df_data_features)
                transformed_data.append(transformed_feature)
            data = np.concatenate(transformed_data, axis=1)
        else:
            data = df_data.values

        # setattr(self, f'data_x_{self.domain}', data[border1:border2])
        # setattr(self, f'data_y_{self.domain}', data[border1:border2])
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.timeenc == 0:
            # setattr(self, f'data_stamp_{self.domain}', df_stamp[border1:border2][['month', 'day', 'weekday', 'hour']].values)
            self.data_stamp = df_stamp[border1:border2][['month', 'day', 'weekday', 'hour']].values
        elif self.timeenc == 1:
            # setattr(self, f'data_stamp_{self.domain}', data_stamp[border1:border2])
            self.data_stamp = data_stamp[border1:border2]

        # # 13. check if there is missing value
        # assert len(df_raw[df_raw['Active_Power'].isnull()]['Active_Power'].index.tolist()) == 0
        # assert len(df_raw[df_raw['Weather_Temperature_Celsius'] < 0].index.tolist()) == 0
        # assert len(df_raw[df_raw['Global_Horizontal_Radiation'] < 0].index.tolist()) == 0
        # assert len(df_raw[df_raw['Diffuse_Horizontal_Radiation'] < 0].index.tolist()) == 0
        # assert len(df_raw[df_raw['Weather_Relative_Humidity'] > 100].index.tolist()) == 0


    def __getitem__(self, index):
        # self.seq_len = getattr(self, f'seq_len_{self.domain}')
        # self.label_len = getattr(self, f'label_len_{self.domain}')
        # self.pred_len = getattr(self, f'pred_len_{self.domain}')
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # seq_x = getattr(self, f'data_x_{self.domain}')[s_begin:s_end]
        # seq_y = getattr(self, f'data_y_{self.domain}')[r_begin:r_end]
        # seq_x_mark = getattr(self, f'data_stamp_{self.domain}')[s_begin:s_end]
        # seq_y_mark = getattr(self, f'data_stamp_{self.domain}')[r_begin:r_end]  
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # return len(getattr(self, f'data_x_{self.domain}')) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return getattr(self, f'scaler_{self.domain}_3').inverse_transform(data)
        # return self.scaler_3.inverse_transform(data)
   
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
