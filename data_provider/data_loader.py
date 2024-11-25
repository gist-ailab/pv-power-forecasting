import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.timefeatures import time_features
import warnings
import copy
import pickle
import joblib
import random
from collections import defaultdict

warnings.filterwarnings('ignore')


class Dataset_PV(Dataset):
    def __init__(self, root_path, data, flag='train', features='S', data_path=None, target='Active_Power', scale=True, 
                 timeenc=0, freq='h', size=None):
        
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.root_path = root_path
        self.data = data # 데이터 선택
        self.flag = flag
        self.features = features
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.scalers = {}
        self.site_data = {}
        self.capacity_dict = {}  # 사이트별 최대 출력량 저장

        self.mapping_name = pd.read_csv('./data_provider/dataset_name_mappings.csv')

        self.split_configs = {
            # 사이트로 나누는 loc
            'DKASC_AliceSprings': {
                # 'train': [57, 61, 70, 92, 59, 212, 213, 218, 56, 66, 52, 90, 72, 77, 60, 74, 67, 73, 214, 58, 68, 54, 79, 84],
                # 'val': [64, 99, 71, 98, 93, 100, 97],
                # 'test': [63, 85, 55, 69]
                'train': [1, 4, 7, 9, 10, 13, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36],
                'val': [2, 5, 8, 11, 14],
                'test': [3, 6, 12, 20, 28, 32]
            },
            'DKASC_Yulara': {
                'train': [1, 4, 7],
                'val': [2, 5],
                'test': [3, 6]
            },
            'GIST': {
                'train': [1, 4, 7, 9, 10, 13, 15],
                'val': [2, 5, 8, 11, 14],
                'test': [3, 6, 12, 16]
            },
            'German': {
                'train': [1, 4, 7, 8, 9],
                'val': [2, 5],
                'test': [3, 6]
            },
            'Miryang': {
                'train': [1, 4, 7, 2, 6],
                'val': [3],
                'test': [5]
            },
            #### 날짜로 나누는 loc
            'UK' : {
                'train' : 0.7,
                'val' : 0.2,
                'test' : 0.1
            },
            'OEDI_California': { # 2017.12.05  2023.10.31
                'train' : [2017, 2021],
                'val' : [2021, 2022],
                'test' : [2022, 2024]

            },
            'OEDI_Georgia' : {  # 2018.03.29  2022.03.10
                'train' : [2018, 2020],
                'val' : [2020, 2021],
                'test' : [2021, 2023]
            },
        }
        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        # 데이터 선택해서 로드하기
        # csv로 이름 매핑
        # 이름에서 정보 추출하기

        # train, val, test 분할
        
        # 스케일러 적용
        # 나머지 프로세싱
        # getitem, len 정의

    def load_preprocessed_data(self):

        """파일 목록을 가져오기"""
        all_files = os.listdir(self.root_path)
        self.site_files = {}
        
        # 데이터셋에 대한 이름 매핑 가져오기
        
        
        for file_name in all_files:
            if not file_name.endswith('.csv'):
                continue
            if 'YMCA' in file_name:
                continue
         
            # 이름 매핑
            mapped_name = self.mapping_name[self.mapping_name['original_name'] == file_name]['mapping_name'].values[0]
            # 이름 정보 추출
            site_id = int(mapped_name.split('_')[0])
            capacity = float(mapped_name.split('_')[1])

            # 해당 flag에 속한 데이터가 아니면 skip
            if ('UK' not in self.data) and ('OEDI' not in self.data):     
                if site_id not in self.split_configs[self.data][self.flag]:
                    continue

            self.site_files.setdefault(site_id, []).append(file_name)
            self.capacity_dict[site_id] = capacity

            file_path = os.path.join(self.root_path, file_name)  
            df_raw = pd.read_csv(file_path)
            if 'Normalized_Active_Power' in df_raw.columns:
                df_raw.drop('Normalized_Active_Power', axis=1, inplace=True)


            # 날짜 별로 나누는 site의 경우, 필요한 날짜만 추출
            if ('OEDI' in self.data):
                df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
                data_mask = (df_raw['timestamp'] >= f'{self.split_configs[self.data][self.flag][0]}-01-01') & (df_raw['timestamp'] < f'{self.split_configs[self.data][self.flag][1]}-01-01')
                df_raw = df_raw[data_mask]
            
            # UK의 경우 퍼센트 비율로 나누기
            elif ('UK' in self.data):
                start_date = df_raw['timestamp'].min()
                end_date = df_raw['timestamp'].max()
                date_range = end_date - start_date

                train_end = start_date + date_range * self.split_configs[self.data]['train']
                val_end = start_date + date_range * self.split_configs[self.data]['train'] + self.split_configs[self.data]['val']


                if self.flag == 'train':
                    date_mask = (df_raw['timestamp'] >= start_date) & (df_raw['timestamp'] < train_end)
                elif self.flag == 'val':
                    date_mask = (df_raw['timestamp'] >= train_end) & (df_raw['timestamp'] < val_end)
                else:  # test
                    date_mask = (df_raw['timestamp'] >= val_end) & (df_raw['timestamp'] <= end_date)
                df_raw = df_raw[date_mask]

            # scale 적용
            if self.scale:
                self.scalers[site_id] = {}
                # if self.flag == 'train':
                self._fit_and_save_scalers(site_id, df_raw)
                # else:
                    # self._load_scalers(site_id, df_raw)
                self._apply_scalers_to_data(site_id, df_raw)

            df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)
            
            self.site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp,
                    'site_id': site_id,
                    'capacity': capacity,
                    'original_name': file_name,
                    'mapped_name': mapped_name
                })
            
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

          
    def _fit_and_save_scalers(self, site_id, data):
        """표준화(StandardScaler) 적용 및 저장"""
        os.makedirs(os.path.join(self.root_path, f'scalers/{site_id}'), exist_ok=True)
        
        for col in data.columns:
            if col == 'timestamp':
                continue
                
            scaler = StandardScaler()
            scaler.fit(data[[col]])
            self.scalers[site_id][col] = scaler

            with open(os.path.join(self.root_path, f'scalers/{site_id}/{col}_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)

        # Save statistics for the target variable
        target_stats = {
            'mean': data[self.target].mean(),
            'std': data[self.target].std(),
            'capacity': self.capacity_dict[site_id]
        }
        with open(os.path.join(self.root_path, f'scalers/{site_id}/{self.target}_stats_train.pkl'), 'wb') as f:
            pickle.dump(target_stats, f)


    def _apply_scalers_to_data(self, site_id, df_raw):
        """스케일러 적용"""
        print(f"Before scaling - {self.target} range:", df_raw[self.target].min(), df_raw[self.target].max())
        
        for col in df_raw.columns:
            if col == 'timestamp':
                continue
            # print(f"Applying scaler to {col}")
            scaler = self.scalers[site_id][col]
            df_raw[col] = scaler.transform(df_raw[[col]])
                
        print(f"After scaling - {self.target} range:", df_raw[self.target].min(), df_raw[self.target].max())

    def _load_scalers(self, site_id, data):
        """표준화(StandardScaler) 로드"""
        for col in data.columns:
            if col == 'timestamp':
                continue

            with open(os.path.join(self.root_path, f'scalers/{site_id}/{col}_scaler.pkl'), 'rb') as f:
                self.scalers[site_id][col] = pickle.load(f)
        
        with open(os.path.join(self.root_path, f'scalers/{site_id}/{self.target}_stats_train.pkl'), 'rb') as f:
            target_stats = pickle.load(f)
            self.capacity_dict[site_id] = target_stats['capacity']

    def inverse_transform(self, locations, data):
        """
        StandardScaler 역변환만 수행
        """
        batch_size, seq_length, n_channels = data.shape
        result = np.empty_like(data)

        for batch_idx in range(batch_size):
            site_id = locations[batch_idx].item() if torch.is_tensor(locations[batch_idx]) else locations[batch_idx]
            scaler = self.scalers[site_id][self.target]
            
            # StandardScaler 역변환만 수행
            batch_data = data[batch_idx].reshape(-1, n_channels)
            inverse_transformed = scaler.inverse_transform(batch_data)
            result[batch_idx] = inverse_transformed.reshape(seq_length, n_channels)

        return result

    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        # print(f"Target range - Min: {seq_y.min():.2f}, Max: {seq_y.max():.2f}")
        # print(f"Input range - Min: {seq_x.min():.2f}, Max: {seq_x.max():.2f}")
        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds

    def __len__(self):
        return len(self.indices)
    

    def _process_file(self, df_raw, site_id):
        """시간 인코딩 및 데이터 처리"""
        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)

        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else:
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T

        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values

        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove(self.target)

        if self.features == 'M' or self.features == 'MS':
            df_x = df_raw[cols + [self.target]]
        elif self.features == 'S':
            df_x = df_raw[[self.target]]

        df_y = df_raw[[self.target]]
        site = np.array([site_id]) * len(df_x)

        return df_x, df_y, time_feature, timestamp, site
    
    # def _get_name_mapping(self):
    #     """데이터셋별 파일 이름 매핑 반환 => inverse transform site 식별 위함"""
        
    #     mappings = {
    #         'DKASC_AliceSprings': {
    #             '1.8048249483108585_100-Site_DKA-M1_A-Phase.csv'    : '01_1.8048249483108585_100-Site_DKA-M1_A-Phase.csv',
    #             '5.489253242810574_57-Site_DKA-M16_A-Phase.csv'     : '02_5.489253242810574_57-Site_DKA-M16_A-Phase.csv',
    #             '4.0456333955129_99-Site_DKA-M4_C-Phase.csv'        : '03_4.0456333955129_99-Site_DKA-M4_C-Phase.csv',
    #             '5.5049305359522505_56-Site_DKA-M20_A-Phase.csv'    : '04_5.5049305359522505_56-Site_DKA-M20_A-Phase.csv',
    #             '4.606877764066067_70-Site_DKA-M5_A-Phase.csv'      : '05_4.606877764066067_70-Site_DKA-M5_A-Phase.csv',
    #             '5.5889055331548_59-Site_DKA-M19_C-Phase.csv'       : '06_5.5889055331548_59-Site_DKA-M19_C-Phase.csv',
    #             '4.681516726811734_106-Site_DKA-M5_C-Phase.csv'     : '07_4.681516726811734_106-Site_DKA-M5_C-Phase.csv',
    #             '5.588966925938926_55-Site_DKA-M20_B-Phase.csv'     : '08_5.588966925938926_55-Site_DKA-M20_B-Phase.csv',
    #             '4.713372310002634_74-Site_DKA-M18_C-Phase.csv'     : '09_4.713372310002634_74-Site_DKA-M18_C-Phase.csv',
    #             '5.641808271408075_60-Site_DKA-M18_A-Phase.csv'     : '10_5.641808271408075_60-Site_DKA-M18_A-Phase.csv',
    #             '4.722724914550775_84-Site_DKA-M5_B-Phase.csv'      : '11_4.722724914550775_84-Site_DKA-M5_B-Phase.csv',
    #             '5.648005247116092_64-Site_DKA-M17_B-Phase.csv'     : '12_5.648005247116092_64-Site_DKA-M17_B-Phase.csv',
    #             '5.025952617327375_214-Site_DKA-M18_B-Phase_II.csv' : '13_5.025952617327375_214-Site_DKA-M18_B-Phase_II.csv',
    #             '5.651080489158642_54-Site_DKA-M15_C-Phase.csv'     : '14_5.651080489158642_54-Site_DKA-M15_C-Phase.csv',
    #             '5.106038808822617_92-Site_DKA-M6_B-Phase.csv'      : '15_5.106038808822617_92-Site_DKA-M6_B-Phase.csv',
    #             '5.675088723500576_212-Site_DKA-M15_C-Phase_II.csv' : '16_5.675088723500576_212-Site_DKA-M15_C-Phase_II.csv',
    #             '5.1098860502243_58-Site_DKA-M17_C-Phase.csv'       : '17_5.1098860502243_58-Site_DKA-M17_C-Phase.csv',
    #             '5.858203013738009_205-Site_Archived_DKA-M15_BPhase_UMG_QCells.csv' : '18_5.858203013738009_205-Site_Archived_DKA-M15_BPhase_UMG_QCells.csv',
    #             '5.1862111488978_52-Site_DKA-M16_C-Phase.csv'       : '19_5.1862111488978_52-Site_DKA-M16_C-Phase.csv',
    #             '6.012480338414508_93-Site_DKA-M4_A-Phase.csv'      : '20_6.012480338414508_93-Site_DKA-M4_A-Phase.csv',
    #             '5.252291798591625_63-Site_DKA-M17_A-Phase.csv'     : '21_5.252291798591625_63-Site_DKA-M17_A-Phase.csv',
    #             '6.012586116790767_213-Site_DKA-M16_A-Phase_II.csv' : '22_6.012586116790767_213-Site_DKA-M16_A-Phase_II.csv',
    #             '5.279052972793592_77-Site_DKA-M18_B-Phase.csv'     : '23_5.279052972793592_77-Site_DKA-M18_B-Phase.csv',
    #             '6.107803106308_66-Site_DKA-M16_B-Phase.csv'        : '24_6.107803106308_66-Site_DKA-M16_B-Phase.csv',
    #             '5.291588862737025_73-Site_DKA-M19_A-Phase.csv'     : '25_5.291588862737025_73-Site_DKA-M19_A-Phase.csv',
    #             '6.494097034136466_79-Site_DKA-M6_A-Phase.csv'      : '26_6.494097034136466_79-Site_DKA-M6_A-Phase.csv',
    #             '5.377141674359625_71-Site_DKA-M2_C-Phase.csv'      : '27_5.377141674359625_71-Site_DKA-M2_C-Phase.csv',
    #             '6.640705426534017_69-Site_DKA-M4_B-Phase.csv'      : '28_6.640705426534017_69-Site_DKA-M4_B-Phase.csv',
    #             '5.381122271219891_218-Site_DKA-M4_C-Phase_II.csv'  : '29_5.381122271219891_218-Site_DKA-M4_C-Phase_II.csv',
    #             '8.239588975906374_67-Site_DKA-M8_A-Phase.csv'      : '30_8.239588975906374_67-Site_DKA-M8_A-Phase.csv',
    #             '5.392713745435067_72-Site_DKA-M15_B-Phase.csv'     : '31_5.392713745435067_72-Site_DKA-M15_B-Phase.csv',
    #             '8.4429276784261_68-Site_DKA-M8_C-Phase.csv'        : '32_8.4429276784261_68-Site_DKA-M8_C-Phase.csv',
    #             '5.424700140953067_90-Site_DKA-M3_A-Phase.csv'      : '33_5.424700140953067_90-Site_DKA-M3_A-Phase.csv',
    #             '8.563250303268434_98-Site_DKA-M8_B-Phase.csv'      : '34_8.563250303268434_98-Site_DKA-M8_B-Phase.csv',
    #             '5.4838057756424_61-Site_DKA-M15_A-Phase.csv'       : '35_5.4838057756424_61-Site_DKA-M15_A-Phase.csv',
    #             '9.772030671437577_85-Site_DKA-M7_A-Phase.csv'      : '36_9.772030671437577_85-Site_DKA-M7_A-Phase.csv'
    #         },

    #         'DKASC_Yulara': {
    #             '208.47536849975663_10-Site_SS-PV1-DB-SS-1A.csv'   : '01_208.47536849975663_10-Site_SS-PV1-DB-SS-1A.csv',
    #             '21.760133743286087_7-Site_SD-PV2-DB-SD-1A.csv'    : '02_21.760133743286087_7-Site_SD-PV2-DB-SD-1A.csv',
    #             '300.15113321940083_11-Site_LD-PV1-DB-LD-1A.csv'   : '03_300.15113321940083_11-Site_LD-PV1-DB-LD-1A.csv',
    #             '36.68936411539717_9-Site_SD-PV2-DB-SD-2A.csv'     : '04_36.68936411539717_9-Site_SD-PV2-DB-SD-2A.csv',
    #             '44.16863854726167_6-Site_SD-PV2-DB-SD-3A.csv'     : '05_44.16863854726167_6-Site_SD-PV2-DB-SD-3A.csv',
    #             '897.5805969238291_5-Site_DG-PV1-DB-DG-M1A.csv'    : '06_897.5805969238291_5-Site_DG-PV1-DB-DG-M1A.csv',
    #             '97.79799715677892_8-Site_CA-PV1-DB-CA-1A.csv'     : '07_97.79799715677892_8-Site_CA-PV1-DB-CA-1A.csv'
    #         },
    #         'GIST': {
    #             '66.0_C07_Samsung-Env-Bldg.csv'          : '01_66.0_C07_Samsung-Env-Bldg.csv',
    #             '37.4_C09_Dasan.csv'                     : '02_37.4_C09_Dasan.csv',
    #             '69.6_C10_Renewable-E-Bldg.csv'          : '03_69.6_C10_Renewable-E-Bldg.csv',
    #             '122.5_C11_GAIA.csv'                     : '04_122.5_C11_GAIA.csv',
    #             '76.8_E02_Animal-Recource-Center.csv'    : '05_76.8_E02_Animal-Recource-Center.csv',
    #             '133.1_E03_GTI.csv'                      : '06_133.1_E03_GTI.csv',
    #             '220.1_E8_Natural-Science-Bldg.csv'      : '07_220.1_E8_Natural-Science-Bldg.csv',
    #             '27.7_E11_DormA.csv'                     : '08_27.7_E11_DormA.csv',
    #             '52.1_E12_DormB.csv'                     : '09_52.1_E12_DormB.csv',
    #             '29.9_N01_Central-Library.csv'           : '10_29.9_N01_Central-Library.csv',
    #             '45.8_N02_LG-Library.csv'                : '11_45.8_N02_LG-Library.csv',
    #             '14.9_N06_College-Bldg.csv'              : '12_14.9_N06_College-Bldg.csv',
    #             '48.2_W06_Student-Union.csv'             : '13_48.2_W06_Student-Union.csv',
    #             '127.9_W11_Facility-Maintenance-Bldg.csv': '14_127.9_W11_Facility-Maintenance-Bldg.csv',
    #             '33.7_W13_Centeral-Storage.csv'          : '15_33.7_W13_Centeral-Storage.csv',
    #             '164.1_Soccer-Field.csv'                 : '16_164.1_Soccer-Field.csv'
    #         },
    #         'Germany': {
    #             '4.75_DE_KN_industrial1_pv_1.csv'                  : '01_4.75_DE_KN_industrial1_pv_1.csv',
    #             '4.170000000000073_DE_KN_industrial1_pv_2.csv'     : '02_4.170000000000073_DE_KN_industrial1_pv_2.csv',
    #             '14.599999999998545_DE_KN_industrial2_pv.csv'      : '03_14.599999999998545_DE_KN_industrial2_pv.csv',
    #             '6.055000000000291_DE_KN_industrial3_pv_facade.csv': '04_6.055000000000291_DE_KN_industrial3_pv_facade.csv',
    #             '11.100999999998749_DE_KN_industrial3_pv_roof.csv' : '05_11.100999999998749_DE_KN_industrial3_pv_roof.csv',
    #             '8.197000000000116_DE_KN_residential1_pv.csv'      : '06_8.197000000000116_DE_KN_residential1_pv.csv',
    #             '3.845999999999549_DE_KN_residential3_pv.csv'      : '07_3.845999999999549_DE_KN_residential3_pv.csv',
    #             '8.784999999999854_DE_KN_residential4_pv.csv'      : '08_8.784999999999854_DE_KN_residential4_pv.csv',
    #             '7.64600000000064_DE_KN_residential6_pv.csv'       : '09_7.64600000000064_DE_KN_residential6_pv.csv'
    #         },
    #         'Miryang': {
    #             '101.286_C_99kW.csv'  : '01_101.286_C_99kW.csv',
    #             '30.096_G_50kW.csv'   : '02_30.096_G_50kW.csv',
    #             '52.102_F_64kW.csv'   : '03_52.102_F_64kW.csv',
    #             '533.6_B_658kW.csv'   : '04_533.6_B_658kW.csv',
    #             '62.415_E_69kW.csv'   : '05_62.415_E_69kW.csv',
    #             '75.57_D_89kW.csv'    : '06_75.57_D_89kW.csv',
    #             '826.165_A_999kW.csv' : '07_826.165_A_999kW.csv'
    #         }
    #     }


    #     return mappings.get(self.data, {})
   

##########################################################################################
    
class Dataset_DKASC_AliceSprings(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='', scaler='MinMaxScaler',
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
        # assert flag in ['train', 'test', 'val']
        # type_map = {'train': 0, 'val': 1, 'test': 2}
        # self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        if self.flag == 'pred': 
            self.flag = 'test'

        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        
   
        # # Total 개
        # self.LOCATIONS = [
        #     '52-Site_DKA-M16_C-Phase.csv'    ,    # 33
        #     '54-Site_DKA-M15_C-Phase.csv'    ,
        #     '55-Site_DKA-M20_B-Phase.csv'    ,    # 29
        #     '56-Site_DKA-M20_A-Phase.csv'    ,    # 30
        #     '57-Site_DKA-M16_A-Phase.csv'    ,
        #     '58-Site_DKA-M17_C-Phase.csv'    ,    # 36
        #     '59-Site_DKA-M19_C-Phase.csv'    ,    # 38
        #     '60-Site_DKA-M18_A-Phase.csv'    ,
        #     '61-Site_DKA-M15_A-Phase.csv'    ,    # 23
        #     '63-Site_DKA-M17_A-Phase.csv'    ,    # 34
        #     '64-Site_DKA-M17_B-Phase.csv'    ,    # 37
        #     '66-Site_DKA-M16_B-Phase.csv'    ,
        #     '67-Site_DKA-M8_A-Phase.csv'     ,    # 21
        #     '68-Site_DKA-M8_C-Phase.csv'     ,    # 20
        #     '69-Site_DKA-M4_B-Phase.csv'     ,    # 17
        #     '70-Site_DKA-M5_A-Phase.csv'     ,    # 3
        #     '71-Site_DKA-M2_C-Phase.csv'     ,    # 18
        #     '72-Site_DKA-M15_B-Phase.csv'    ,    # 26
        #     '73-Site_DKA-M19_A-Phase.csv'    ,    # 35
        #     '74-Site_DKA-M18_C-Phase.csv'    ,    # 31
        #     '77-Site_DKA-M18_B-Phase.csv'    ,
        #     '79-Site_DKA-M6_A-Phase.csv'     ,    # 7
        #     '84-Site_DKA-M5_B-Phase.csv'     ,    # 12
        #     '85-Site_DKA-M7_A-Phase.csv'     ,    # 10
        #     '90-Site_DKA-M3_A-Phase.csv'     ,    # 14
        #     '92-Site_DKA-M6_B-Phase.csv'     ,    # 13
        #     '93-Site_DKA-M4_A-Phase.csv'     ,    # 8
        #     '97-Site_DKA-M10_B+C-Phases.csv' ,    # 22
        #     '98-Site_DKA-M8_B-Phase.csv'     ,    # 19
        #     '98-Site_DKA-M8_B-Phase(site19).csv',
        #     '99-Site_DKA-M4_C-Phase.csv'     ,
        #     '100-Site_DKA-M1_A-Phase.csv'    ,    # 16A
        #     '212-Site_DKA-M15_C-Phase_II.csv',    # 25
        #     '213-Site_DKA-M16_A-Phase_II.csv',    # 24
        #     '214-Site_DKA-M18_B-Phase_II.csv',    # 32
        #     '218-Site_DKA-M4_C-Phase_II.csv'      # 9A
        # ]

    
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        # site 별로 데이터 저장
        site_all_data = {site_id: [] for site_id in selected_sites}
        # all_data = [] # 전체 데이터셋

        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp
                })

                # for scaling
                site_all_data[site_id].append(df_x)

        
        # combined_data = pd.concat(all_data, axis=0)
           
        self.scalers = {}          
        if self.scaler:
            for site_id in selected_sites:
                self.scalers[site_id] = {}
                site_combined_data = pd.concat(site_all_data[site_id], axis=0)

                if self.flag == 'train':

                    for col in site_combined_data.columns:
                        if col == 'timestamp': continue
                        if self.scaler == 'MinMaxScaler':
                            scaler = MinMaxScaler()
                        else: 
                            scaler = StandardScaler()
                    
                        scaler.fit(site_combined_data[[col]])
                        self.scalers[site_id][col] = scaler

                        os.makedirs(os.path.join(self.root_path, f'scalers/{site_id}'), exist_ok=True)
                        with open(os.path.join(self.root_path, f'scalers/{site_id}/{col}_scaler.pkl'), 'wb') as f:
                            pickle.dump(scaler, f)

                    with open(os.path.join(self.root_path, f'scalers/{site_id}/Active_Power_min_max_train.pkl'), 'wb') as f:
                        print(f"Active_Power min: {site_combined_data['Active_Power'].min()}")
                        print(f"Active_Power max: {site_combined_data['Active_Power'].max()}")
                        pickle.dump([site_combined_data['Active_Power'].min(), site_combined_data['Active_Power'].max()], f)


                else:
                    for col in site_combined_data.columns:
                        if col == 'timestamp': continue
                        with open(os.path.join(self.root_path, f'scalers/{site_id}/{col}_scaler.pkl'), 'rb') as f:
                            self.scalers[site_id][col] = pickle.load(f)

                        
                if self.flag == 'val':
                    with open(os.path.join(self.root_path, f'scalers/{site_id}/Active_Power_min_max_val.pkl'), 'wb') as f:
                            print(f"Site {site_id} Active_Power min: {site_combined_data['Active_Power'].min()}")
                            print(f"Site {site_id} Active_Power max: {site_combined_data['Active_Power'].max()}")
                            pickle.dump([site_combined_data['Active_Power'].min(), site_combined_data['Active_Power'].max()], f)
                
                elif self.flag == 'test':
                    train_path = os.path.join(self.root_path, f'scalers/{site_id}/Active_Power_min_max_train.pkl')
                    val_path = os.path.join(self.root_path, f'scalers/{site_id}/Active_Power_min_max_val.pkl')

                    with open(train_path, 'rb') as f:
                        train_min, train_max = pickle.load(f)
                    with open(val_path, 'rb') as f:
                        val_min, val_max = pickle.load(f)

                    self.ap_max_by_site = getattr(self, 'ap_max_by_site', {})
                    self.ap_min_by_site = getattr(self, 'ap_min_by_site', {})
                    self.ap_max_by_site[site_id] = max(train_max, val_max)
                    self.ap_min_by_site[site_id] = min(train_min, val_min)

                  
                
            # scaler 적용 및 site data update
            for data in site_data[site_id]:
                for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[site_id][col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])

        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('-')[0])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            

            train_keys=[57, 61, 70, 92, 59, 212, 213, 218, 56, 66, 52, 90, 72, 77, 60, 74, 67, 73, 214, 58, 68, 54, 79, 84]
            val_keys=[64, 99, 71, 98, 93, 100, 97]
            test_keys=[63, 85, 55, 69]

            site_split = {
                'train': train_keys,
                'val': val_keys,
                'test': test_keys  
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)

    # 평가 시 필요함
    def inverse_transform(self, data, locations):
   
        batch_size, seq_length, n_channels = data.shape
    
        result = np.empty_like(data)
    
        # 배치 단위로 처리
        for batch_idx in range(batch_size):
            site_id = locations[batch_idx].item() if torch.is_tensor(locations[batch_idx]) else locations[batch_idx]
            scaler = self.scalers[site_id]['Active_Power']
            
            # 채널별 데이터를 한 번에 처리 
            batch_data = data[batch_idx].reshape(-1, n_channels)
            inverse_transformed = scaler.inverse_transform(batch_data)
            result[batch_idx] = inverse_transformed.reshape(seq_length, n_channels)
    
        return result
    


class Dataset_DKASC_Yulara(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='', scaler='MinMaxScaler',
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
        self.flag = flag

        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        

        # Total 개
        self.LOCATIONS = [
            '10-Site_SS-PV1-DB-SS-1A.csv',
            '5-Site_DG-PV1-DB-DG-M1A.csv',
            '7-Site_SD-PV2-DB-SD-1A.csv',
            '9-Site_SD-PV2-DB-SD-2A.csv',
            '11-Site_LD-PV1-DB-LD-1A.csv',
            '6-Site_SD-PV2-DB-SD-3A.csv',
            '8-Site_CA-PV1-DB-CA-1A.csv'
        ]

        self.scalers = {}
        self.x_list = []
        self.y_list = []
        self.ds_list = []


        random.seed(42)
        if self.data_path['type'] == 'all':
            random.shuffle(self.LOCATIONS)  # 데이터 섞기
            total_size = len(self.LOCATIONS)
            train_size = int(0.6 * total_size)
            val_size = int(0.3 * total_size)
                    
            if self.flag == 'train':
                self.train = self.LOCATIONS[:train_size]
                print(f'[INFO] Train Locations: total {len(self.train)}')
                print(f'[INFO] Location List: {self.train}')
                
            elif self.flag == 'val':
                self.val = self.LOCATIONS[train_size:train_size + val_size]
                print(f'[INFO] Valid Locations: total {len(self.val)}')
                print(f'[INFO] Location List: {self.val}')

            elif self.flag == 'test':
                self.test = self.LOCATIONS[train_size + val_size:]
                print(f'[INFO] Test Locations: total {len(self.test)}')
                print(f'[INFO] Location List: {self.test}')



        elif self.data_path['type'] == 'debug':
            
            if self.flag == 'train':
                self.train = self.data_path['train']
            
            elif self.flag == 'val':
                self.val = self.data_path['val']
            
            elif self.flag == 'test':
                self.test = self.data_path['test']
            

        
        self.load_preprocessed_data()

        
    
    def load_preprocessed_data(self):

        # 1. Flag에 맞게 데이터 불러오기
        if self.flag == 'train':
            df_raw = self.load_and_concat_data(self.train)
        
        elif self.flag == 'val':
            df_raw = self.load_and_concat_data(self.val)

        elif self.flag == 'test':
            df_raw = self.load_and_concat_data(self.test)

        

        # 2. Time encoding (년, 월, 일, 요일, 시간)
        if self.timeenc == 0:
            data_stamp = pd.DataFrame()
            data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
            data_stamp['year'] = df_raw.date.apply(lambda row: row.year, 1)
            data_stamp['month'] = df_raw.timestamp.apply(lambda row: row.month, 1)
            data_stamp['day'] = df_raw.date.apply(lambda row: row.day, 1)
            data_stamp['weekday'] = df_raw.date.apply(lambda row: row.weekday(), 1)
            data_stamp['hour'] = df_raw.date.apply(lambda row: row.hour, 1)
            
            data_stamp = data_stamp[['month', 'day', 'weekday', 'hour']].values

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # # 3. Encoding 후, 'date' column 생성
        # df_raw['date'] = data_stamp
       
        # 4. Columns 순서 정렬 (timestamp, date, ..... , target)
        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove('Active_Power')
        cols.remove('Wind_Speed')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw = df_raw[cols + [self.target]]


        # 5. Scaler 적용
        if self.scale: 
            # Train일 때는, Scaler Fit 후에, 저장
            if self.flag == 'train':
                for col in df_raw.columns:
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: scaler = StandardScaler()
                    df_raw[col] = scaler.fit_transform(df_raw[[col]])
                    self.scalers[col] = scaler 
                    # Scaler를 pickle 파일로 저장
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(scaler, f)
        
            else:
                # Val, Test일 때는, 저장된 Scaler 불러와서 적용
                self.scalers = {}
                transformed_df = df_raw.copy()  

                for col in df_raw.columns:                   
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
                 
                    # Scaler가 존재하는지 확인
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            scaler = pickle.load(f) 
                
                        # 해당 칼럼에 스케일러 적용 (transform)
                        df_raw[col] = scaler.transform(transformed_df[[col]])
                        self.scalers[col] = scaler
                
                    else:
                        print(f"Scaler for column {col} not found.")
            

        # 6. 입력할 칼럼들 지정하여 리스트 생성
        if self.features == 'M' or self.features == 'MS':
            # date 열을 제외
            cols_data = df_raw.columns[1:]
            df_x = df_raw[cols_data]
        elif self.features == 'S':
            # Active Power만 추출
            df_x = df_raw[[self.target]]

        self.x_list = df_x.values
        # 타겟은 마지막 열인 Active_Power
        self.y_list = df_raw[[self.target]].values
        # date columns만 선택
        self.ds_list = data_stamp


    
    # 파일 경로를 가져와서 DataFrame으로 합치는 함수
    def load_and_concat_data(self, file_list):
        df_list = []
        for file in file_list:
            file_path = os.path.join(self.root_path, file)
            df = pd.read_csv(file_path)  
            assert (df.isnull().sum()).sum() == 0, "허용되지 않은 열에 결측치가 존재합니다."            
            df_list.append(df) 
        return pd.concat(df_list, ignore_index=True) 

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.x_list[s_begin:s_end]
        seq_y = self.y_list[r_begin:r_end]
        seq_x_mark = self.ds_list[s_begin:s_end]
        seq_y_mark = self.ds_list[r_begin:r_end]
            
        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark


    def __len__(self):    
        return len(self.x_list) - self.seq_len - self.pred_len + 1



    def inverse_transform(self, data, locations):
   
        batch_size, seq_length, n_channels = data.shape
    
        result = np.empty_like(data)
    
        # 배치 단위로 처리
        for batch_idx in range(batch_size):
            site_id = locations[batch_idx].item() if torch.is_tensor(locations[batch_idx]) else locations[batch_idx]
            scaler = self.scalers[site_id]['Active_Power']
            
            # 채널별 데이터를 한 번에 처리 
            batch_data = data[batch_idx].reshape(-1, n_channels)
            inverse_transformed = scaler.inverse_transform(batch_data)
            result[batch_idx] = inverse_transformed.reshape(seq_length, n_channels)
    
        return result

class Dataset_GIST(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='GIST_sisuldong.csv', target='Active_Power',
                 scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
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
        self.flag = flag

    
        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        self.LOCATIONS = [
            'C07_Samsung-Env-Bldg.csv',
            'C09_Dasan.csv',
            'C10_Renewable-E-Bldg.csv',
            'C11_GAIA.csv',
            'E02_Animal-Recource-Center.csv',
            'E03_GTI.csv',
            'E08_Natural-Science-Bldg.csv',
            'E11_DormA.csv',
            'E12_DormB.csv',
            'N01_Central-Library.csv',
            'N02_LG-Library.csv',
            'N06_College-Bldg.csv',
            'W06_Student-Union.csv',
            'W11_Facility-Maintenance-Bldg.csv', 
            'W13_Centeral-Storage.csv',
            'O14_Soccer-Field.csv',
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋

        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp
                })

                # for scaling
                all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                        pickle.dump(scaler, f)
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                    print(f"Active_Power min: {combined_data['Active_Power'].min()}")
                    print(f"Active_Power max: {combined_data['Active_Power'].max()}")
                    pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)
            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()

            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0][1:])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            
            # TODO: 재현 가능하게 고정하기
            # [INFO] Train sites: [92, 214, 52, 61, 99, 60, 72, 68, 55, 67, 64, 212, 59, 71, 69, 66, 90, 73, 70, 79, 213, 74, 85]
            # train 1310017
            # [INFO] Val sites: [54, 58, 63, 98, 77, 84]
            # val 348608
            # [INFO] Test sites: [57, 93, 56, 218, 100]
            # test 250599

            random.seed(42)
            random.shuffle(site_ids)
            total_sites = len(site_ids)
            train_end = int(total_sites * 0.7)
            val_end = train_end + int(total_sites * 0.2)

            site_split = {
                'train': site_ids[:train_end],
                'val': site_ids[train_end:val_end],
                'test': site_ids[val_end:]  
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
       
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data    


class Dataset_German(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='', target='Active_Power',
                 scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
        # size [seq_len, label_len, pred_len]
        # info
        print("start")
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
        self.flag = flag

    
        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        self.LOCATIONS = {
            'DE_KN_industrial1_pv_1.csv'     : '01_DE_KN_industrial1_pv_1.csv',
            'DE_KN_industrial1_pv_2.csv'     : '02_DE_KN_industrial1_pv_2.csv',
            'DE_KN_industrial2_pv.csv'       : '03_DE_KN_industrial2_pv.csv',
            'DE_KN_industrial3_pv_facade.csv': '04_DE_KN_industrial3_pv_facade.csv',
            'DE_KN_industrial3_pv_roof.csv'  : '05_DE_KN_industrial3_pv_roof.csv',
            'DE_KN_residential1_pv.csv'      : '06_DE_KN_residential1_pv.csv',
            'DE_KN_residential3_pv.csv'      : '07_DE_KN_residential3_pv.csv',
            'DE_KN_residential4_pv.csv'      : '08_DE_KN_residential4_pv.csv',
            'DE_KN_residential6_pv.csv'      : '09_DE_KN_residential6_pv.csv'
        }


        if self.flag == 'train':
            # 파일명 변경 수행
            for original_name, new_name in self.LOCATIONS.items():
                original_path = os.path.join(self.root_path, original_name)
                new_path = os.path.join(self.root_path, new_name)
                
                # 파일이 존재하는 경우에만 이름을 변경
                if os.path.exists(original_path):
                    os.rename(original_path, new_path)
                    print(f"{original_name} -> {new_name} (파일명 변경 완료)")


        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋
        self.site_ap_max = {}
        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                self.site_ap_max[site_id] = df_raw['Active_Power'].max()
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp,
                    'ap_max': self.site_ap_max[site_id]
                })

                # for scaling
                all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                                pickle.dump(scaler, f)
                # with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                #     print(f"Train Active_Power min: {combined_data['Active_Power'].min()}")
                #     print(f"Train Active_Power max: {combined_data['Active_Power'].max()}")
                #     pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)
            
            
            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()


            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            # site_ids = list(self.site_files.keys())
            
            train_sites = [8, 1, 7, 3, 4, 6]
            val_sites = [9]
            test_sites = [2, 5]

            site_split = {
                'train': train_sites,
                'val': val_sites,
                'test': test_sites  
            }

        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]
        # site_ap_max = np.array([data['ap_max']]).repeat(s_end - s_begin).reshape(-1, 1)

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)

    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
       
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data

class Dataset_UK(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='', target='Active_Power',
                 scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
        # size [seq_len, label_len, pred_len]
        # info
        print("start")
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
        self.flag = flag

    
        self.scaler = scaler

        self.root_path = root_path
        self.data_path = data_path
        

        
        self.LOCATIONS = [
            '01_Forest_Road.csv',
            '02_Maple_Drive_East.csv',
            '03_YMCA.csv'
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        # selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋

        # for site_id in selected_sites:
        for site_id, file_name in self.site_files.items():
           
            file_path = os.path.join(self.root_path, file_name)
            df_raw = pd.read_csv(file_path)
            
            # 날짜별로 분리
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='raise')
            df_raw.sort_values('timestamp', inplace=True)
            df_raw.reset_index(drop=True, inplace=True)

            start_date = df_raw['timestamp'].min()
            end_date = df_raw['timestamp'].max()
            total_days = (end_date - start_date).days
            
            train_end = start_date + pd.Timedelta(days=int(total_days * 0.5))
            val_end = train_end + pd.Timedelta(days=int(total_days * 0.3))

            if self.flag == 'train':
                df_raw = df_raw[df_raw['timestamp'] <= train_end]
            elif self.flag == 'val':
                df_raw = df_raw[(df_raw['timestamp'] > train_end) & (df_raw['timestamp'] <= val_end)]
            else: df_raw = df_raw[df_raw['timestamp'] > val_end]
            ###################

            df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
            site_data.setdefault(site_id, []).append({
                'x': df_x,
                'y': df_y,
                'data_stamp': time_feature,
                'timestamp': timestamp
            })

                # for scaling
            all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                                pickle.dump(scaler, f)
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                    print(f"Active_Power min: {combined_data['Active_Power'].min()}")
                    print(f"Active_Power max: {combined_data['Active_Power'].max()}")
                    pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)
                                
            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()

            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0])
                self.site_files[site_id] = file_name
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files[site_id] = self.data_path[key]
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            print(self.site_files)

            random.seed(42)
            random.shuffle(site_ids)
            total_sites = len(site_ids)
            train_end = int(total_sites * 0.7)
            val_end = train_end + int(total_sites * 0.2)

            site_split = {
                'train': site_ids[:train_end],
                'val': site_ids[train_end:val_end],
                'test': site_ids[val_end:]  
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
       
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data
    
class Dataset_OEDI_Georgia(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='', target='Active_Power',
                 scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
        # size [seq_len, label_len, pred_len]
        # info
        print("start")
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
        self.flag = flag

    
        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        import os

        # # 현재 파일의 경로 기준으로 preprocessed 디렉토리 경로 생성
        # current_file_path = os.path.abspath(__file__)
        # preprocessed_dir = os.path.join(os.path.dirname(current_file_path), '../data/OEDI/9069(Georgia)/preprocessed')

        # preprocessed 폴더 내의 CSV 파일명들을 LOCATIONS에 저장
        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋

        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp
                })

                # for scaling
                all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                                pickle.dump(scaler, f)
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                    print(f"Active_Power min: {combined_data['Active_Power'].min()}")
                    print(f"Active_Power max: {combined_data['Active_Power'].max()}")
                    pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)


            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()

            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
     
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('.')[0].split('inv')[-1])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            
            # TODO: 재현 가능하게 고정하기
            # [INFO] Train sites: [92, 214, 52, 61, 99, 60, 72, 68, 55, 67, 64, 212, 59, 71, 69, 66, 90, 73, 70, 79, 213, 74, 85]
            # train 1310017
            # [INFO] Val sites: [54, 58, 63, 98, 77, 84]
            # val 348608
            # [INFO] Test sites: [57, 93, 56, 218, 100]
            # test 250599

            random.seed(42)
            random.shuffle(site_ids)
            total_sites = len(site_ids)
            train_end = int(total_sites * 0.7)
            val_end = train_end + int(total_sites * 0.2)

            site_split = {
                'train': site_ids[:train_end],
                'val': site_ids[train_end:val_end],
                'test': site_ids[val_end:]  
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
       
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data
    
class Dataset_OEDI_California(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='', target='Active_Power',
                 scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
        # size [seq_len, label_len, pred_len]
        # info
        print("start")
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
        self.flag = flag

    
        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        import os

        # # 현재 파일의 경로 기준으로 preprocessed 디렉토리 경로 생성
        # current_file_path = os.path.abspath(__file__)
        # preprocessed_dir = os.path.join(os.path.dirname(current_file_path), '../data/OEDI/2107(Arbuckle_California)/preprocessed')

        # preprocessed 폴더 내의 CSV 파일명들을 LOCATIONS에 저장
        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋

        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp
                })

                # for scaling
                all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                                pickle.dump(scaler, f)
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                    print(f"Active_Power min: {combined_data['Active_Power'].min()}")
                    print(f"Active_Power max: {combined_data['Active_Power'].max()}")
                    pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)


            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()

            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
     
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('.')[0].split('inv')[-1])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            
            # TODO: 재현 가능하게 고정하기
            # [INFO] Train sites: [92, 214, 52, 61, 99, 60, 72, 68, 55, 67, 64, 212, 59, 71, 69, 66, 90, 73, 70, 79, 213, 74, 85]
            # train 1310017
            # [INFO] Val sites: [54, 58, 63, 98, 77, 84]
            # val 348608
            # [INFO] Test sites: [57, 93, 56, 218, 100]
            # test 250599

            random.seed(42)
            random.shuffle(site_ids)
            total_sites = len(site_ids)
            train_end = int(total_sites * 0.7)
            val_end = train_end + int(total_sites * 0.2)

            site_split = {
                'train': site_ids[:train_end],
                'val': site_ids[train_end:val_end],
                'test': site_ids[val_end:]  
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('-')[0])],
                'val': [int(self.data_path['val'].split('-')[0])],
                'test': [int(self.data_path['test'].split('-')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
       
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data
    
class Dataset_Miryang(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='', target='Active_Power',
                scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
    # size [seq_len, label_len, pred_len]
    # info
        print("start")
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
        self.flag = flag


        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        import os

        # # 현재 파일의 경로 기준으로 preprocessed 디렉토리 경로 생성
        # current_file_path = os.path.abspath(__file__)
        # preprocessed_dir = os.path.join(os.path.dirname(current_file_path), '../data/OEDI/2107(Arbuckle_California)/preprocessed')

        # preprocessed 폴더 내의 CSV 파일명들을 LOCATIONS에 저장
        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        all_data = [] # 전체 데이터셋

        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)

                # 데이터 저장
                site_data.setdefault(site_id, []).append({
                    'x': df_x,
                    'y': df_y,
                    'data_stamp': time_feature,
                    'timestamp': timestamp
                })

                # for scaling
                all_data.append(df_x)

        
        combined_data = pd.concat(all_data, axis=0)
           
            # 스케일러 적용        
        if self.scaler:
            # train때는 전체 데이터셋에 대해 칼럼 별로 스케일러를 fit
            if self.flag == 'train':
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    if self.scaler == 'MinMaxScaler':
                        scaler = MinMaxScaler()
                    else: 
                        scaler = StandardScaler()
                    
                    scaler.fit(combined_data[[col]])
                    self.scalers[col] = scaler
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'wb') as f:
                                pickle.dump(scaler, f)
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'wb') as f:
                    print(f"Active_Power min: {combined_data['Active_Power'].min()}")
                    print(f"Active_Power max: {combined_data['Active_Power'].max()}")
                    pickle.dump([combined_data['Active_Power'].min(), combined_data['Active_Power'].max()], f)


            else:
                for col in combined_data.columns:
                    if col == 'timestamp': continue
                    with open(os.path.join(self.root_path, f'{col}_scaler.pkl'), 'rb') as f:
                        scaler = pickle.load(f)
                        self.scalers[col] = scaler
                with open(os.path.join(self.root_path, f'Active_Power_min_max.pkl'), 'rb') as f:
                    min_max = pickle.load(f)
                    self.ap_min = min_max[0]
                    self.ap_max = min_max[1]
                    self.ap_min_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_min]).reshape(1, -1)).squeeze()
                    self.ap_max_normalized = self.scalers['Active_Power'].transform(np.array([self.ap_max]).reshape(1, -1)).squeeze()

            # scaler 적용 및 site data update
            for site_id in site_data.keys():
                for data in site_data[site_id]:
                    for col in data['x'].columns:
                        if col == 'timestamp': continue
                        scaler = self.scalers[col]
                        data['x'][col] = scaler.transform(data['x'][[col]])  
                        if col == 'Active_Power':    
                            data['y'][col] = scaler.transform(data['y'][[col]])
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
     
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            site_ids = list(self.site_files.keys())
            
            # TODO: 재현 가능하게 고정하기
            # [INFO] Train sites: [92, 214, 52, 61, 99, 60, 72, 68, 55, 67, 64, 212, 59, 71, 69, 66, 90, 73, 70, 79, 213, 74, 85]
            # train 1310017
            # [INFO] Val sites: [54, 58, 63, 98, 77, 84]
            # val 348608
            # [INFO] Test sites: [57, 93, 56, 218, 100]
            # test 250599
            train_sites = [1, 4, 7]
            val_sites = [2, 6]
            test_sites = [3, 5]
            # random.seed(42)
            # random.shuffle(site_ids)
            # total_sites = len(site_ids)
            # train_end = int(total_sites * 0.7)
            # val_end = train_end + int(total_sites * 0.2)

            site_split = {
                'train': train_sites,
                'val': val_sites,
                'test': test_sites 
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('_')[0])],
                'val': [int(self.data_path['val'].split('_')[0])],
                'test': [int(self.data_path['test'].split('_')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
        # print(data.shape)
        # print(data[-1].shape) #16,1
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data
    

    # def inverse_transform_tensor(self, data):
    #     data_org = data.clone()
    #     inverse_data = torch.zeros_like(data_org)
        
        
    #     scaler = self.scalers['Active_Power']
            
    #     # 스케일러의 종류에 따라 역변환 방식 결정
    #     if isinstance(scaler, MinMaxScaler):
    #         data_min = torch.tensor(scaler.data_min_, device=data.device)
    #         data_max = torch.tensor(scaler.data_max_, device=data.device)
    #         data_range = data_max - data_min
    #         inverse_sample = data[i] * data_range + data_min
    #     elif isinstance(scaler, StandardScaler):
    #         mean = torch.tensor(scaler.mean_, device=data.device)
    #         scale = torch.tensor(scaler.scale_, device=data.device)
    #         inverse_sample = data[i] * scale + mean
    #     else:
    #         raise ValueError(f"Unsupported scaler type: {type(scaler)}")
        
    #     inverse_data[i] = inverse_sample.reshape(data[i].shape)
    
    # return inverse_data

####################################################

class Dataset_Miryang_MinMax(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='', target='Active_Power',
                scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
    # size [seq_len, label_len, pred_len]
    # info
        print("start")
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
        self.flag = flag


        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        import os

        # # 현재 파일의 경로 기준으로 preprocessed 디렉토리 경로 생성
        # current_file_path = os.path.abspath(__file__)
        # preprocessed_dir = os.path.join(os.path.dirname(current_file_path), '../data/OEDI/2107(Arbuckle_California)/preprocessed')

        # preprocessed 폴더 내의 CSV 파일명들을 LOCATIONS에 저장
        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    # site 별로 scaler 적용
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        # all_data = [] # 전체 데이터셋
        self.scalers = {}
        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            self.scalers[site_id] = {}
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                
                
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)
                
                # if self.flag == 'train':
                for col in df_x.columns:
                    scaler = MinMaxScaler()
                    if col == 'timestamp': continue
                    elif col == 'Active_Power':
                        capacity_ap_max = int(os.path.basename(file_path).split('_')[1].split('kW')[0])
                        capacity_ap_min = 0
                        # # `data_min_`과 `data_max_` 속성에 수동으로 설정할 값을 지정
                        # scaler.data_min_ = capacity_ap_min
                        # scaler.data_max_ = capacity_ap_max
                        # scaler.data_range_ = scaler.data_max_ - scaler.data_min_

                        # # scale_ 속성은 feature_range에 맞춘 스케일을 정의 (data_range로 자동 계산됨)
                        # scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / scaler.data_range_
                        # scaler.min_ = scaler.feature_range[0] - scaler.data_min_ * scaler.scale_
                        scaler.fit([[capacity_ap_min], [capacity_ap_max]])
                        
                        df_y[col] = scaler.transform(df_y[[col]])
                        df_x[col] = scaler.transform(df_x[[col]])
                    else:
                        df_x[col] = scaler.fit_transform(df_x[[col]])

                    self.scalers[site_id][col] = scaler

                os.makedirs(os.path.join(self.root_path, 'scaler'), exist_ok=True)
                with open(os.path.join(self.root_path, 'scaler', f'{site_id}_scalers.pkl'), 'wb') as f:
                    pickle.dump(self.scalers[site_id], f)
                # else:
                #     with open(os.path.join(self.root_path, 'scaler', f'{site_id}_scalers.pkl'), 'rb') as f:
                #         self.scalers[site_id] = pickle.load(f)

                #     for col in df_x.columns:
                #         if col == 'timestamp': continue
                #         scaler = self.scalers[site_id][col]
                #         df_x[col] = scaler.transform(df_x[[col]])
                #         if col == 'Active_Power':
                #             df_y[col] = scaler.transform(df_y[[col]])
                        

            # 데이터 저장
            site_data.setdefault(site_id, []).append({
                'x': df_x,
                'y': df_y,
                'data_stamp': time_feature,
                'timestamp': timestamp
            })

                # for scaling
        #     all_data.append(df_x)

        
        # combined_data = pd.concat(all_data, axis=0)
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
     
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            # site_ids = list(self.site_files.keys())
            
            train_sites = [1, 4, 7]
            val_sites = [2, 6]
            test_sites = [3, 5]

            site_split = {
                'train': train_sites,
                'val': val_sites,
                'test': test_sites 
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('_')[0])],
                'val': [int(self.data_path['val'].split('_')[0])],
                'test': [int(self.data_path['test'].split('_')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    # def inverse_transform(self, site, data):
    #     data_org = data.copy()
       
    #     data[-1] = self.scalers[site]['Active_Power'].inverse_transform(data[-1].reshape(-1, 1))
    #     data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
    #     return data

    def inverse_transform(self, sites, data):
        # 결과를 저장할 배열 초기화
        # print(data.shape) (1024, 16, 1)
        # print(sites.shape) torch.Size([1024, 1])
        data_org = data.copy()
        inverse_data = np.zeros_like(data.squeeze())
        # print(sites.shape) torch.Size([1024, 256, 1])
        # print(data.shape) (1024, 16, 1)
        # 각 사이트별로 데이터의 역변환 수행
        for i, site_id in enumerate(sites.squeeze()):
            # print('site_id', site_id)
            # print(site_id)
            # print(site_id.shape) torch.Size([256])
            inverse_data[i] = self.scalers[site_id.item()]['Active_Power'].inverse_transform(data[i].reshape(-1, 1)).flatten()
        inverse_data.reshape(data_org.shape[0], data_org.shape[1], 1)
        # print(f'{site_id}inverse, {inverse_data}')
        return inverse_data


class Dataset_Miryang_Standard(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='', target='Active_Power',
                scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
    # size [seq_len, label_len, pred_len]
    # info
        print("start")
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
        self.flag = flag


        self.scaler = scaler
        

        self.root_path = root_path
        self.data_path = data_path
        

        
        import os

        # # 현재 파일의 경로 기준으로 preprocessed 디렉토리 경로 생성
        # current_file_path = os.path.abspath(__file__)
        # preprocessed_dir = os.path.join(os.path.dirname(current_file_path), '../data/OEDI/2107(Arbuckle_California)/preprocessed')

        # preprocessed 폴더 내의 CSV 파일명들을 LOCATIONS에 저장
        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.site_data = {}

        self.load_preprocessed_data()
        self.indices = self.create_sequences_indices()

        
    # site 별로 scaler 적용
    def load_preprocessed_data(self):
        # 파일 로드 및 사이트별 데이터 수집
        self._load_files()
        # train, val, test split, 해당하는 flag에 맞게 데이터 가져옴
        selected_sites = self._split_sites()[self.flag]
        site_data = {}
        # all_data = [] # 전체 데이터셋
        self.scalers = {}
        self.ap_max = {}
        for site_id in selected_sites:
            site_files = self.site_files.get(site_id, [])
            self.scalers[site_id] = {}
            for file_name in site_files:
                file_path = os.path.join(self.root_path, file_name)
                df_raw = pd.read_csv(file_path)
                for col in df_raw.columns:
                    scaler = MinMaxScaler()
                    if col == 'timestamp': continue
                    df_raw[col] = scaler.fit_transform(df_raw[[col]])                        

                    self.scalers[site_id][col] = scaler
                self.ap_max[site_id] = df_raw['Active_Power'].max()

                
                df_x, df_y, time_feature, timestamp, site = self._process_file(df_raw, site_id)
                
                os.makedirs(os.path.join(self.root_path, 'scaler'), exist_ok=True)
                with open(os.path.join(self.root_path, 'scaler', f'{site_id}_scalers.pkl'), 'wb') as f:
                    pickle.dump(self.scalers[site_id], f)
                        
            # # 'ap_max' 열을 추가
            # df_x = df_x.assign(ap_max=self.ap_max[site_id])

            # # 'Active_Power' 칼럼이 마지막이 되도록 열 순서 재정렬
            # cols = [col for col in df_x.columns if col != 'Active_Power'] + ['Active_Power']
            # df_x = df_x[cols]
            # 데이터 저장
            site_data.setdefault(site_id, []).append({
                'x': df_x,
                'y': df_y,
                'data_stamp': time_feature,
                'timestamp': timestamp
            })

                # for scaling
        #     all_data.append(df_x)

        
        # combined_data = pd.concat(all_data, axis=0)
    
        self.site_data = site_data

    def _process_file(self, df_raw, site_id):
        # Time encoding ((년,) 월, 일, 요일, 시간)

        data_stamp = pd.DataFrame()
        data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
        data_stamp['year'] = data_stamp.date.apply(lambda row: row.year, 1)
        data_stamp['month'] = data_stamp.date.apply(lambda row: row.month, 1)
        data_stamp['day'] = data_stamp.date.apply(lambda row: row.day, 1)
        data_stamp['weekday'] = data_stamp.date.apply(lambda row: row.weekday(), 1)
        data_stamp['hour'] = data_stamp.date.apply(lambda row: row.hour, 1)
        if self.timeenc == 0:
            time_feature = data_stamp[['month', 'day', 'weekday', 'hour']].values
        else: 
            time_feature = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq).T
        timestamp = data_stamp[['year', 'month', 'day', 'weekday', 'hour']].values


        # 필요한 칼럼만 추출
        cols = df_raw.columns.tolist()
     
        cols.remove('timestamp')
        cols.remove('Active_Power')
        
        # df_raw = df_raw[['date'] + cols + [self.target]]
    
        df_raw = df_raw[cols + [self.target]]

        # TODO: MS, S, M 구분해서 처리하는 코드 추가

        df_x = df_raw[cols + [self.target]]
        # df_x['Wind_Speed'] = np.log
        df_y = df_raw[[self.target]]
        site_id = np.array([site_id]) * len(df_x)
       
        return df_x, df_y, time_feature, timestamp, site_id


    def _load_files(self):
        if self.data_path['type'] == 'all':
            all_files = [f for f in os.listdir(self.root_path) if f.endswith('.csv')]
            self.site_files = {}
            for file_name in all_files:
                site_id = int(file_name.split('_')[0])
                self.site_files.setdefault(site_id, []).append(file_name)
        if self.data_path['type'] == 'debug':
            self.site_files = {}
            for key in ['train', 'val', 'test']:
                site_id = int(self.data_path[key].split('-')[0])
                self.site_files.setdefault(site_id, []).append(self.data_path[key])
    
    
    def _split_sites(self):
        if self.data_path['type'] == 'all':
            # site_ids = list(self.site_files.keys())
            
            train_sites = [1, 4, 7]
            val_sites = [2, 6]
            test_sites = [3, 5]

            site_split = {
                'train': train_sites,
                'val': val_sites,
                'test': test_sites 
            }
        elif self.data_path['type'] == 'debug':
            site_split = {
                'train': [int(self.data_path['train'].split('_')[0])],
                'val': [int(self.data_path['val'].split('_')[0])],
                'test': [int(self.data_path['test'].split('_')[0])]
            }

        print(f'[INFO] {self.flag.capitalize()} sites: {site_split[self.flag]}')
        return site_split
    
    def create_sequences_indices(self):
        indices = []
        for site_id, data_list in self.site_data.items():
            for data in data_list:
                x_len = len(data['x'])
                max_start = x_len - self.seq_len - self.pred_len + 1
                indices.extend([(site_id, data, i) for i in range(max_start)])
        return indices

    
    def __getitem__(self, index):
        site_id, data, start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data['x'].iloc[s_begin:s_end].values
        seq_y = data['y'].iloc[r_begin:r_end].values
        seq_x_mark = data['data_stamp'][s_begin:s_end]
        seq_y_mark = data['data_stamp'][r_begin:r_end]
        site = np.array([site_id]).repeat(s_end - s_begin).reshape(-1, 1)
        seq_x_ds = data['timestamp'][s_begin:s_end]
        seq_y_ds = data['timestamp'][r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark, site, seq_x_ds, seq_y_ds
    
    
    def __len__(self):    
        return len(self.indices)



    # 평가 시 필요함
    # def inverse_transform(self, site, data):
    #     data_org = data.copy()
       
    #     data[-1] = self.scalers[site]['Active_Power'].inverse_transform(data[-1].reshape(-1, 1))
    #     data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
    #     return data

    # def inverse_transform(self, sites, data):
    #     # 결과를 저장할 배열 초기화
    #     # print(data.shape) (1024, 16, 1)
    #     # print(sites.shape) torch.Size([1024, 1])
    #     data_org = data.copy()
    #     inverse_data = np.zeros_like(data.squeeze())
    #     # print(sites.shape) torch.Size([1024, 256, 1])
    #     # print(data.shape) (1024, 16, 1)
    #     # 각 사이트별로 데이터의 역변환 수행
    #     for i, site_id in enumerate(sites.squeeze()):
    #         # print('site_id', site_id)
    #         # print(site_id)
    #         # print(site_id.shape) torch.Size([256])
    #         inverse_data[i] = self.scalers[site_id.item()]['Active_Power'].inverse_transform(data[i].reshape(-1, 1)).flatten()
    #     inverse_data.reshape(data_org.shape[0], data_org.shape[1], 1)
    #     # print(f'{site_id}inverse, {inverse_data}')
    #     return inverse_data
    def inverse_transform(self, sites, data):
        # 데이터 복사
        data_org = data.copy()
        inverse_data = np.zeros_like(data_org)
        
        # 각 사이트별로 데이터의 역변환 수행
        for i in range(len(sites)):
            site_id = sites[i]
            if isinstance(site_id, torch.Tensor):
                site_id = site_id.item()
            if site_id not in self.scalers:
                raise ValueError(f"Scaler for site {site_id} not found.")
            scaler = self.scalers[site_id]['Active_Power']
            reshaped_data = data[i].reshape(-1, 1)
            inverse_transformed = scaler.inverse_transform(reshaped_data)
            inverse_data[i] = inverse_transformed.reshape(data[i].shape)
        
        return inverse_data

    def inverse_transform_tensor(self, sites, data):
        data_org = data.clone()
        inverse_data = torch.zeros_like(data_org)
        
        for i in range(len(sites)):
            site_id = sites[i]
            if isinstance(site_id, torch.Tensor):
                site_id = site_id.item()
            scaler = self.scalers[site_id]['Active_Power']
            
            # 스케일러의 종류에 따라 역변환 방식 결정
            if isinstance(scaler, MinMaxScaler):
                data_min = torch.tensor(scaler.data_min_, device=data.device)
                data_max = torch.tensor(scaler.data_max_, device=data.device)
                data_range = data_max - data_min
                inverse_sample = data[i] * data_range + data_min
            elif isinstance(scaler, StandardScaler):
                mean = torch.tensor(scaler.mean_, device=data.device)
                scale = torch.tensor(scaler.scale_, device=data.device)
                inverse_sample = data[i] * scale + mean
            else:
                raise ValueError(f"Unsupported scaler type: {type(scaler)}")
            
            inverse_data[i] = inverse_sample.reshape(data[i].shape)
        
        return inverse_data

####################################################

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                features='MS', data_path='', target='Active_Power',
                scale=True, timeenc=0, freq='h', scaler='MinMaxScaler'):
    # size [seq_len, label_len, pred_len]
    # info
        print("start")
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
        self.flag = flag


        self.scaler = scaler
        

        self.LOCATIONS = [
            file_name for file_name in os.listdir(self.root_path) if file_name.endswith('.csv')
        ]
        
        self.scalers = {}
        self.x_list = []
        self.y_list = []
        self.ds_list = []

        
        random.seed(42)
        if self.data_path['type'] == 'all':
            print("all")
            random.shuffle(self.LOCATIONS)  # 데이터 섞기
            total_size = len(self.LOCATIONS)
            train_size = int(0.6 * total_size)
            val_size = int(0.3 * total_size)
                    
            if self.flag == 'train':
                self.train = self.LOCATIONS[:train_size]
                print(f'[INFO] Train Locations: total {len(self.train)}')
                print(f'[INFO] Location List: {self.train}')
                
            elif self.flag == 'val':
                self.val = self.LOCATIONS[train_size:train_size + val_size]
                print(f'[INFO] Valid Locations: total {len(self.val)}')
                print(f'[INFO] Location List: {self.val}')

            elif self.flag == 'test':
                self.test = self.LOCATIONS[train_size + val_size:]
                print(f'[INFO] Test Locations: total {len(self.test)}')
                print(f'[INFO] Location List: {self.test}')



        elif self.data_path['type'] == 'debug':
            
            if self.flag == 'train':
                self.train = self.data_path['train']
            
            elif self.flag == 'val':
                self.val = self.data_path['val']
            
            elif self.flag == 'test':
                self.test = self.data_path['test']
            

        

        self.load_preprocessed_data() 
        


    def load_preprocessed_data(self):

        
        # 1. Flag에 맞게 데이터 불러오기
        if self.flag == 'train':
            df_raw = self.load_and_concat_data(self.train)
        
        elif self.flag == 'val':
            df_raw = self.load_and_concat_data(self.val)

        elif self.flag == 'test':
            df_raw = self.load_and_concat_data(self.test)

        # 칼럼 순서와 이름 확인
        print(f"[INFO] Loaded columns: {df_raw.columns}")

        # 2. Time encoding (년, 월, 일, 요일, 시간)
        if self.timeenc == 0:
            data_stamp = pd.DataFrame()
            data_stamp['date'] = pd.to_datetime(df_raw.timestamp)
            data_stamp['year'] = df_raw.date.apply(lambda row: row.year, 1)
            data_stamp['month'] = df_raw.timestamp.apply(lambda row: row.month, 1)
            data_stamp['day'] = df_raw.date.apply(lambda row: row.day, 1)
            data_stamp['weekday'] = df_raw.date.apply(lambda row: row.weekday(), 1)
            data_stamp['hour'] = df_raw.date.apply(lambda row: row.hour, 1)
            
            data_stamp = data_stamp.drop(['date'], axis=1).values()

        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_raw['timestamp'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # 3. Encoding 후, date columns 생성
        # df_raw['date'] = data_stamp

      
        # 4. Columns 순서 정렬 (timestamp, date, ..... , target)
        cols = df_raw.columns.tolist()
        # cols.remove('date')
        cols.remove('timestamp')
        cols.remove('Active_Power')
        df_raw = df_raw[cols + [self.target]]

        if self.scale: 
            # Train일 때는, Scaler Fit 후에, 저장
            if self.flag == 'train' or self.flag == 'val' or self.flag == 'test':
                for col in df_raw.columns:
                    scaler = StandardScaler()
                    df_raw[col] = scaler.fit_transform(df_raw[[col]])
                    self.scalers[col] = scaler 
                    # Scaler를 pickle 파일로 저장
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
                    with open(path, 'wb') as f:
                        pickle.dump(scaler, f)
        
            else:
                # Val, Test일 때는, 저장된 Scaler 불러와서 적용
                self.scalers = {}
                transformed_df = df_raw.copy()  

                for col in df_raw.columns:
                    path = os.path.join(self.root_path, f'{col}_scaler.pkl')
            
                    # Scaler가 존재하는지 확인
                    if os.path.exists(path):
                        with open(path, 'rb') as f:
                            scaler = pickle.load(f) 
                
                        # 해당 칼럼에 스케일러 적용 (transform)
                        df_raw[col] = scaler.transform(transformed_df[[col]]) 
                        self.scalers[col] = scaler
                
                    else:
                        print(f"Scaler for column {col} not found.")


        # 6. 입력할 칼럼들 지정하여 리스트 생성
        if self.features == 'M' or self.features == 'MS':
            # date 열을 제외
            cols_data = df_raw.columns[1:]
            df_x = df_raw[cols_data]
        elif self.features == 'S':
            # Active Power만 추출
            df_x = df_raw[[self.target]]
        

        self.x_list = df_x.values
        # 타겟은 마지막 열인 Active_Power
        self.y_list = df_raw[[self.target]].values
        # date columns만 선택
        self.ds_list = data_stamp      
        
    # 파일 경로를 가져와서 DataFrame으로 합치는 함수
    def load_and_concat_data(self, file_list):
        df_list = []
        for file in file_list:
            file_path = f"{self.root_path}/{file}"  
            df = pd.read_csv(file_path)  
            assert (df.isnull().sum()).sum() == 0, "허용되지 않은 열에 결측치가 존재합니다."            
            df_list.append(df) 
        return pd.concat(df_list, ignore_index=True) 

    def __getitem__(self, index):
        s_begin = index * (self.seq_len + self.label_len + self.pred_len)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.x_list[s_begin:s_end]
        seq_y = self.y_list[r_begin:r_end]
        seq_x_mark = self.ds_list[s_begin:s_end]
        seq_y_mark = self.ds_list[r_begin:r_end]

        return seq_x, seq_y.reshape(-1, 1), seq_x_mark, seq_y_mark


    def __len__(self):
        return (len(self.x_list) - self.seq_len - self.label_len - self.pred_len + 1) // (self.seq_len + self.label_len + self.pred_len)



    # 평가 시 필요함
    def inverse_transform(self, data):
        data_org = data.copy()
        
        data = self.scalers['Active_Power'].inverse_transform(data.reshape(-1, 1))
        data = data.reshape(data_org.shape[0], data_org.shape[1], -1)
        return data


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


class Dataset_SineMax(Dataset):
    def __init__(self, root_path=None, flag='train', size=None,
                 features='S', data_path='', scaler=None,
                 target='Sine', scale=False, timeenc=0, freq='h'):
        # 기본적인 정보 초기화
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.scaler = scaler

        # sine 파형 생성
        total_len = 10000  # 충분히 긴 데이터 생성
        time_steps = np.linspace(0, 2 * np.pi * (total_len / 24), total_len)  # 24시간 주기
        self.y_data = np.maximum(0, np.sin(time_steps)).reshape(-1, 1)

        # 정규화 (필요한 경우)
        if self.scale and self.scaler is not None:
            self.scaler.fit(self.y_data)  # y_data의 스케일 조정
            self.y_data = self.scaler.transform(self.y_data)

        # 더미 데이터 생성
        self.data_stamp = np.tile(np.arange(total_len).reshape(-1, 1), (1, 4))  # time encoding
        self.site = np.zeros((total_len, 1))  # 단일 사이트

        # 시퀀스 인덱스 생성
        self.indices = self.create_sequences_indices()

    def create_sequences_indices(self):
        max_start = len(self.y_data) - self.seq_len - self.pred_len + 1
        return list(range(max_start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_idx = self.indices[index]
        s_begin = start_idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.y_data[s_begin:s_end]
        seq_y = self.y_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        site = self.site[s_begin:s_end]
        seq_x_ds = self.data_stamp[s_begin:s_end]
        seq_y_ds = self.data_stamp[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
            torch.tensor(site, dtype=torch.float32),
            torch.tensor(seq_x_ds, dtype=torch.float32),
            torch.tensor(seq_y_ds, dtype=torch.float32),
        )

    def inverse_transform(self, data):
        if self.scaler is not None and self.scale:
            return self.scaler.inverse_transform(data)
        return data  # 스케일링이 적용되지 않았다면 그대로 반환
    


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