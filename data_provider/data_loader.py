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
from datetime import datetime

warnings.filterwarnings('ignore')

class Dataset_DKASC(Dataset):
    def __init__(self,
                 root_path, data_path=None,
                 split_configs=None, flag='train',
                 size=None, timeenc=0,
                 freq='h', scaler=True,
                ):
        """
        이 예시는 installation 단위로 데이터가 나뉘어 있고,
        split_configs로 train, val, test 설정을 받아 데이터를 분할합니다.

        Args:
            root_path (str): 데이터 파일들이 저장된 경로.
            data_path (str, optional): 추가적인 데이터 경로. 단일 PV array 데이터를 가져오는 경우 사용.
            split_configs (dict): train, val, test 설정.
            flag (str): 'train', 'val', 'test' 중 하나.
            size (tuple, optional): seq_len, label_len, pred_len의 길이를 포함.
            timeenc (int): 시간 인코딩 방법.
            freq (str): 시간 데이터 빈도.
            scaler (bool): 스케일링 여부.
        """

        if size is None:
            raise ValueError("size cannot be None. Please specify seq_len, label_len, and pred_len explicitly.")
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test'], "flag must be 'train', 'val', or 'test'."

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.timeenc = timeenc
        self.freq = freq
        self.scaler = scaler
        self.split_configs = split_configs

        # 입력 채널 정의
        self.input_channels = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                               'Weather_Relative_Humidity', 'Wind_Speed', 'Active_Power']

        # mapping 파일 로드
        mapping_df = pd.read_csv('./data_provider/dataset_name_mappings.csv')
        dataset_name = self.__class__.__name__.split('_')[-1]  # 클래스 이름에서 데이터셋 이름 추출
        self.current_dataset = mapping_df[mapping_df['dataset'] == dataset_name]
        self.current_dataset['index'] = self.current_dataset['mapping_name'].apply(lambda x: int(x.split('_')[0]))  # index 열 추가

        # flag에 따른 installation 리스트 설정 (train, val, test)
        self.inst_list = self.split_configs[flag]

        # 스케일러 저장 경로
        self.scaler_dir = os.path.join(root_path, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)

        # 데이터를 저장할 리스트
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.inst_id_list = []
        self.capacity_info = {}

        # 데이터 준비 및 indices 생성
        self._prepare_data()
        self.indices = self._create_indices()

    def _prepare_data(self):
        for inst_id in self.inst_list:
            # inst_id를 기반으로 파일 이름 가져오기
            file_row = self.current_dataset[self.current_dataset['index'] == inst_id]
            if file_row.empty:
                raise ValueError(f"No matching file found for inst_id {inst_id} in dataset {current_dataset}.")
            
            file_name = file_row['original_name'].values[0]
            # 파일명과 capacity 정보 추출
            file_name = file_row['original_name'].values[0]
            try:
                capacity = float(file_name.split('_')[0])
                self.capacity_info[inst_id] = capacity
                self.inst_id_list.append(inst_id)
            except (IndexError, ValueError):
                raise ValueError(f"Invalid capacity format in filename: {file_name}")
            
            # 데이터 로드 및 전처리
            csv_path = os.path.join(self.root_path, file_name)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Data file not found: {csv_path}")
            
            # read dataset
            df_raw = pd.read_csv(csv_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='coerce')

            # 필요한 컬럼만 추출
            df_raw = df_raw[['timestamp'] + self.input_channels]

            # 시간 피처 생성
            df_stamp = pd.DataFrame()
            df_stamp['timestamp'] = df_raw['timestamp']
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['timestamp'].dt.month
                df_stamp['day'] = df_stamp['timestamp'].dt.day
                df_stamp['weekday'] = df_stamp['timestamp'].dt.weekday
                df_stamp['hour'] = df_stamp['timestamp'].dt.hour
                data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
            else:
                data_stamp = time_features(df_stamp['timestamp'], freq=self.freq).transpose(1, 0)

            df_data = df_raw[self.input_channels]
            scaler_path = os.path.join(self.scaler_dir, f"{file_name}_scaler.pkl")

            if self.scaler:
                # 스케일러 fit & transform 로직
                if not os.path.exists(scaler_path):
                    scaler_dict = {}
                    for ch in self.input_channels:
                        scaler = StandardScaler()
                        scaler.fit(df_data[[ch]])
                        scaler_dict[ch] = scaler
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler_dict, f)
                else:
                    with open(scaler_path, 'rb') as f:
                        scaler_dict = pickle.load(f)

                transformed_data = [scaler_dict[ch].transform(df_data[[ch]]) for ch in self.input_channels]
                data = np.hstack(transformed_data)
            else:
                data = df_data.values

            self.data_x_list.append(data)
            self.data_y_list.append(data)
            self.data_stamp_list.append(data_stamp)

    def _create_indices(self):
        indices = []
        for inst_idx, data_x in enumerate(self.data_x_list):
            total_len = len(data_x)
            max_start = total_len - self.seq_len - self.pred_len + 1
            for s in range(max_start):
                indices.append((inst_idx, s))
        return indices

    def __getitem__(self, index):
        inst_idx, s_begin = self.indices[index]
        inst_id = self.inst_id_list[inst_idx]

        data_x = self.data_x_list[inst_idx]
        data_y = self.data_y_list[inst_idx]
        data_stamp = self.data_stamp_list[inst_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, inst_id

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data, inst_ids):
        """
        스케일링된 데이터를 원래 스케일로 변환
        
        Args:
            inst_ids: 배치 내 각 데이터의 installation ID (배치 크기만큼의 길이)
            data: 변환할 데이터 (batch_size, seq_len, feature_dim)
        Returns:
            inverse_data: 역변환된 데이터 (입력과 같은 shape)
        """
        if not self.scaler:
            return data
            
        data_org = data.copy()
        inverse_data = np.zeros_like(data_org)
        
        # unique한 installation IDs 추출
        unique_inst_ids = np.unique(inst_ids)
        
        # 각 unique installation에 대해
        for inst_id in unique_inst_ids:
            # 현재 installation의 스케일러 로드
            file_row = self.current_dataset[self.current_dataset['index'] == inst_id]
            file_name = file_row['original_name'].values[0]
            scaler_path = os.path.join(self.scaler_dir, f"{file_name}_scaler.pkl")
            
            with open(scaler_path, 'rb') as f:
                scaler_dict = pickle.load(f)
                
            # 현재 installation에 해당하는 데이터의 인덱스 찾기
            inst_mask = (inst_ids == inst_id)
            
            # 해당하는 데이터만 추출하여 역변환
            inst_data = data_org[inst_mask].reshape(-1, 1)
            inverse_inst_data = scaler_dict['Active_Power'].inverse_transform(inst_data)
            
            # 역변환된 데이터를 원래 위치에 복원
            inverse_data[inst_mask] = inverse_inst_data.reshape(data_org[inst_mask].shape)
        
        return inverse_data



        # scaler_path = os.path.join(self.scaler_dir, f"{inst_id}_scaler.pkl")

        # if not os.path.exists(scaler_path):
        #     raise FileNotFoundError(f"Scaler for installation {inst_id} not found.")
        # with open(scaler_path, 'rb') as f:
        #     scaler_dict = pickle.load(f)

        # return scaler_dict[self.target].inverse_transform(data)


########################################################################################

class Dataset_GIST(Dataset):
    def __init__(self, root_path, data_path = None,
                 flag='train', size=None,
                 features='MS', target='Active_Power', timeenc=0, freq='h', domain='target',
                 scaler=True,
                 train_inst=[1,2],
                 val_inst=[3],
                 test_inst=[4],
                 input_channels=['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                                 'Weather_Relative_Humidity', 'Wind_Speed'],
                 data=None):
        """
        이 예시는 installation 단위로 데이터가 나뉘어 있고,
        train_inst, val_inst, test_inst로 어떤 installation이 어느 단계에 속하는지 알려줍니다.

        원칙:
        - 각 installation은 자신이 속한 단계에서(Train/Val/Test) scaler를 fit & save (처음 처리 시)하고 transform 수행.
        - 이후 동일 installation에 대한 재처리 시에는 이미 저장된 scaler를 로드하여 transform만 할 수 있음.
        """

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size
        
        assert flag in ['train', 'val', 'test'], "flag must be 'train', 'val', or 'test'."
        self.flag = flag
        self.features = features
        self.target = target
        self.scaler = scaler
        self.timeenc = timeenc
        self.freq = freq
        self.domain = domain
        self.root_path = root_path
        self.input_channels = input_channels + [target]

        # flag에 따라 어떤 installation을 처리할지 결정
        if self.flag == 'train':
            self.inst_list = train_inst
        elif self.flag == 'val':
            self.inst_list = val_inst
        else:
            self.inst_list = test_inst

        # scaler를 저장할 디렉토리
        self.scaler_dir = os.path.join(root_path, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)

        # 데이터를 저장할 리스트
        self.data_x_list = []
        self.data_y_list = []
        self.data_stamp_list = []
        self.inst_id_list = []
        
        # 데이터 준비 및 indices 생성
        self._prepare_data()
        self.indices = self._create_indices()

    def _prepare_data(self):
        for inst_id in self.inst_list:
            csv_file = f"installation_{inst_id}.csv"
            csv_path = os.path.join(self.root_path, csv_file)
            df_raw = pd.read_csv(csv_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], errors='raise')

            # 필요한 컬럼만
            df_raw = df_raw[['timestamp'] + self.input_channels]

            # 시간 피처 생성
            df_stamp = pd.DataFrame()
            df_stamp['timestamp'] = df_raw['timestamp']
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['timestamp'].dt.month
                df_stamp['day'] = df_stamp['timestamp'].dt.day
                df_stamp['weekday'] = df_stamp['timestamp'].dt.weekday
                df_stamp['hour'] = df_stamp['timestamp'].dt.hour
                data_stamp = df_stamp[['month','day','weekday','hour']].values
            else:
                data_stamp = time_features(df_stamp['timestamp'], freq=self.freq).transpose(1,0)

            df_data = df_raw[self.input_channels]

            scaler_path = os.path.join(self.scaler_dir, f"installation_{inst_id}_scaler.pkl")

            if self.scaler:
                # 스케일러 fit & transform 로직
                # 만약 이 installation의 스케일러가 존재하지 않는다면 새로 fit
                if not os.path.exists(scaler_path):
                    # 여기서가 이 installation을 처음 처리하는 시점
                    # 이 시점이 Train/Val/Test 중 어느 단계든 상관 없이, 해당 installation은 자신의 전체 데이터를 사용해 fit 가능
                    scaler_dict = {}
                    for ch in self.input_channels:
                        scaler = StandardScaler()
                        # 여기서는 installation 전체 데이터 사용 (현재 code에서는 filtering 없음)
                        scaler.fit(df_data[[ch]])
                        scaler_dict[ch] = scaler
                    # fit한 스케일러 저장
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler_dict, f)
                else:
                    # 이미 scaler가 있다면 로드
                    with open(scaler_path, 'rb') as f:
                        scaler_dict = pickle.load(f)

                # transform 수행
                transformed_data = [scaler_dict[ch].transform(df_data[[ch]]) for ch in self.input_channels]
                data = np.hstack(transformed_data)
            else:
                data = df_data.values

            self.data_x_list.append(data)
            self.data_y_list.append(data)
            self.data_stamp_list.append(data_stamp)
            self.inst_id_list.append(inst_id)

    def _create_indices(self):
        indices = []
        for inst_idx, data_x in enumerate(self.data_x_list):
            total_len = len(data_x)
            max_start = total_len - self.seq_len - self.pred_len + 1
            for s in range(max_start):
                indices.append((inst_idx, s))
        return indices

    def __getitem__(self, index):
        inst_idx, s_begin = self.indices[index]
        data_x = self.data_x_list[inst_idx]
        data_y = self.data_y_list[inst_idx]
        data_stamp = self.data_stamp_list[inst_idx]

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.indices)

    def inverse_transform(self, data, inst_id):
        scaler_path = os.path.join(self.scaler_dir, f"installation_{inst_id}_scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler for installation {inst_id} not found.")
        with open(scaler_path, 'rb') as f:
            scaler_dict = pickle.load(f)

        return scaler_dict[self.target].inverse_transform(data)









#######################################################################################

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