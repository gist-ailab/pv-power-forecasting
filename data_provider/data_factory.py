# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
#      Dataset_DKASC_AliceSprings, Dataset_DKASC_Yulara, Dataset_GIST, Dataset_German, Dataset_UK, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_Miryang, Dataset_Miryang_MinMax, Dataset_Miryang_Standard, Dataset_SineMax
from data_provider.data_loader import Dataset_DKASC, Dataset_GIST, Dataset_Miryang, Dataset_Germany, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_UK, Dataset_SineMax, Dataset_GISTchrono, Dataset_Germanychrono, Dataset_GIST_Spring, Dataset_GIST_Summer, Dataset_GIST_Autumn, Dataset_GIST_Winter
from torch.utils.data import DataLoader, ConcatDataset
import torch

data_dict = {
    'DKASC' : Dataset_DKASC,
    'SineMax': Dataset_SineMax,
    'GIST': Dataset_GIST,
    'Miryang': Dataset_Miryang,
    'Germany': Dataset_Germany,
    'OEDI_Georgia': Dataset_OEDI_Georgia,
    'OEDI_California': Dataset_OEDI_California,
    'UK': Dataset_UK,
    'GIST_Spring': Dataset_GIST_Spring,
    'GIST_Summer': Dataset_GIST_Summer,
    'GIST_Autumn': Dataset_GIST_Autumn,
    'GIST_Winter': Dataset_GIST_Winter,
    'GISTchrono': Dataset_GISTchrono,
    'Germanychrono': Dataset_Germanychrono,
}

split_configs = {
    'DKASC': {
        'train': [1, 3, 5, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                  33, 34, 35, 36, 37, 39, 41, 42],
        'val': [2, 8, 14, 32, 38],   # 2
        'test': [4, 6, 31, 40, 43]  # 4, 43
    },
    # 사이트로 나누는 loc
    'DKASC_AliceSprings': {
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
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },
    'Germany': {
        'train': [2, 3, 4, 5, 6, 7, 8],
        'val': [9],
        'test': [1]
    },
    'Miryang': {
        'train': [1, 2, 3, 5, 7],
        'val': [6],
        'test': [4]
    },
    'GIST_Spring': {
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },
    'GIST_Summer': {
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },
    'GIST_Autumn': {
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },
    'GIST_Winter': {
        'train': [1, 4, 5, 6, 7, 8, 9, 10, 11, 13],
        'val': [2, 12],
        'test': [3, 14]
    },

    #### 날짜로 나누는 loc

    'OEDI_California': { # 2018.01.23. ~ 2023.10.31
        'train' : 0.75, # 180123 ~ 220522   => 4 years and 4 months
    'val' : 0.1,        # 220523 ~ 221218   => 7 months
        'test' : 0.15   # 221219 ~ 231031   => 10 months
    },
    'OEDI_Georgia' : {  # 2018.03.29. ~ 2022.03.10.
        'train' : 0.75, # 180329 ~ 210314   => 3 years
        'val' : 0.1,    # 210315 ~ 210805   => 5 months
        'test' : 0.15   # 210806 ~ 220310   => 7 months
    },
    'UK' : {
        'train' : 0.75,
        'val' : 0.1,
        'test' : 0.15
    },

    # ablation study for chronological on GIST dataset
    'GISTchrono' : {
        'train': [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'val': [3, 14], # 3: E03_GTI, 13: C10_Renewable-E-Bldg
        'test': [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    },
    # ablation study for chronological on Germany dataset
    'Germanychrono': {
        'train': [1, 3, 4, 5, 7, 8, 9],
        'val': [2, 6],  # 2: DE_KN_industrial2_pv, 6: DE_KN_industrial3_pv_facade
        'test': [10, 11, 12, 13, 14, 15, 16, 17, 18]
    },
}


def data_provider(args, flag, distributed=False):
    ## flag : train, val, test
    Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        # Data = Dataset_DKASC_AliceSprings
    else: # train, val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        data_type=args.data_type,
        split_configs=split_configs[args.data],
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq,
        scaler=args.scaler
        )
    print(flag, data_set.__len__())
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_set,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=shuffle_flag
        )
        shuffle_flag = False  # When using a sampler, DataLoader shuffle must be False
    else:
        sampler = None

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True,
        sampler=sampler)
    return data_set, data_loader
