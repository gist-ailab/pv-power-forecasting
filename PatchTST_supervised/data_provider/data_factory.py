from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
     Dataset_pv_DKASC, Dataset_pv_DKASC_multi,Dataset_pv_SolarDB, Dataset_pv_GIST, CrossDomain_Dataset
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'pv_DKASC': Dataset_pv_DKASC,
    'pv_DKASC_multi': Dataset_pv_DKASC_multi,
    'pv_SolarDB': Dataset_pv_SolarDB,
    'pv_GIST': Dataset_pv_GIST,
    'CrossDomain': CrossDomain_Dataset,
}


def data_provider(args, flag):
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
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if 'CDTST' in args.model:
        data_set = Data(
            source_root_path=args.source_root_path,
            target_root_path=args.target_root_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            source_data_path=args.source_data_path,
            target_data_path=args.target_data_path,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    print(flag, data_set.__len__())
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
