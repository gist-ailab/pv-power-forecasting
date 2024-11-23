# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
#      Dataset_DKASC_AliceSprings, Dataset_DKASC_Yulara, Dataset_GIST, Dataset_German, Dataset_UK, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_Miryang, Dataset_Miryang_MinMax, Dataset_Miryang_Standard, Dataset_SineMax
from data.provider.data_loader import Dataset_PV, Dataset_SineMax
from torch.utils.data import DataLoader, ConcatDataset

data_dict = {
    'Source' : Dataset_PV,
    'SineMax': Dataset_SineMax,
    'DKASC_AliceSprings': Dataset_PV,
    'DKASC_Yulara': Dataset_PV,
    'GIST': Dataset_PV,
    'German': Dataset_PV,
    'UK': Dataset_PV,
    'OEDI_Georgia': Dataset_PV,
    'OEDI_California': Dataset_PV,
    'Miryang': Dataset_PV,
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
        # Data = Dataset_DKASC_AliceSprings
    else: # train, val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    def create_dataset(Data, root_path):
        return Data(
            root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            scaler=args.scaler,
        )
    
    if args.data == 'Source':
        root_path_1, root_path_2 = args.root_path.split(',')

        alice_springs = create_dataset(Data, root_path_1)
        yulara = create_dataset(Data, root_path_2)
        source_dataset = ConcatDataset([alice_springs, yulara])

        print(f"{flag} - Alice Springs length: {alice_springs.__len__()}")
        print(f"{flag} - Yulara length: {yulara.__len__()}")
        print(f"{flag} - Combined length: {source_dataset.__len__()}")
        
        data_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True
        )
        return source_dataset, data_loader
    
    else:
        data_set = create_dataset(Data, args.root_path)
        print(flag, data_set.__len__())
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,)
        return data_set, data_loader
