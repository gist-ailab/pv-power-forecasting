from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
     Dataset_DKASC_AliceSprings, Dataset_DKASC_Yulara, Dataset_GIST, Dataset_German, Dataset_UK, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_Miryang, Dataset_Miryang_MinMax, Dataset_Miryang_Standard, Dataset_SineMax
from torch.utils.data import DataLoader, ConcatDataset

data_dict = {
    'DKASC_AliceSprings': Dataset_DKASC_AliceSprings,
    'DKASC_Yulara': Dataset_DKASC_Yulara,
    'DKASC_Source' : 'DKASC_Source',
    'GIST': Dataset_GIST,
    'Germany': Dataset_German,
    'UK': Dataset_UK,
    'OEDI_Georgia': Dataset_OEDI_Georgia,
    'OEDI_California': Dataset_OEDI_California,
    'Miryang': Dataset_Miryang,
    'Miryang_MinMax': Dataset_Miryang_MinMax,
    'Miryang_Standard': Dataset_Miryang_Standard,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'SineMax': Dataset_SineMax,
    'custom': Dataset_Custom,
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
        Data = Dataset_DKASC_AliceSprings
    else: # train, val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    def create_dataset(data_class, root_path):
        return data_class(
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
    
    if args.data == 'DKASC_Source':
        root_path_1, root_path_2 = args.root_path.split(',')

        alice_springs = create_dataset(Dataset_DKASC_AliceSprings, root_path_1)
        yulara = create_dataset(Dataset_DKASC_Yulara, root_path_2)
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
