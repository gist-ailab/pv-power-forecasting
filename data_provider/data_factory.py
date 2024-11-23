# from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, \
#      Dataset_DKASC_AliceSprings, Dataset_DKASC_Yulara, Dataset_GIST, Dataset_German, Dataset_UK, Dataset_OEDI_Georgia, Dataset_OEDI_California, Dataset_Miryang, Dataset_Miryang_MinMax, Dataset_Miryang_Standard, Dataset_SineMax
from data_provider.data_loader import Dataset_PV, Dataset_SineMax
from torch.utils.data import DataLoader, ConcatDataset
import torch

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


def data_provider(args, flag, distributed=False):
    ## flag : train, val, test
    Data = data_dict[args.data[0]]
    # Data = data_dict[args.data]
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
    
    




    def create_dataset(Data, root_path, data):
        return Data(
            root_path=root_path,
            data_path=args.data_path,
            data=data,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq
        )
    
    if args.data[0] == 'Source':
        dataset_list = []
        for data, root_path in zip(args.data[1:], args.root_path):
            data = create_dataset(Data, root_path, data)
            print(f"{flag} - {root_path} length: {data.__len__()}")
            dataset_list.append(data)
        source_dataset = ConcatDataset(dataset_list)

        print(f"{flag} - Combined length: {source_dataset.__len__()}")
        
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
            source_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=True,
            sampler=sampler
        )
        return source_dataset, data_loader
    
    else:
        data_set = create_dataset(Data, args.root_path[0], args.data[0])
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
            sampler=None)
        return data_set, data_loader
