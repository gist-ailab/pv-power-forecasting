
    #####  debug  #####
    args.gpu = 0
    args.random_seed = 2021
    args.is_training = 1
    
    ## debug DKASC
    args.root_path = './dataset/pv/'
    args.data_path = '91-Site_DKA-M9_B-Phase.csv'
    args.data = 'pv_DKASC'
    args.enc_in = 4
    args.dec_in = 4
    args.c_out = 4
    
    # ## debug GIST
    # args.root_path = './dataset/GIST/'
    # args.data_path = 'sisuldong.csv'
    # args.data = 'pv_GIST'
    # args.enc_in = 4
    # args.dec_in = 4
    # args.c_out = 4
    
    # ## debug other datasets
    # args.root_path = './dataset/'
    # args.data_path = 'weather.csv'      # 'weather.csv', 'ETTh1.csv'
    # args.data = 'custom'                # 'custom', 'ETTh1'
    # args.enc_in = 21
    # args.dec_in = 21
    # args.c_out = 21
    
    
    args.embed = 'timeF'    # 'timeF', 'fixed'
    args.model_id = 'debug'
    args.features = 'M'
    args.seq_len = 96       # 336
    args.label_len = 48
    args.pred_len = 96      # 96
    
    args.model = 'Transformer'
    args.seq_len = 96       # 336
    args.label_len = 48
    args.pred_len = 96      # 96

    args.e_layers = 2       # 3
    args.d_layers = 1
    args.factor = 3
        

    args.n_heads = 16
    args.d_model = 128
    args.d_ff = 256
    args.dropout = 0.2
    args.fc_dropout = 0.2
    args.head_dropout = 0
    args.patch_len = 16
    args.stride = 8
    args.des = 'Exp'
    args.train_epochs = 100
    args.patience = 20
    args.itr = 1
    args.batch_size = 128
    args.learning_rate = 0.00001
