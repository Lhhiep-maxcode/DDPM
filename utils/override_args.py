def override_config(config, args, train=False):
    # override params for dataset
    if train:
        if args.root_dir:
            config['dataset_params']['root'] = args.root_dir
        if args.train_dir:
            config['dataset_params']['train'] = args.train_dir
        if args.test_dir:
            config['dataset_params']['test'] = args.test_dir

    # override params for diffusion
    if args.num_timesteps:
        config['diffusion_params']['num_timesteps'] = args.num_timesteps
    if args.beta_start:
        config['diffusion_params']['beta_start'] = args.beta_start
    if args.beta_end:
        config['diffusion_params']['beta_end'] = args.beta_end
        
    # override params for model
    if args.im_channels:
        config['model_params']['im_channels'] = args.im_channels
    if args.im_size:
        config['model_params']['im_size'] = args.im_size
    if args.time_emb_dim:
        config['model_params']['time_emb_dim'] = args.time_emb_dim
    if args.num_heads:
        config['model_params']['num_heads'] = args.num_heads
    if args.dropout:
        config['model_params']['dropout'] = args.dropout

    # override params for training
    if train:
        if args.batch_size:
            config['train_params']['batch_size'] = args.batch_size
        if args.num_epochs:
            config['train_params']['num_epochs'] = args.num_epochs
        if args.lr:
            config['train_params']['lr'] = args.lr
        if args.task_name:
            config['train_params']['task_name'] = args.task_name
        if args.saved_ckpt_name:
            config['train_params']['saved_ckpt_name'] = args.saved_ckpt_name
        if args.load_ckpt_path:
            config['train_params']['load_ckpt_path'] = args.load_ckpt_path
        if args.save_every:
            config['train_params']['save_every'] = args.save_every

    # override params for inference
    if not train:
        if args.num_samples:
            config['infer_params']['num_samples'] = args.num_samples
        if args.num_grid_rows:
            config['infer_params']['num_grid_rows'] = args.num_grid_rows
        if args.load_ckpt_path:
            config['infer_params']['load_ckpt_path'] = args.load_ckpt_path
        if args.task_name:
            config['infer_params']['task_name'] = args.task_name
        if args.video_name:
            config['infer_params']['video_name'] = args.video
    return config