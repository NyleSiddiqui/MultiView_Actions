import json

def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'ntu_rgbd_120':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.train_annotations = 'data/NTUTrain_CVmap.csv'
        cfg.test_annotations = 'data/NTUTest_CVmap.csv'
        cfg.num_actions = 120
        
    elif dataset == 'ntu_rgbd_60':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.train_annotations = 'data/NTU60Train_CSmap.csv'
        cfg.test_annotations = 'data/NTU60Test_CSmap.csv'
        cfg.num_actions = 60
        
    elif dataset == "pkummd":
        cfg.videos_folder =  '/home/c3-0/datasets/PKUMMD/RGB_VIDEO'
        cfg.train_annotations = '/data/PKUMMDTrainCS_map.csv'
        cfg.test_annotations = 'data/PKUMMDTestCS_map.csv'
        cfg.num_actions = 51

    elif dataset == 'numa':
        cfg.videos_folder =  '/home/c3-0/datasets/NUMA/multuview_action_videos'
        cfg.train_annotations = "data/NUMATrain_CV.csv"
        cfg.test_annotations = "data/NUMATest_CV.csv"
        cfg.num_actions = 10
        
    else:
        raise NotImplementedError
        
    cfg.dataset = dataset
    cfg.saved_models_dir = './results/saved_models'
    cfg.outputs_folder = './results/outputs'
    cfg.tf_logs_dir = './results/logs'
    return cfg
