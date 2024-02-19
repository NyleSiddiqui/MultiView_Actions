import json

def build_config(dataset):
    cfg = type('', (), {})()
    if dataset == 'ntu_rgbd_120':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.train_annotations = 'data/NTUTrain_CVmap.csv'
        cfg.test_annotations = 'data/NTUTest_CVmap.csv'
        cfg.train_subjects = range(80) #CS: 53, CV: 80
        cfg.test_subjects = range(69)  #CS: 53, CV: 69
        cfg.num_actions = 120
        cfg.num_subjects = 106
        
    elif dataset == 'ntu_rgbd_60':
        cfg.videos_folder =  '/home/c3-0/datasets/NTU_RGBD_120/nturgb+d_rgb'
        cfg.train_annotations = 'data/NTU60Train_CSmap.csv'
        cfg.test_annotations = 'data/NTU60Test_CSmap.csv'
        cfg.train_subjects = range(20) #CS: 20, CV: 40
        cfg.test_subjects = range(20)  #CS: 20, CV: 40
        cfg.num_actions = 60
        cfg.num_subjects = 40
        
    elif dataset == "pkummd":
        cfg.videos_folder =  '/home/c3-0/datasets/PKUMMD/RGB_VIDEO'
        cfg.train_annotations = '/data/PKUMMDTrainCS_map.csv'
        cfg.test_annotations = 'data/PKUMMDTestCS_map.csv'
        cfg.train_subjects = range(56) #CS: 56 CV: 66
        cfg.test_subjects = range(7)  #CS:  7 CV: 63
        cfg.num_subjects = 66
        cfg.num_actions = 51

    elif dataset == 'numa':
        cfg.videos_folder =  '/home/c3-0/datasets/NUMA/multuview_action_videos'
        cfg.train_annotations = "data/NUMATrain_CV.csv"
        cfg.test_annotations = "data/NUMATest_CV.csv"
        cfg.train_subjects = range(10) #CS: 9, CV: 10
        cfg.test_subjects = range(9)  #CS: 1, CV: 9 (sub 5 has no viewpoint 3 for some reason
        cfg.num_actions = 10
        cfg.num_subjects = 10
        
    else:
        raise NotImplementedError
        
    cfg.dataset = dataset
    cfg.saved_models_dir = './results/saved_models'
    cfg.outputs_folder = './results/outputs'
    cfg.tf_logs_dir = './results/logs'
    return cfg
