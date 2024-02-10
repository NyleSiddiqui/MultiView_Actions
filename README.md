<a href='https://nylesiddiqui.github.io/DVANet_webpage'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2312.05719'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvanet-disentangling-view-and-action-features/action-recognition-on-n-ucla)](https://paperswithcode.com/sota/action-recognition-on-n-ucla?p=dvanet-disentangling-view-and-action-features)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvanet-disentangling-view-and-action-features/action-recognition-in-videos-on-pku-mmd)](https://paperswithcode.com/sota/action-recognition-in-videos-on-pku-mmd?p=dvanet-disentangling-view-and-action-features)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvanet-disentangling-view-and-action-features/action-recognition-in-videos-on-ntu-rgbd-120)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ntu-rgbd-120?p=dvanet-disentangling-view-and-action-features)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvanet-disentangling-view-and-action-features/action-recognition-in-videos-on-ntu-rgbd)](https://paperswithcode.com/sota/action-recognition-in-videos-on-ntu-rgbd?p=dvanet-disentangling-view-and-action-features)

# DVANet: Disentangling View and Action Features for Multi-View Action Recognition [AAAI 2024]
[Nyle Siddiqui](https://nylesiddiqui.github.io/), [Praveen Tirupattur](https://scholar.google.com/citations?user=zA7RnbUAAAAJ&hl=en), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)

> **Abstract:**
> In this work, we present a novel approach to multi-view action recognition where we guide learned action representations to be separated from view-relevant information in a video. When trying to classify action instances captured from multiple viewpoints, there is a higher degree of difficulty due to the difference in background, occlusion, and visibility of the captured action from different camera angles. To tackle the various problems introduced in multi-view action recognition, we propose a novel configuration of learnable transformer decoder queries, in conjunction with two supervised contrastive losses, to enforce the learning of action features that are robust to shifts in viewpoints. Our disentangled feature learning occurs in two stages: the transformer decoder uses separate queries to separately learn action and view information, which are then further disentangled using our two contrastive losses. We show that our model and method of training significantly outperforms all other uni-modal models on four multi-view action recognition datasets: NTU RGB+D, NTU RGB+D 120, PKU-MMD, and N-UCLA. Compared to previous RGB works, we see maximal improvements of 1.5%, 4.8%, 2.2%, and 4.8% on each dataset, respectively.

### Usage Instructions
1. Navigate to desired directory
2. Modify parameters python file if necessary
3. Run the main python file

Example:
```bash
sbatch script.slurm
