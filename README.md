# **An Annotation-free Restoration Network for Cataractous Fundus Images**
There is little access to large datasets of cataract images paired with their corresponding clear ones. Therefore, it is unlikely to build a restoration model for cataract images through supervised learning.

Here, we propose an annotation-free restoration network for cataractous fundus images [[arXir]](https://arxiv.org/abs/2203.07737).

![](./images/arcnet_overview.png)

# Structure-consistent Restoration Network for Cataract Fundus Image Enhancement

We propose a method of structure-consistent restoration network for cataract fundus image enhancement [[arXiv]](https://arxiv.org/abs/2206.04684). 

![](./images/scrnet_overview.png)

# A Generic Fundus Image Enhancement Network Boosted by Frequency Self-supervised Representation Learning

We propose a generic fundus image enhancement network (GFE-Net) to solute any image degradation without supervised data.


# Prerequisites

\- Win10

\- Python 3

\- CPU or NVIDIA GPU + CUDA CuDNN

# Environment (Using conda)

```
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing opencv-python

conda install pytorch torchvision -c pytorch # add cuda90 if CUDA 9

conda install visdom dominate -c conda-forge # install visdom and dominate
```

# Simulate cataract-like images

Use the script in ./utils/catacact_simulation.py

Before simulation, you can use ./data/run_pre_process.py to get the pre-processed image and mask.


# Visualization when training

python -m visdom.server

# To open this link in the browser

http://localhost:8097/

# Dataset preparation

To set up your own dataset constructed like images/cataract_dataset. Note that the number of source images should be bigger than the number of target images, or you can design you own data loader.

Use util/get_mask.py to get the mask of target and source, and place them into the dataset.

For a cataract dataset, the architecture of the directory should be:

--cataract_dataset

​	--source

​		--AB.jpg

​	--source_mask

​		--AB.png

​	--target

​		--A.jpg

​	--target_mask

​		--A.png

## Trained model's weight

**Note:** If you want to use ArcNet in your own dataset, please re-train a new model with your own data, because it is a method based on domain adaptation, which means it needs target data (without ground truth) in the training phase.

For the model of ArcNet 'An Annotation-free Restoration Network for Cataractous Fundus Images'', ScrNet, 'Structure-consistent Restoration Network for Cataract Fundus Image Enhancement', and  GFE-Net, 'A Generic Fundus Image Enhancement Network Boosted by Frequency Self-supervised Representation Learning' please download the pretrained model and place the document based on the following table:

|        | Baidu Cloud                                                  | Google Cloud                                                 | Directory                                        |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| ArcNet | [ArcNet [3xg0]](https://pan.baidu.com/s/1hFt0bMpBb5V0Gj0ogYHGbA) | [ArcNet](https://drive.google.com/file/d/1VJ-_W7rRmy90AcgeAJtt_z7fgeBpC4Id/view?usp=share_link) | project_root/checkpoints/scrnet/latest_net_G.pth |
| SCRNet | [SCR-Net in Baidu [o32j]](https://pan.baidu.com/s/1WKeyjMoXElkOJ4gkNO-71w) | [SCRNet](https://drive.google.com/file/d/1TwZTPLGRQAobvJM99mjxCkKgAKFib9Ap/view?usp=sharing) | project_root/checkpoints/arcnet/latest_net_G.pth |
| GFENet | Coming soon                                                  | Coming soon                                                  | project_root/checkpoints/gfenet/latest_net_G.pth |

# Command to run

Please note that root directory is the project root directory.

## Train

For ArcNet:

```
python train.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --batch_size 8 --gpu_ids 0
```

For ScrNet:

```
python train.py --dataroot ./images/cataract_dataset --name scrnet --model scrnet --input_nc 3 --direction AtoB --dataset_mode cataract_with_mask --norm instance --batch_size 8 --gpu_ids 0 --lr_policy linear --n_epochs 150 --n_epochs_decay 50
```

For GFE-Net:

Released soon.

## Test & Visualization

For ArcNet:

```
python test.py --dataroot ./images/cataract_dataset --name arcnet --model arcnet --netG unet_256 --input_nc 6 --direction AtoB --dataset_mode cataract_guide_padding --norm batch --gpu_ids 0
```

For ScrNet:

```
python test.py --dataroot ./images/cataract_dataset --name scrnet --model scrnet --netG unet_combine_2layer --direction AtoB --dataset_mode cataract_with_mask --input_nc 3 --output_nc 3
```

For GFE-Net:

Released soon.

## Train and test for 512 x 512

Please modify the option of 'load_size' to 512 when using train.py

```
--load_size 512 --crop_size 256
```

In  test.py

```
--load_size 512 --crop_size 512
```

# Reference

[1] A. Mitra, S. Roy, S. Roy, and S. K. Setua, “Enhancement and restoration of non-uniform illuminated fundus image of retina obtained through thin layer of cataract,” Computer methods and programs in biomedicine, vol. 156, pp. 169–178, 2018.

[2] Cheng J ,  Li Z ,  Gu Z , et al. Structure-Preserving Guided Retinal Image Filtering and Its Application for Optic Disk Analysis[J]. IEEE TRANSACTIONS ON MEDICAL IMAGING MI, 2018.

[3] L. Cao, H. Li, and Y. Zhang, “Retinal image enhancement using lowpass filtering and α-rooting,” Signal Processing, vol. 170, p. 107445, 2020.

[4] Isola P ,  Zhu J Y ,  Zhou T , et al. Image-to-Image Translation with Conditional Adversarial Networks[C]// IEEE Conference on Computer Vision & Pattern Recognition. IEEE, 2016.

[5] Zhu J Y ,  Park T ,  Isola P , et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks[J]. IEEE, 2017.

[6] Luo Y ,  K  Chen,  Liu L , et al. Dehaze of Cataractous Retinal Images Using an Unpaired Generative Adversarial Network[J]. IEEE Journal of Biomedical and Health Informatics, 2020, PP(99):1-1.

[7] Z. Shen, H. Fu, J. Shen, and L. Shao, “Modeling and enhancing lowquality retinal fundus images,” IEEE transactions on medical imaging, vol. 40, no. 3, pp. 996–1006, 2020.

[8] Li H ,  Liu H ,  Hu Y , et al. Restoration Of Cataract Fundus Images Via Unsupervised Domain Adaptation[C]// 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI). IEEE, 2021.

[9] Li H, Liu H, Hu Y, et al. An Annotation-free Restoration Network for Cataractous Fundus Images[J]. IEEE Transactions on Medical Imaging, 2022.

# Citation

```
@article{li2022annotation,
  title={An Annotation-free Restoration Network for Cataractous Fundus Images},
  author={Li, Heng and Liu, Haofeng and Hu, Yan and Fu, Huazhu and Zhao, Yitian and Miao, Hanpei and Liu, Jiang},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
@article{li2022structure,
  title={Structure-consistent Restoration Network for Cataract Fundus Image Enhancement},
  author={Li, Heng and Liu, Haofeng and Fu, Huazhu and Shu, Hai and Zhao, Yitian and Luo, Xiaoling and Hu, Yan and Liu, Jiang},
  journal={arXiv preprint arXiv:2206.04684},
  year={2022}
}
```
