<h1 align="center">Clustering-based Adaptive Query Generation for Semantic Segmentation</h1>

This repository is an official Pytorch implementation of the paper [**"Clustering-guided Class Activation for Weakly Supervised Semantic Segmentation"**](https://ieeexplore.ieee.org/abstract/document/10381698) <br>
Yeong Woo Kim and Wonjun Kim <br>
***IEEE Access***, Jan. 2024. </br>
<p align="center">
  <img src="https://github.com/DCVL-WSSS/ClusterCAM/assets/49578893/82ccf953-05b2-4b3e-9441-90b3a247a493" alt="The overall architecture of the proposed method."/>
</p>

*The overall architecture of the proposed method.*

## Installation
- Requirements
  - Pytorch >= 1.10

```bash
# We suggest to create a new conda environment with python version 3.9
conda create -n ClusterCAM python=3.9 -y
conda activate ClusterCAM

# Install Pytorch that is compatible with your CUDA version
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge

# Install requirements
pip install -r requirements.txt
conda install -c  conda-forge pydensecrf
```

## Dataset Preparation
- Download PASCAL VOC2012 dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012 (augmented annotations from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html), [DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0))
- Download MS COCO dataset:
  ```bash
  wget http://images.cocodataset.org/zips/train2014.zip
  wget http://images.cocodataset.org/zips/val2014.zip
  ```

- The resulting data structure should follow the hierarchy as below.

   ```
   ${REPO_DIR}  
   |-- voc12  
   |   |-- VOC2012
   |       |-- JPEGImages
   |       |-- Annotations
   |       |-- ImageSets
   |       |-- SegmentationClass
   |       |-- SegmentationClassAug
   |       |-- ...
   |-- coco   
   |   |-- COCO14
   |       |-- anno
   |       |-- annotations
   |       |-- SegmentationClass
   |       |-- train2014
   |       |-- val2014
   |       |-- ...
   |-- run_e2e.sh 
   |-- infer_e2e.sh 
   |-- ...
   ```

## How to use it
### Train
```bash
sh run_e2e.sh       # for the PASCAL VOC 2012 experiment
sh run_e2e_coco.sh  # for the MS COCO 2014 experiment
```
### Inference
```bash
sh infer_e2e.sh      # for the PASCAL VOC 2012 experiment
sh infer_e2e_coco.sh # for the MS COCO 2014 experiment
```
## Results
### Quantitative results
| Model                        | Dataset   | Valid | Test | Checkpoint            |
| ---------------------------- | --------- | ----- | -------- | --------------- |
| ClsuterCAM                | PASCAL VOC 2012 |70.3   | 70.7     | [Download](https://drive.google.com/file/d/1GqLasPff6hk_X9wJPr-wWmq4XwdhsWB9/view?usp=sharing)|
| ClsuterCAM                | MS COCO 2014      | 41.8   | -     | [Download](https://drive.google.com/file/d/1Gu01U0g6_usorubqydM3guX6XfOCtvRT/view?usp=sharing)|
### Qualitative results
![sem_seg_voc](https://github.com/DCVL-WSSS/ClusterCAM/assets/49578893/f71fd63d-55e4-48ef-9061-1d696cbc07c4)
*Results of semantic segmentation on the PASCAL VOC 2012 dataset [24]. From top to bottom: input images, ground truths, results by AFA, ToCo, and ClusterCAM (ours).*
![ablation_clusterCAM](https://github.com/DCVL-WSSS/ClusterCAM/assets/49578893/0105bd8f-81e2-4e80-865d-a5d1c205ea1f)
*Visualization examples of attention weights and ClusterCAMs. From top to bottom: input images, ground truths, patch-to-class attention weights, cluster-to-class attention weights, and ClusterCAMs.*

## Acknowledgments
This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government [Ministry of Science and ICT (MSIT)] under Grant 2023R1A2C1003699.

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[xulianuwa/MCTformer](https://github.com/xulianuwa/MCTformer)  </br>
[rulixiang/afa](https://github.com/rulixiang/afa)  </br>

## Citation
```bibtex
@ARTICLE{10381698,
  author={Kim, Yeong Woo and Kim, Wonjun},
  journal={IEEE Access}, 
  title={Clustering-Guided Class Activation for Weakly Supervised Semantic Segmentation}, 
  year={2024},
  volume={12},
  number={},
  pages={4871-4880},
  doi={10.1109/ACCESS.2024.3350176}}
