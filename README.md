[![arXiv](https://img.shields.io/badge/arXiv-2306.05584-b31b1b.svg)](https://arxiv.org/abs/2306.05584)
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

# Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation (NeurIPS 2023)

## Citation
If our work has been helpful in your research, please consider citing it as follows:

```
@inproceedings{zhong2023multi,
  title={Multi-body SE (3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation},
  author={Zhong, Jia-Xing and Cheng, Ta-Ying and He, Yuhang and Lu, Kai and Zhou, Kaichen and Markham, Andrew and Trigoni, Niki},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## 1. Setup

### Prerequisites

**(1) PyTorch Installation**

   Ensure you have a GPU-supported version of PyTorch that is compatible with your system. We have confirmed compatibility with PyTorch 1.9.0.

**(2) Additional Libraries**

   Install the [PointNet2 library](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master) and other required Python packages:

   ```bash
   cd pointnet2
   python setup.py install
   cd ..
   pip install -r requirements.txt
   ```

**(3) EPN Dependencies**

   Follow the specific instructions in `./EPN_PointCloud/README.md` to set up [EPN](https://github.com/nintendops/EPN_PointCloud):

   ```bash
   cd EPN_PointCloud
   pip install -r requirements.txt
   cd vgtk
   python setup.py install
   ```

**(4) [Optional] Open3D Installation**

   For visualizing point cloud segmentation:

   ```bash
   pip install open3d
   ```

## 2. Data Preparation

### SAPIEN Dataset (Provided by [MBS](https://github.com/huangjh-pub/multibody-sync))

Download necessary data from the following links and place them in your specified `${SAPIEN}` directory:

- **Training + Validation Set (`mbs-shapepart`)**: [Google Drive](https://drive.google.com/file/d/1aGTn-PYxLjnhj9UKlv4YFV3Mt1E3ftci/view?usp=sharing)
- **Test Set (`mbs-sapien`)**: [Google Drive](https://drive.google.com/file/d/1HR2X0DjgXLwp8K5n2nsvfGTcDMSckX5Z/view?usp=sharing)

## 3. Initial Scene Flow Estimation

Download the checkpoint for the self-supervised scene flow network from [OGC](https://github.com/vLAR-group/OGC):
- **Checkpoint (`sapien_unsup`)**: [Dropbox](https://www.dropbox.com/s/k4hv71952i0yrye/OGC_ckpt.zip?dl=0&e=1&file_subpath=%2Fckpt%2Fflow%2Fsapien%2Fsapien_unsup%2Fbest.pth.tar).

In our experiments, we directly use the **same** trained checkpoint as their scene flow network for fair comparisons. 
If needed, the scene flow network can be trained and tested as follows:
### Training

Train the model using the provided configuration:

```bash
python train_flow.py config/flow/sapien/sapien_unsup.yaml
```

### Testing

Evaluate and save the scene flow estimations with:

```bash
python test_flow.py config/flow/sapien/sapien_unsup.yaml --split ${SPLIT} --save
```
Replace `${SPLIT}` with either `train`, `val`, or `test` as required.

## 4. Multi-body SE(3) Equivariant Models

### Supervised Learning

Train the segmentation network using full annotations:

```bash
# Two 12GB GPUs are required.
CUDA_VISIBLE_DEVICES=0,1 python eq_train_2head_sup.py config/seg/sapien/eq_sapien_2head_sup_sapien.yaml
```

Evaluate the segmentation results with:

```bash
python eq_test_2head_seg.py config/seg/sapien/eq_sapien_2head_sup_sapien.yaml --split test
```

### Unsupervised Learning

Train the model without annotations:

```bash
# Two 12GB GPUs are required.
CUDA_VISIBLE_DEVICES=0,1 python eq_train_2head_unsup.py config/seg/sapien/sapien_unsup_woinv.yaml 
```

Evaluate segmentation and scene-flow results:

```bash
# Segmentation
python eq_test_2head_seg.py config/seg/sapien/eq_sapien_2head_unsup_woinv.yaml --split test --round 0
# Scene Flow: Two 12GB GPUs are required.
CUDA_VISIBLE_DEVICES=0,1 python eq_test_2head_oa_icp.py config/seg/sapien/eq_sapien_2head_unsup_woinv.yaml --split test
```
