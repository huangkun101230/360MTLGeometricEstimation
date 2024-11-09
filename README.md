<p align="center">

  <h1 align="center">Multi-task Geometric Estimation of Depth and Surface Normal from Monocular 360° Images</h1>
  <p align="center">
    <a href="https://github.com/huangkun101230">Kun Huang</a>,
    <a href="https://people.wgtn.ac.nz/fanglue.zhang?_ga=2.161972092.1710887990.1730665987-888529436.1730407824">Fang-Lue Zhang*</a>,
    <a href="https://fangfang-zhang.github.io/">Fangfang Zhang</a>,
    <a href="https://profiles.cardiff.ac.uk/staff/laiy4">Yu-Kun Lai</a>,
    <a href="https://profiles.cardiff.ac.uk/staff/rosinpl">Paul Rosin</a>,
    <a href="https://people.wgtn.ac.nz/neil.dodgson?_ga=2.172996195.1710887990.1730665987-888529436.1730407824">Neil A. Dodgson</a>,

  </p>
    <p align="center">
    *Corresponding authors

  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2411.01749">Paper</a>
  <div align="center"></div>
</p>

## Introduction
Geometric estimation is required for scene understanding and analysis in panoramic 360° images. Current methods usually predict a single feature, such as depth or surface normal. These methods can lack robustness, especially when dealing with intricate textures or complex object surfaces. <b>We introduce a novel multi-task learning (MTL) network that simultaneously estimates depth and surface normals from 360° images</b>. Our first innovation is our MTL architecture, which enhances predictions for both tasks by integrating geometric information from depth and surface normal estimation, enabling a deeper understanding of 3D scene structure. Another innovation is our fusion module, which bridges the two tasks, allowing the network to learn shared representations that improve accuracy and robustness.

<p align="center">
  <a href="">
    <img src="assets/relit.gif" alt="teaser" width="95%">
  </a>
</p>
<p align="left">
  Our MTL model provides more accurate geometric estimations for 360° images compared to other methods, particularly in the red rectangle highlighted regions. The results are visualized as 3D point clouds, with both RGB data and color-coded surface normal maps.
</p>
<br>

<p align="center">
  <a href="">
    <img src="./assets/pipeline.png" alt="pipeline" width="95%">
  </a>
</p>
<p align="left">
  Our network architecture. Our network consists of two branches: $B_{depth}$ (in blue) and $B_{normal}$ (in red), dedicated to depth and surface normal estimation, respectively. A fusion module (in green) is employed to fuse the feature maps between each encoder level of $B_{depth}$ and $B_{normal}$ and feed the fused features into the next encoder level. The fused features are also concatenated with the original depth or normal features and fed to the corresponding decoder blocks. The final depth and normal maps are predicted in a multi-scale manner.
</p>
<br>

## Installation
Provide installation instructions for your project. Include any dependencies and commands needed to set up the project.

```shell
# Clone the repository
git clone https://github.com/huangkun101230/360MTLGeometricEstimation.git
cd 360MultitaskGeometricEstimation

# Install dependencies
conda env create -f conda_env.yml
conda activate mtl360
```


## Running
Please [download our pretrained models](https://drive.google.com/drive/folders/1B_GI-3mc8hgLWi0OXq-msBbGxeyuyNeB?usp=sharing), and save these models to "saved_models/models".
To test on provided data in "./input_data"
```shell
python evaluate.py
```
The results will be saved at "./results/saved_models/"

For training our model, please modify the path in our dataset:
For example, in datasets/dataset3D60.py, function gather_filepaths, change local="./input_data/" with your downloaded path

and run
```shell
python train.py
```

## Dataset
We mainly evaluate our method on [3D60 dataset](https://vcl3d.github.io/3D60/) and [Structured3D dataset](https://structured3d-dataset.org/).


## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@article{huang2024multi,
  title={Multi-task Geometric Estimation of Depth and Surface Normal from Monocular 360 $\{$$\backslash$deg$\}$ Images},
  author={Huang, Kun and Zhang, Fang-Lue and Zhang, Fangfang and Lai, Yu-Kun and Rosin, Paul and Dodgson, Neil A},
  journal={arXiv preprint arXiv:2411.01749},
  year={2024}
}
```
