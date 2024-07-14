# Dual-stage Hyperspectral Image Classification Model with Spectral Supertoken [ECCV 2024]
by [Peifu Liu](https://scholar.google.com/citations?user=yrRXe-8AAAAJ&hl=zh-CN), [Tingfa Xu](https://scholar.google.com/citations?user=vmDc8dwAAAAJ&hl=zh-CN), [Jie Wang](https://roywangj.github.io/), Huan Chen, Huiyan Bai, and [Jianan Li](https://scholar.google.com.hk/citations?user=sQ_nP0ZaMn0C&hl=zh-CN&oi=ao).

[![arXiv](https://img.shields.io/badge/📃-arXiv-ff69b4)](https://arxiv.org/abs/2407.07307v1)
[![Google Drive](https://img.shields.io/badge/Google_Drive-4285F4?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/19UY5cgeXG03d56cj4CUSS85mr0rgR6Wt?usp=sharing)

## Requirements
In this repository, we provide a `requirements.txt` file that lists all the dependencies. Additionally, the installation `.whl` file for GDAL can be found at [Google Drive](https://drive.google.com/drive/folders/19UY5cgeXG03d56cj4CUSS85mr0rgR6Wt?usp=sharing) and can be installed directly using pip:
``` bash
pip install -r requirements.txt
pip install GDAL-3.4.1-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.whl
```

## Getting Started
### Preparation
Please download [WHU-OHS](https://irsip.whu.edu.cn/resources/resources_v2.php) dataset in `data`, which should be organized as follows:
```
|--data
    |--tr
        |--image
            |--O1_0001.tif
            |--O1_0002.tif
            |--...
        |--label
            |--O1_0001.tif
            |--O1_0002.tif
            |--...
    |--ts
        |--image
            |--O1_0003.tif
            |--O1_0004.tif
            |--...
        |--label
            |--O1_0003.tif
            |--O1_0004.tif
            |--...
    |--val
        |--image
            |--O1_0015.tif
            |--O1_0042.tif
            |--...
        |--label
            |--O1_0015.tif
            |--O1_0042.tif
            |--...
```

Our DSTC utilizes pre-trained weights. The pre-trained weights for ResNet and Swin will be downloaded automatically, while those for PVT can be downloaded from [Google Drive](https://drive.google.com/drive/folders/19UY5cgeXG03d56cj4CUSS85mr0rgR6Wt?usp=sharing). Please place them in the `/models/pre-trained` folder.

### Testing
If you wish to validate our method, our pre-trained weights are available on [Google Drive](https://drive.google.com/drive/folders/19UY5cgeXG03d56cj4CUSS85mr0rgR6Wt?usp=sharing). Please download them to the `/models/checkpoints` folder. Then run:
```bash
sh test.sh
```

### Training
To train our model, execute the `train_and_test.sh script`.  Model checkpoints will be stored in the `DataStorage/` directory. After training, the script will proceed to test the model and save the visualization results. 
```bash
sh train_and_test.sh
```


## Acknowledgement
We refer to the following repositories:
- [Context Cluster](https://github.com/ma-xu/Context-Cluster)
- [SPIN](https://github.com/ArcticHare105/SPIN)
- [CVSSN](https://github.com/lms-07/CVSSN)

Thanks for their great work!


## License
This project is licensed under the [LICENSE.md](LICENSE.md).