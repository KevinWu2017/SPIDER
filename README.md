# SPIDER

> SPIDER: Unleashing Sparse Tensor Cores for Stencil Computation via Strided Swapping (PPoPP'26)

## Abstract

This repository contains the source code of SPIDER, a novel  system to converting stencil computation into sparse matrix multiplication using Sparse Tensor Core (SpTC).

## Preprequisties

- Hardware
  - 1x NVIDIA A100 GPU
- Software
  - CUDA (12.8 Tested). Should support all CUDA version above 12.5 (supporting `mma.sp.ordered_metadata`)
  - GCC (11.4 Tested) or Clang.
  - cuDNN (9.8.0 Tested).

You can use the `Dockerfile` to quickly prepare the environment. 

## Installing

### Preparing

First clone the code repository by:
```shell
git clone --recurse-submodules https://github.com/KevinWu2017/SPIDER.git SPIDER
cd SPIDER
```

> [!NOTE]
> If you prefer using docker for environment preparation, you can run:
> ```shell
> # Please make sure you nvidia driver supports cuda-12.8 (driver version newer than 570.26).
> docker build -t spider_ae .
> docker run --gpus all --name spider_ae -it spider_ae /bin/bash
> ```

Then create the virtual environment and install packages:
```shell
python3 -m venv spider_venv
source spider_venv/bin/activate
pip install -r requirements.txt
```

### Build

In project root, you can build the SPIDER and baselines by using:
```shell
./scripts/build_SPIDER.sh # build SPIDER
./scripts/build_baseline.sh # build baselines.
```

### Running
The executable program is listed in ./build/bin/
```shell
# e.g. To run a 2d stencil with SpTC and half precision, you can use
./build/bin/stencil_2d_half_sparse -M 10240 -N 10240 -T 1000 --profile
# you can check the acutal parameter input by running
./build/bin/stencil_2d_half_sparse -h
```

### Result Reproduction

Please first make sure the python venv is activated (by running `source spider_venv/bin/active`)

#### Figure 10
```shell
./scripts/Figure10_run.sh
python3 scripts/Figure10_draw.py
```
The output figure is saved in `./outputs/Figure10.pdf`

#### Figure 11
```shell
./scripts/Figure11_run.sh
python3 scripts/Figure11_draw.py
```
The output figure is saved in `./outputs/Figure11.pdf`

#### Figure 12
```shell
./scripts/Figure12_run.sh
python3 scripts/Figure12_draw.py
```
The output figure is saved in `./outputs/Figure12.pdf`

## Contact
If you have any questions, please send an email to the author at qiqi.gu@sjtu.edu.cn or cpwu_sjtu@sjtu.edu.cn.

## Reference
If you use our code, please cite our paper:

```bibtex
@misc{gu2025sptcstencilusingsparsetensor,
      title={SPTCStencil: Using Sparse Tensor Cores for Stencil Computation}, 
      author={Qiqi GU and Chenpeng Wu and Heng Shi and Jianguo Yao},
      year={2025},
      eprint={2506.22035},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2506.22035}, 
}
```

## Credits
This codebase uses [ConvStencil](https://github.com/HPHEX/ConvStencil), [LoRAStencil](https://github.com/HPHEX/LoRAStencil) and [TCStencil](https://github.com/buaa-hipo/TCStencil) as baseline . We sincerely thank the authors for their excellent work!
