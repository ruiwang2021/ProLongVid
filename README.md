# ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning

## Datasets

Annotations and videos have been uploaded to [huggingface](https://huggingface.co/datasets/prolongvid/ProLongVid_data). 

## Models

| **Model**                          | **Huggingface** |
|-----------------------------------|------------------|
| Image-SFT-7B | https://huggingface.co/prolongvid/prolongvid_image_sft_7B |
| ProLongVid-Stage-1-7B | https://huggingface.co/prolongvid/prolongvid_stage1_7B |
| ProLongVid-Stage-2-7B | https://huggingface.co/prolongvid/prolongvid_stage2_7B |
| ProLongVid-Stage-3-7B | https://huggingface.co/prolongvid/prolongvid_7B |

## Installation

For training:
```
- conda create -n llava python==3.10 -y
- conda activate llava
- conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
- pip install -e ".[train]"
- pip install bitsandbytes
- pip install tensorboardX
- pip install transformers==4.43.4
- pip install flash-attn==2.6.3 --no-build-isolation
```

## Training

For example, for stage-2 training, run the following script:
```
bash scripts/train/train_stage2.sh 
```

## Eval

Please use the code of lmms-eval in this repo to install the environment, and follow the instruction of this version to perform evaluation of video benchmarks.

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out: wangrui21@m.fudan.edu.cn

## Citation

If you find our works useful for your research, please consider citing:
```
@inproceedings{
wang2025prolongvid,
title={ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning},
author={Rui Wang and Bohao Li and Xiyang Dai and Jianwei Yang and Yi-Ling Chen and Zhen Xing and Yifan Yang and Dongdong Chen and Xipeng Qiu and Zuxuan Wu and Yu-Gang Jiang},
booktitle={EMNLP},
year={2025}
}
```
