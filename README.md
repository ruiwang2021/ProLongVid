# [EMNLP 2025] ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning

> [**ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning**](https://aclanthology.org/2025.emnlp-main.1570)
> <br>
> [Rui Wang](https://scholar.google.com/citations?user=116smmsAAAAJ&hl=en), [Bohao Li](https://bohao-lee.github.io/), [Xiyang Dai](https://scholar.google.com/citations?user=QC8RwcoAAAAJ&hl=en), [Jianwei Yang](https://jwyang.github.io), Yi-Ling Chen, [Zhen Xing](https://chenhsing.github.io/), Yifan Yang, [Dongdong Chen](https://www.dongdongchen.bid/), [Xipeng Qiu](https://xpqiu.github.io/), [Zuxuan Wu](https://zxwu.azurewebsites.net/) and [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)
> <br>

## Datasets

ProLongVid data (Annotations and videos) have been uploaded to [huggingface](https://huggingface.co/datasets/prolongvid/ProLongVid_data). 

## Models

| **Model**         | **Frame (Train)** | **Frame (Test)** | Video-MME (w/o sub) | **Huggingface** |
|-------------------|-------------------|------------------|---------------------|-----------------|
| Image-SFT-7B | - | 32 | 57.6 | [prolongvid_image_sft_7B](https://huggingface.co/prolongvid/prolongvid_image_sft_7B) |
| ProLongVid-Stage-1-7B | 32 | 32 | 60.1 | [prolongvid_stage1_7B](https://huggingface.co/prolongvid/prolongvid_stage1_7B) |
| ProLongVid-Stage-2-7B | 128 | 128 | 63.6 | [prolongvid_stage2_7B](https://huggingface.co/prolongvid/prolongvid_stage2_7B) |
| ProLongVid-Stage-3-7B | 192 | 192 | 63.8 | [prolongvid_7B](https://huggingface.co/prolongvid/prolongvid_7B) |
|  | 192 | 256 | 64.7 |  |

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

## ToDo

- [ ] More comprehensive training and testing tutorials.
- [ ] More efficient training framework that supports **sequence parallelism**.
- [ ] Original **Dense Video Caption** data.
- [ ] New models trained from stronger image-LMM baseline.

## Contact

For questions, feedback, or collaboration opportunities, feel free to reach out: wangrui21@m.fudan.edu.cn

## Acknowledgement

We build this repo based on [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT). Thanks for their wonderful works.

## Citation

If you find our works useful for your research, please consider citing:
```
@inproceedings{wang2025prolongvid,
  title={ProLongVid: A Simple but Strong Baseline for Long-context Video Instruction Tuning},
  author={Rui Wang and Bohao Li and Xiyang Dai and Jianwei Yang and Yi-Ling Chen and Zhen Xing and Yifan Yang and Dongdong Chen and Xipeng Qiu and Zuxuan Wu and Yu-Gang Jiang},
  booktitle={EMNLP},
  year={2025}
}
```
