# Minusformer

The repo is the official implementation for the paper: [Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals](https://arxiv.org/abs/2402.02332).

Cited by [Awesome Time Series Forecasting/Prediction Papers](https://github.com/ddz16/TSFpaper); [English Blog](); [Zhihu](https://zhuanlan.zhihu.com/p/703948963); [CSDN Blog](https://blog.csdn.net/liangdaojun/article/details/139748253)

## 1. Introduction

It is find that ubiquitous time series (TS) forecasting models are prone to severe overfitting. To cope with this problem, we embrace a de-redundancy approach to progressively reinstate the intrinsic values of TS for future intervals. Specifically, we renovate the vanilla Transformer by reorienting the information aggregation mechanism from addition to subtraction. 
The output of subsequent modules in this branch will subtract the previously learned results, enabling the model to learn the residuals of the supervision signal, layer by layer.
This designing facilitates the learning-driven implicit progressive decomposition of the input and output streams, empowering the model with heightened versatility, interpretability, and resilience against overfitting.
Extensive experiments demonstrate the proposed method outperform existing state-of-the-art methods, yielding an average performance improvement of **11.9%** across various datasets.

<div align=center><img src="Images/performance.jpg" width="500"></div>

## 2. Contributions

 - It is find that ubiquitous time series (TS) forecasting models are prone to severe overfitting.
 - We renovate the vanilla Transformer by reorienting the information aggregation mechanism from addition to subtraction. And an auxiliary output branch is incorporated into each block of the original model to construct a highway leading to the ultimate prediction.
 - The proposed Minusformer facilitates the learning-driven implicit progressive decomposition of the input and output streams, empowering the model with heightened versatility, interpretability, and resilience against overfitting.
 - Minusformer outperform existing state-of-the-art methods, yielding an average performance improvement of **11.9%** across various datasets.

<div align=center><img src="Images/arch.jpg" width="600"></div>

## 3. Training and Testing Minusformer
### 1) Dataset 
The datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2ea5ca3d621e4e5ba36a/).

### 2) Clone the code repository
```git
git clone git@github.com:Anoise/Minusformer.git
```

### 3) Training on Time Series Dataset
Go to the directory "Minusformer/", we'll find that the bash scripts are all in the 'scripts' folder, like this:

```
scripts/
├── Electricity
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minusformer_96S.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
├── ETTh1
│   ├── Minusformer_ETTh1_336M.sh
│   ├── Minusformer_ETTh1_96M.sh
│   └── Minusformer_ETTh1_96S.sh
├── ETTh2
│   ├── Minusformer_ETTh2_336M.sh
│   ├── Minusformer_ETTh2_96M.sh
│   └── Minusformer_ETTh2_96S.sh
├── ETTm1
│   ├── Minusformer_ETTm1_336M.sh
│   ├── Minusformer_ETTm1_96M.sh
│   └── Minusformer_ETTm1_96S.sh
├── ETTm2
│   ├── Minusformer_ETTm2_336M.sh
│   ├── Minusformer_ETTm2_96M.sh
│   └── Minusformer_ETTm2_96S.sh
├── Exchange
│   └── Minusformer_96S.sh
├── Pems
│   ├── Minusformer_336M.sh
│   └── Minusformer_96M.sh
├── SolarEnergy
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
├── Traffic
│   ├── Minus-Autoformer_96M.sh
│   ├── Minus-Flowformer_96M.sh
│   ├── Minusformer_336M.sh
│   ├── Minusformer_96M.sh
│   ├── Minusformer_96S.sh
│   ├── Minus-Informer_96M.sh
│   └── Minus-Periodformer_96M.sh
└── Weather
    ├── Minus-Autoformer_96M.sh
    ├── Minus-Flowformer_96M.sh
    ├── Minusformer_336M.sh
    ├── Minusformer_96M.sh
    ├── Minusformer_96S.sh
    ├── Minus-Informer_96M.sh
    └── Minus-Periodformer_96M.sh    
```

Then, you can run the bash script like this:
```shell
    bash scripts/Electricity/Minusformer-96M.sh
```



Note that:
- Model was trained with Python 3.7 with CUDA 11.2.
- Model should work as expected with pytorch >= 1.12 support was recently included.

## 4. Performace on Multivariate Time Series

Minusformer achieves the consistent SOTA performance across all datasets and prediction length configurations.

<div align=center><img src="Images/m_table.jpg"></div>

## 5. Performace on Univariate Time Series

Minusformer continues to maintain a SOTA performance across various prediction length settings compared to the benchmarks.

<div align=center><img src="Images/n_table.jpg" width="600"></div>

## 6. Very Effectiveness

Ablation studies on various components of Minusformer. All results are averaged across all prediction lengths. The variables X and Y represent the input and output streams, while the signs ‘+’ and ‘-’ denote the addition or subtraction operations used when the streams’ aggregation. The letter ‘G’ denotes adding a gating mechanism to the output of each block.

<div align=center><img src="Images/variates.jpg"></div>

## 7. Good Generality 

Ablation Studies of Minusformer with Various Attention. All results are averaged across all prediction lengths. The tick labels of the X-axis are the abbreviation of Attention types.

<div align=center><img src="Images/other_attn.jpg"></div>

## 8. Good Interpretability

Visualization depicting the output of each block in Minusformer. The experiment was implemented on the Traffic dataset using the setting of Input-96-Predict-96. The utilized models have the same hyperparameter settings and similar performance.

<div align=center><img src="Images/interpretable.jpg"></div>

## 9. Go Deeper

Given the Minusformer’s robustness against overfitting, it can be designed with considerable depth. Even with the Minusformer blocks deepened to 8
or 16, it continues to exhibit excellent performance.

<div align=center><img src="Images/godeeper.jpg" width="400"></div>

## 10. Citations

Daojun Liang, Haixia Zhang, Dongfeng Yuan, Bingzheng Zhang and Minggao Zhang, Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals. arXiv preprint arXiv:2402.02332  (2024).

```
@article{liang2024minusformer,
  title={Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals},
  author={Liang, Daojun and Zhang, Haixia and Yuan, Dongfeng and Zhang, Bingzheng and Zhang, Minggao},
  journal={arXiv preprint arXiv:2402.02332},
  year={2024}
}
```


