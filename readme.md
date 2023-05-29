html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>

Implmentation for TMLR paper: `MLPInit: Embarrassingly Simple GNN Training Acceleration with MLP Initialization`, [[Openreview](https://openreview.net/forum?id=LjDFIWWVVa)], [[Arxiv](https://arxiv.org/abs/2301.13443)], by Xiaotian Han*, Zhimeng Jiang*, Hongye Jin*, Zirui Liu, Na Zou, Qifan Wang, Xia Hu

# Introduction

Lots of fairness definitions (e.g., demographic parity, equalized opportunity) has been proposed to solve different types of fairness issues. In this paper, we focus on the measurement of demographic parity, $\Delta DP$, which requires the predictions of a machine learning model should be independent on sensitive attributes.

## Drawbacks of commonly used $\Delta DP$ 
In this paper, we rethink the rationale of $\Delta DP$ and investigate its limitations on measuring the violation of demographic parity. There are two commonly used implementations of $\Delta DP$, including $\Delta DP_c$ (i.e., the difference of the mean of the predictive probabilities between different groups, and $\Delta DP_b$ (i.e., the difference of the proportion of positive prediction between different groups. We argue that $\Delta DP$, as a metric, has the following drawbacks:
- **First, zero-value $\Delta DP$ does not guarantee zero violation of demographic parity.** One fundamental requirement for the demographic parity metric is that the zero-value metric must be equivalent to the achievement of demographic parity, and vice versa. However, zero-value $\Delta DP$ does not indicate the establishment of demographic parity since $\Delta DP$ is a necessary but insufficient condition for demographic parity. An illustration of ACS-Income data is shown in \cref{fig:intro} to demonstrate that $\Delta DP$ fails to assess the violation of demographic parity since it reaches (nearly) zero on an unfair model (the middle subfigure in \cref{fig:intro}). 
- **Second, the value of $\Delta DP$ does not accurately quantify the violation of demographic parity and the level of fairness.** Different values of the same metric should represent different levels of unfairness, which is still true even in a monotonously transformed space. $\Delta DP$ does not satisfy this property, resulting in it being unable to compare the level of fairness based solely on its value.
- **Third, $\Delta DP_b$ value is highly correlated to the selection of the threshold for the classification task.** To make a decision based on predictive probability, one predefined threshold is needed. If the threshold for downstream tasks changes, the proportion of positive predictions of different groups will change accordingly, resulting in a change in $\Delta DP_b$. The selection of the threshold greatly affects the value of $\Delta DP_b$ (validated by:
<p align="center">
<img width="600" src="./figure/income_pdf_cdf.pdf>
</p>


<p align="center">
<img width="600" src="./figure/adult_pdf_cdf.pdf>
</p>

One specific $\Delta DP_b=0$ can not guarantee demographic parity under the on-the-fly threshold change. 



## The Proposed **ABPC** & **ABCC**

We propose two distribution-level metrics, namely **A**rea **B**etween **P**robability density function **C**urves (ABPC) and **A**rea **B**etween **C**umulative density function **C**urves (ABCC), to retire $\Delta DP_{c}$ and $\Delta DP_{b}$, respectively.

The advantage is that such independence can be guaranteed over any threshold, while $\Delta DP$ can only guarantee independence over a specific threshold.

The proposed metrics satisfy all (or partial) two criteria to guarantee the correctness of measuring demographic parity and address the limitations of the existing metrics, as well as estimation tractability from limited data samples. 

We further propose two distribution-level metrics, **A**rea **B**etween **P**robability density function **C**urves (ABPC) and **A**rea **B**etween **C**umulative density function **C**urves (ABCC), to resolve the limitations of $\Delta DP$, which are theoretically and empirically capable of measuring the violation of demographic parity.


n view of the drawbacks of fairness metrics ∆DP for demographic parity, we first propose two criteria to
theoretically guide the development of the metric on demographic parity: 1) Sufficiency: zero-value fairness
metric must be a necessary and sufficient condition to achieve demographic parity. 2) Fidelity: The metric
should accurately reflect the degree of unfairness, and the difference of such a metric indicates the fairness
gap in terms of demograph

In this section, we formally define two distribution-level metrics to measure the violation of demographic parity. \textsf{ABPC} is defined as the total variation distance ($TV$) between probability density functions with different sensitive attribute groups as follows:
\begin{equation} \label{eq:ABPC}
    \mathsf{\textsf{ABPC}} = TV(f_0(x), f_1(x)) = \int_{0}^{1}\left|f_0(x) - f_1(x) \right| \mathrm{d}x,
\end{equation}
where $f_0(x)$ and $f_1(x)$ are the PDFs of the predictive probability of different demographic groups. Similarly, \textsf{ABCC} is defined as the total variation between prediction cumulative density functions with different sensitive attribute groups as follows:
\begin{equation}\label{eq:ABCC}
    \mathsf{\textsf{ABCC}} = TV(F_0(x), F_1(x)) = \int_{0}^{1}\left|F_0(x) - F_1(x) \right| \mathrm{d}x,
\end{equation}
where $F_0(x)$ and $F_1(x)$ are the CDF of the predictive probability of demographic groups.

Note that the proposed metrics can be easily extended to the multi-value sensitive attribute setting. Suppose the sensitive attribute has $m$ values, then we compute the \textsf{ABPC} of each pair of groups with different sensitive attributes and then average them. \textsf{ABCC} for multi-value sensitive attributes with $m$ values can also be computed in a similar way. Since the $m$ is small in practice, the computational complexity is acceptable.

In this section, we formally define two distribution-level metrics to measure the violation of demographic parity. **ABPC** is defined as the total variation distance (TV) between probability density functions with different sensitive attribute groups as follows: 
ABPC = TV(f_0(x), f_1(x)) = ∫_{0}^{1}|f_0(x) - f_1(x) | dx,    
where $f_0(x)$ and $f_1(x)$ are the PDFs of the predictive probability of different demographic groups. 
Similarly, **ABCC** is defined as the total variation between prediction cumulative density functions with different sensitive attribute groups as follows:
 
ABCC = TV(F_0(x), F_1(x)) = ∫_{0}^{1}|F_0(x) - F_1(x)| dx,
where $F_0(x)$ and $F_1(x)$ are the CDF of the predictive probability of demographic groups. 




# Implementation 


## 1. Python package
```
torch                         1.10.0
statsmodels                   0.13.1
scikit-learn                  1.0.1
pandas                        1.3.4
numpy                         1.21.2
aif360                        0.4.0
```

## 2. Run cmd 
run the following commands at current directy
```
bash run.sh
```

```python
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_mlp.py --data_path ./data/adult  --dataset adult --sensitive_attr sex --exp_name adult_mlp --batch_size 256 --epoch 10 --seed 31314
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_reg.py --data_path ./data/adult  --dataset adult --sensitive_attr sex --exp_name adult_reg --batch_size 256 --epoch 10 --seed 31314 --lam 1
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_adv.py --data_path ./data/adult --dataset adult --sensitive_attr sex --exp_name adult_adv --batch_size 256 --epoch 40 --seed 31314 --lam 170
```

## 3. Python Implementation for ABPC and ABCC
```python
def ABPC( y_pred, y_gt, z_values, bw_method = "scott", sample_n = 5000 ):

    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # KDE PDF     
    kde0 = gaussian_kde(y_pre_0, bw_method = bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method = bw_method)

    # integration
    x = np.linspace(0, 1, sample_n)
    kde1_x = kde1(x)
    kde0_x = kde0(x)
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    return abpc

```


```python
def ABCC( y_pred, y_gt, z_values, sample_n = 10000 ):

    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # empirical CDF 
    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    # integration
    x = np.linspace(0, 1, sample_n)
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    return abcc
```