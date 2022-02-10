## 1. Instruction
Code for submission (Retiring $\Delta \text{DP}$: Brand-New Metrics for Demographic Parity)


## 2. Python package
```
torch                         1.10.0
statsmodels                   0.13.1
scikit-learn                  1.0.1
pandas                        1.3.4
numpy                         1.21.2
aif360                        0.4.0
```

## 3. Run cmd 
run the following commands at current directy
```
bash run.sh
```

```python
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_mlp.py --data_path ./data/adult  --dataset adult --sensitive_attr sex --exp_name adult_mlp --batch_size 256 --epoch 10 --seed 31314
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_reg.py --data_path ./data/adult  --dataset adult --sensitive_attr sex --exp_name adult_reg --batch_size 256 --epoch 10 --seed 31314 --lam 1
CUDA_VISIBLE_DEVICES=0 python -u ./src/bs_tabular_adv.py --data_path ./data/adult --dataset adult --sensitive_attr sex --exp_name adult_adv --batch_size 256 --epoch 40 --seed 31314 --lam 170
```

## 4. Code for ABPC and ABCC
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