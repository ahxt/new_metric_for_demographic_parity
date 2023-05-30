
import numpy as np
import pandas as pd
import numpy as np
from numpy import random
import os
import torch
from sklearn import metrics
import numpy as np
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from torch.utils.data import TensorDataset




# Define a function named ABPC that takes in four arguments:
# y_pred: predicted values
# y_gt: ground truth values
# z_values: binary values indicating the group membership of each sample
# bw_method: bandwidth method for the kernel density estimation (default is "scott")
# sample_n: number of samples to generate for the integration (default is 5000)
def ABPC( y_pred, y_gt, z_values, bw_method = "scott", sample_n = 5000 ):

    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the kernel density estimation (KDE) for each group
    kde0 = gaussian_kde(y_pre_0, bw_method = bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method = bw_method)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the KDEs at the x values
    kde1_x = kde1(x)
    kde0_x = kde0(x)

    # Compute the area between the two KDEs using the trapezoidal rule
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    # Return the computed ABPC value
    return abpc






# Define a function named ABCC that takes in three arguments:
# y_pred: predicted values
# y_gt: ground truth values
# z_values: binary values indicating the group membership of each sample
# sample_n: number of samples to generate for the integration (default is 10000)
def ABCC( y_pred, y_gt, z_values, sample_n = 10000 ):

    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    # Extract the predicted values for each group
    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    # Compute the empirical cumulative distribution function (ECDF) for each group
    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    # Generate a set of x values for the integration
    x = np.linspace(0, 1, sample_n)

    # Evaluate the ECDFs at the x values
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # Compute the area between the two ECDFs using the trapezoidal rule
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    # Return the computed ABCC value
    return abcc



# Define a function named demographic_parity that takes in three arguments:
# y_pred: predicted values
# z_values: binary values indicating the group membership of each sample
# threshold: threshold value for the predicted values (default is 0.5)
def demographic_parity(y_pred, z_values, threshold=0.5):

    # Extract the predicted values for each group and apply the threshold if it is not None
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]

    # Compute the absolute difference between the mean predicted value for each group
    parity = abs(y_z_1.mean() - y_z_0.mean())

    # Return the computed demographic parity value
    return parity




def seed_everything(seed=1314):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def metric_evaluation(y_gt, y_pre, s, prefix=""):

    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s= s.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    dp = demographic_parity(y_pre, s)
    dpe = demographic_parity(y_pre, s, threshold=None)
    abpc = ABPC( y_pre, y_gt, s)
    abcc = ABCC( y_pre, y_gt, s)

    metric_name = [ "accuracy", "ap", "dp", "dpe", "abpc", "abcc" ]
    metric_name = [ prefix+x for x in metric_name]
    metric_val = [ accuracy, ap, dp, dpe, abpc, abcc ]

    return dict( zip( metric_name, metric_val))



class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()