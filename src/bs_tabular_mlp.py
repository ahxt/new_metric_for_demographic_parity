import logging
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import preprocess_adult_data
from utils import seed_everything, PandasDataSet, metric_evaluation
from model import MLP


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./data/")
    parser.add_argument('--dataset', type=str, default="adult")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--sensitive_attr', type=str, default="sex")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_hidden', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--exp_name', type=str, default="mlp")
    parser.add_argument('--log_screen', type=str, default="True")
    parser.add_argument('--round', type=int, default=0)


    args = parser.parse_args()
    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    log_screen = eval(args.log_screen)
    sensitive_attr = args.sensitive_attr
    exp_name = args.exp_name
    model = args.model
    num_hidden = args.num_hidden
    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    round = args.round


    seed_everything(seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    if log_screen is True:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info('parser.prog: {}'.format(parser.prog))
    logger.info("args:{}".format(args))


    if dataset_name == "adult":
        logger.info(f'Dataset: adult')
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed=42, path = '/data/han/data/fairness/adult', sensitive_attributes=sensitive_attr)
        X = pd.DataFrame( np.concatenate( [X_train, X_val, X_test] ) )
        y = pd.DataFrame( np.concatenate( [y_train, y_val, y_test] ) )[0]
        s = pd.DataFrame( np.concatenate( [A_train, A_val, A_test] ) )
    else:
        logger.info(f'Wrong dataset_name')

    logger.info(f'X.shape: {X.shape}')
    logger.info(f'y.shape: {y.shape}')
    logger.info(f's.shape: {s.shape}')
    logger.info(f's.shape: {s.value_counts().to_dict()}')


    n_features = X.shape[1]

    # split into train/val/test set
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_train, y_train, s_train, test_size=0.1 / 0.8, stratify=y_train, random_state=seed)


    logger.info(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}')
    logger.info(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}')
    logger.info(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}')


    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_val = X_val.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)


    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loader_no_shuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader( val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader( test_data, batch_size=batch_size, shuffle=False)


    s_val = s_val.values
    y_val = y_val.values

    s_test = s_test.values
    y_test = y_test.values

    s_train = s_train.values
    y_train = y_train.values


    clf = MLP(n_features=n_features, n_hidden=num_hidden).to(device)
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clf.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):

        ce_loss_list = []
        for x, y, s in train_loader:
            x = x.to(device)
            y = y.to(device)
            clf.zero_grad()
            p_y = clf(x)
            loss = clf_criterion(p_y, y)
            loss = loss
            ce_loss_list.append(loss.item())
            loss.backward()
            clf_optimizer.step()


        
        epoch_dict = vars(args)
        epoch_dict["epoch"] = epoch
        epoch_dict["ce_loss"] = np.mean(ce_loss_list)


        # validate train_data
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in train_loader_no_shuffle:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())



        pre_clf_train = np.concatenate(y_pre_list)
        s_train_sex = s_train

        train_metric = metric_evaluation( y_gt=y_train, y_pre= pre_clf_train, s=s_train_sex, prefix="train_")
        epoch_dict.update(train_metric)



        # validate val_data
        with torch.no_grad():
            # pre_clf_val_0, pre_clf_val_1, pre_clf_val_all  = clf(val_data.tensors[0].to(device))
            y_pre_list = []
            for x, y, s in val_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())



        # pre_clf_val = pre_clf_val_all.detach().cpu().numpy()
        pre_clf_val = np.concatenate(y_pre_list)
        s_val_sex = s_val

        val_metric = metric_evaluation( y_gt=y_val, y_pre= pre_clf_val, s=s_val_sex, prefix="val_")
        epoch_dict.update(val_metric)



        # validate test_data
        with torch.no_grad():
            y_pre_list = []
            for x, y, s in test_loader:
                x = x.to(device)
                y = y.to(device)
                p_y = clf(x)
                y_pre_list.append(p_y[:, 0].data.cpu().numpy())


        pre_clf_test = np.concatenate(y_pre_list)
        s_test_sex = s_test

        train_metric = metric_evaluation( y_gt=y_test, y_pre= pre_clf_test, s=s_test_sex, prefix="test_")
        epoch_dict.update(train_metric)

        # one epoch
        logger.info(f"epoch_dict: {epoch_dict}")



    logger.info(f"done experiment")





