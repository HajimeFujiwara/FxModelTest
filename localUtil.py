# ガウス過程回帰による為替予測モデルに固有なユーティリティ関数を定義する

import torch
import pandas as pd
from pandas import Timestamp
from typing import Tuple, Callable
from data_preprocessing import DataConverter
from scipy import stats


def mk_data_for_gpr(orgData: DataConverter, base_date: Timestamp, device: str) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, torch.Tensor, pd.DataFrame]:
    train_data, test_data = orgData.get_train_test_df(base_date)

    drop_headers = ['date', 'Premium', 'DFd', 'DFf']
    mk_feature_fun: Callable[[pd.DataFrame], torch.Tensor] = lambda train_data: torch.from_numpy(train_data.drop(drop_headers, axis=1).values)

    train_prem = torch.from_numpy(train_data['Premium'].values)
    train_feature = mk_feature_fun(train_data)
    test_feature = mk_feature_fun(test_data)

    train_prem = train_prem.to(device)
    train_feature = train_feature.to(device)
    test_feature = test_feature.to(device)

    return train_prem, train_feature, train_data, test_feature, test_data


def mk_prediction(mean: torch.Tensor, variance: torch.Tensor, test_data: pd.DataFrame) -> Tuple[float, float, float, float]:

    std_dev = torch.sqrt(variance)

    def mk_fxrtn_from_premium(df: pd.DataFrame, prem: float, prem_std: float) -> Tuple[float, float]:
        fxrtn_mean = df['DFf']/df['DFd'] - 1.0 - df['USDJPYIV1MATM']/df['DFd'] * prem
        fxrtn_std = df['USDJPYIV1MATM']/df['DFd'] * prem_std
        return fxrtn_mean, fxrtn_std

    fxrtn_mean, fxrtn_std = mk_fxrtn_from_premium(test_data, mean.detach().cpu().item(), std_dev.detach().cpu().item())
    fxrtn_act, dmy = mk_fxrtn_from_premium(test_data, test_data['Premium'], 0.0)
    # scipyの正規分布オブジェクトを作成
    norm_dist = stats.norm(loc=fxrtn_mean, scale=fxrtn_std)

    # ０以上となる確率を計算
    probability_less_than_x = 1 - norm_dist.cdf(0.0)

    return fxrtn_act, fxrtn_mean, fxrtn_std, probability_less_than_x.item()
