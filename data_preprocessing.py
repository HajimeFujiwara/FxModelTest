import boto3
import pandas as pd
from pandas import Timestamp
import numpy as np
from typing import Dict, Optional, List, Tuple, Callable
from typing_extensions import Self  # version 3.11より typingに含まれる
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class S3DataFrame:
    """
    S3からデータを読み込んでデータフレームとして扱うためのクラス。
    """
    bucket_name: str  # S3バケットの名前
    file_name: str  # ファイル名
    df: Optional[pd.DataFrame] = None  # データフレーム

    def __post_init__(self):
        """
        クラスの初期化後にデータフレームが提供されていない場合、S3からデータを読み込みます。
        %表示の変数は実数に変換します。
        """
        if self.df is None:
            self.df = self._load_csv_from_s3()
            if 'dates' in self.df.columns:
                self.df = self.df.rename(columns={'dates': 'date'})
            self.df['date'] = pd.to_datetime(self.df['date'])

        for col in self.df.columns:
            if self.df[col].dtype == 'object' and self.df[col].str.contains('%').any():
                self.df[col] = self.df[col].str.rstrip('%').astype('float') / 100.0

        # インフレ関連変数の追加
        self._addInflationFeature()

    def _load_csv_from_s3(self) -> pd.DataFrame:
        """
        S3からCSVファイルを読み込んでデータフレームとして返します。
        """
        s3 = boto3.client('s3')
        try:
            obj = s3.get_object(Bucket=self.bucket_name, Key=self.file_name)
            df = pd.read_csv(obj['Body'], parse_dates=['dates'])
        except Exception as e:
            print(f"S3からのデータ読み込みエラー: {e}")
            df = pd.DataFrame()
        return df

    def _addInflationFeature(self) -> None:
        """
        インフレ関連の特徴量を生成し合成する。
        """
        def addInflationFeatureImpl(df: pd.DataFrame, yr: int) -> pd.DataFrame:
            """
            インフレ関連の変数を作成する
            """
            df[f'DFd_{yr}'] = np.exp(-df[f'JPYZeroRate{yr}Y']*yr)
            df[f'DFf_{yr}'] = np.exp(-df[f'USDZeroRate{yr}Y']*yr)
            df[f'DFI_{yr}'] = np.exp((df[f'JPYBEI{yr}Y']-df[f'USDBEI{yr}Y'])*yr)
            df[f'XI_{yr}'] = df[f'DFd_{yr}']/df[f'USDJPYIV{yr}YATM']*(df[f'DFf_{yr}']/df[f'DFd_{yr}'] - df[f'DFI_{yr}'])
            # 補助変数の削除
            df.drop([f'DFd_{yr}', f'DFf_{yr}', f'DFI_{yr}'], axis=1, inplace=True)
            return df

        # 5y,10yのインフレ関連変数を追加
        self.df = addInflationFeatureImpl(self.df, 5)
        self.df = addInflationFeatureImpl(self.df, 10)

        # 10yを優先して欠損値を補い合成変数を作成
        self.df['XI'] = self.df['XI_10'].combine_first(self.df['XI_5'])

        # 合成変数作成用の補助変数を削除
        self.df.drop(['XI_10', 'XI_5'], axis=1, inplace=True)

    def remove_missing_values(self, subset: list) -> Self:
        """
        指定した列の欠損値を削除した新しいS3DataFrameを返します。
        """
        rtndf = self.df.copy()
        rtndf.dropna(subset=subset, inplace=True)
        return S3DataFrame(self.bucket_name, self.file_name, rtndf)

    def calculate_moving_average(self, headers: List[str], n: int) -> Self:
        """
        指定した列の移動平均を計算し、新しい列を追加した新しいS3DataFrameを返します。
        """
        df = self.df.copy()
        for header in headers:
            new_header = f'{header}_MA_{n}'
            df[new_header] = df[header].rolling(window=n).mean()

        return S3DataFrame(self.bucket_name, self.file_name, df)

    def calculate_returns(self, headers: List[str], n: int, is_shift: bool = False) -> Self:
        """
        指定した列のリターンを計算し、新しいデータフレームを含む新しいS3DataFrameを返します。
        """
        rtndf = self.df.copy()
        for header in headers:
            new_header = f'{header}_Return_{n}'
            rtndf[new_header] = rtndf[header].pct_change(periods=n)
            if is_shift:
                new_header_shft = f'{header}_Return_{n}_Shift'
                rtndf[new_header_shft] = rtndf[new_header].shift(-n)
                rtndf.drop(new_header, axis=1, inplace=True)

        for header in headers:
            new_header = f'{header}_Return_{n}'
            if new_header in rtndf.columns:
                rtndf.dropna(subset=new_header, inplace=True)

            new_header += '_Shift'
            if new_header in rtndf.columns:
                rtndf.dropna(subset=new_header, inplace=True)

        return S3DataFrame(self.bucket_name, self.file_name, rtndf)
    
    @staticmethod
    def genDataFrameObjFromLocalFile(file_name: str) -> Self:
        """
        ローカルファイルからデータフレームを生成する。
        """
        df = pd.read_csv(file_name)
        if 'dates' in df.columns:
            df = df.rename(columns={'dates': 'date'})
        
        df['date'] = pd.to_datetime(df['date'])

        return S3DataFrame('', '', df)
    

#分析用のデータクラス
@dataclass
class DataConverter:
    """
    DataConverterクラスは、S3DataFrameを特定の形式に変換するためのクラスです。
    また、特定の日付を基準にデータをフィルタリングし、訓練データとテストデータを取得する機能も提供します。
    """
    s3df: S3DataFrame
    ndays: int
    feature_set: List[str]
    rtn_set_w_shft: Optional[List[str]] = None
    rtn_set_wo_shft: Optional[List[str]] = None

    def __post_init__(self):
        self.target_set = ['date', 'Premium']
        self.df = self._convert()
        self.s3df = None

    def _convert(self) -> pd.DataFrame:
        """
        S3DataFrameを特定の形式に変換します。
        """
        if self.rtn_set_w_shft is not None:
            self.s3df = self.s3df.calculate_returns(headers=self.rtn_set_w_shft, n=self.ndays, is_shift=True)
        if self.rtn_set_wo_shft is not None:
            self.s3df = self.s3df.calculate_returns(headers=self.rtn_set_wo_shft, n=self.ndays, is_shift=False)

        df = self.s3df.df.copy()
        df['DFd'] = np.exp(-df['JPYZeroRate1M']*self.ndays/250.0)
        df['DFf'] = np.exp(-df['USDZeroRate1M']*self.ndays/250.0)
        df['Premium'] = df['DFd']/df['USDJPYIV1MATM'] * (df['DFf']/df['DFd'] - (1.0 + df[f'USDJPY_Return_{self.ndays}_Shift']))

        return df

    def get_base_dates(self, start_date: str, end_date: str) -> pd.Series:

        get_base_df: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: (df[self.target_set].merge(df[self.feature_set], on='date'))
        drop_na: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df.dropna(subset = df.columns)
        get_base_dates_impl: Callable[[pd.DataFrame], pd.Series] = lambda df: df[(df['date'] >= start_date) & (df['date'] < end_date)]['date']

        return get_base_dates_impl(drop_na(get_base_df(self.df)))

    def _filter_by_date(self, base_date: Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定した日付を基準にデータをフィルタリングします。
        """
        return self.df[self.df['date'] <= base_date], self.df[self.df['date'] > (base_date + pd.Timedelta(days=self.ndays))]

    def get_train_test_df(self, base_date: Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        指定した日付を基準に訓練データとテストデータを取得します。
        """
        train_df, test_df = self._filter_by_date(base_date)
        train_rtn = train_df[self.target_set].merge(train_df[self.feature_set], on='date')
        test_rtn = test_df[self.target_set].merge(test_df[self.feature_set], on='date')

        return train_rtn.dropna(subset=train_rtn.columns), test_rtn.dropna(subset=test_rtn.columns)


# S3DataFrameとヘッダーを引数に取り、時系列とヒストグラムをプロットする関数
def plot_time_series_and_histogram(dc: S3DataFrame, header: Dict[str, str]) -> None:
    # 年を降順にソートし、それぞれの年に対して新しいデータフレームを作成
    for yr in sorted(dc.df['date'].dt.year.unique(), reverse=True):
        new_df = dc.df[dc.df['date'].dt.year == yr]
        # 3つのサブプロットを持つ新しいフィギュアを作成
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 新しいデータフレームとヘッダーの'A'列、最初のサブプロット、そして年を引数に取り、時系列をプロット
        plot_time_series(new_df, header['A'], axes[0], yr)
        # 新しいデータフレームとヘッダーの'B'列、2番目のサブプロット、そして年を引数に取り、時系列をプロット
        plot_time_series(new_df, header['B'], axes[1], yr)
        # 新しいデータフレームとヘッダーの'C'列、3番目のサブプロット、そして年を引数に取り、ヒストグラムをプロット
        plot_histogram(new_df, header['C'], axes[2], yr)

        # サブプロット間の間隔を調整
        plt.tight_layout()
        # フィギュアを表示
        plt.show()


# データフレーム、列名、軸、そして年を引数に取り、時系列をプロットする関数
def plot_time_series(df: pd.DataFrame, column: str, ax: plt.Axes, year: int) -> None:
    # 日付と列の値をプロット
    ax.plot(df['date'], df[column])
    # x軸のラベルを設定
    ax.set_xlabel('Date')
    # y軸のラベルを設定
    ax.set_ylabel(column)
    # タイトルを設定
    ax.set_title(f'{year} Time Series of ' + column)
    # グリッドを表示
    ax.grid(True)


# データフレーム、列名、軸、そして年を引数に取り、ヒストグラムをプロットする関数
def plot_histogram(df: pd.DataFrame, column: str, ax: plt.Axes, year: int) -> None:
    # 列の値のヒストグラムをプロット
    ax.hist(df[column], bins=20, alpha=0.5)
    # x軸のラベルを設定
    ax.set_xlabel(column)
    # y軸のラベルを設定
    ax.set_ylabel('Frequency')
    # タイトルを設定
    ax.set_title(f'{year} Histogram of ' + column)
    # グリッドを表示
    ax.grid(True)
