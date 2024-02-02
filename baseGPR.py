# ガウス過程回帰のユーティリティ関数

# 必要なライブラリのインポート
import torch
import gpytorch as gp

from typing import Tuple, List


# ガウス過程回帰モデルの定義
# 観測誤差も考慮するためにlikelihoodを定義する
class GPModel(gp.models.ExactGP):
    """
    ガウス過程モデルを表すクラスです。

    Attributes:
        mean_module (gpytorch.means.Mean): 平均モジュール
        covar_module (gpytorch.kernels.ScaleKernel): 共分散モジュール
    """

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gp.likelihoods.Likelihood):
        """
        GPModelのコンストラクタです。

        Args:
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            likelihood (gpytorch.likelihoods.Likelihood): モデルの尤度関数
        """
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None

    def set_mean_module(self, mean_module: gp.means.Mean):
        """
        平均モジュールを設定します。

        Args:
            mean_module (gpytorch.means.Mean): 設定する平均モジュール
        """
        self.mean_module = mean_module

    def set_covar_module(self, covar_module):
        """
        共分散モジュールを設定します。

        Args:
            covar_module (gpytorch.kernels.ScaleKernel): 設定する共分散モジュール
        """
        self.covar_module = covar_module

    def set_spectral_mixture_kernel(self, num_mixtures: int, train_x: torch.Tensor, train_y: torch.Tensor):
            """
            SpectralMixtureKernelを設定します。

            Args:
                num_mixtures (int): 混合数
                train_x (torch.Tensor): 訓練データの入力
                train_y (torch.Tensor): 訓練データの出力
            """
            self.covar_module = gp.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=train_x.shape[1])
            self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x: torch.Tensor) -> gp.distributions.MultivariateNormal:
        """
        GPModelの順伝播を行います。

        Args:
            x (torch.Tensor): 入力データ

        Returns:
            gpytorch.distributions.MultivariateNormal: 多変量正規分布
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def standard_initialize(train_x: torch.Tensor, 
                            train_y: torch.Tensor, 
                            device: str) -> Tuple['GPModel', gp.likelihoods.Likelihood, 'GPTrainer']:
        """
        標準設定でGPModelを初期化します。

        Args:
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            device (str): モデルに使用するデバイス('cpu' or 'cuda')

        Returns:
            Tuple['GPModel', gpytorch.likelihoods.Likelihood]: 初期化されたGPModelと尤度関数
        """
        likelihood = gp.likelihoods.GaussianLikelihood()

        model = GPModel(train_x, train_y, likelihood)
        model.set_mean_module(gp.means.ConstantMean())
        model.set_covar_module(gp.kernels.ScaleKernel(gp.kernels.ScaleKernel(gp.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]))))

        likelihood = likelihood.to(device)
        model = model.to(device)

        return model, likelihood, GPTrainer()

    @staticmethod
    def keoops_Matern_initialize(train_x: torch.Tensor,
                                 train_y: torch.Tensor,
                                 nu: float,
                                 device: str) -> Tuple['GPModel', gp.likelihoods.Likelihood, 'GPTrainer']:
        """
        標準設定でGPModelを初期化します。

        Args:
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            device (str): モデルに使用するデバイス('cpu' or 'cuda')

        Returns:
            Tuple['GPModel', gpytorch.likelihoods.Likelihood]: 初期化されたGPModelと尤度関数
        """
        likelihood = gp.likelihoods.GaussianLikelihood()

        model = GPModel(train_x, train_y, likelihood)
        model.set_mean_module(gp.means.ConstantMean())
        model.set_covar_module(gp.kernels.ScaleKernel(gp.kernels.ScaleKernel(gp.kernels.keops.MaternKernel(nu=nu, ard_num_dims=train_x.shape[1]))))

        likelihood = likelihood.to(device)
        model = model.to(device)

        return model, likelihood, GPTrainer()
    
    @staticmethod
    def experiment_initialize(train_x: torch.Tensor,
                             train_y: torch.Tensor,
                             nu: float,
                             device: str) -> Tuple['GPModel', gp.likelihoods.Likelihood, 'GPTrainer']:
        """
        標準設定でGPModelを初期化します。

        Args:
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            device (str): モデルに使用するデバイス('cpu' or 'cuda')

        Returns:
            Tuple['GPModel', gpytorch.likelihoods.Likelihood]: 初期化されたGPModelと尤度関数
        """
        likelihood = gp.likelihoods.GaussianLikelihood()

        model = GPModel(train_x, train_y, likelihood)
        model.set_mean_module(gp.means.ConstantMean())
        
        if nu == 0.0:
            model.set_covar_module(gp.kernels.ScaleKernel(gp.kernels.RBFKernel(ard_num_dims=train_x.shape[1])) 
                                   + gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel(ard_num_dims=train_x.shape[1])))
        else:
            model.set_covar_module(gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu, ard_num_dims=train_x.shape[1])) 
                                   + gp.kernels.ScaleKernel(gp.kernels.PeriodicKernel(ard_num_dims=train_x.shape[1])))

        likelihood = likelihood.to(device)
        model = model.to(device)

        return model, likelihood, GPTrainer()

    @staticmethod
    def spectral_initialize(train_x: torch.Tensor,
                            train_y: torch.Tensor,
                            device: str, 
                            num_mixtures: int = 4) -> Tuple['GPModel', gp.likelihoods.Likelihood, 'GPTrainer']:
        """
        SpectralMixtureKernelを用いてGPModelを初期化します。

        Args:
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            device (str): モデルに使用するデバイス('cpu' or 'cuda')
            num_mixtures (int, optional): 混合数. Defaults to 4.

        Returns:
            Tuple['GPModel', gpytorch.likelihoods.Likelihood]: 初期化されたGPModelと尤度関数
        """
        model, likelihood, trainer = GPModel.standard_initialize(train_x, train_y, device)
        model.set_spectral_mixture_kernel(num_mixtures, train_x, train_y)

        return model, likelihood, trainer


# ガウス過程回帰モデルの学習
class GPTrainer:
    def __init__(self):
        """
        ガウス過程回帰モデルの学習を行うクラス
        """
        pass

    def train(self, 
              model: GPModel,
              likelihood: gp.likelihoods.Likelihood,
              train_x: torch.Tensor, 
              train_y: torch.Tensor, 
              n_iter: int = 100, 
              lr: float = 0.1) -> List[float]:
        """
        ガウス過程回帰モデルの学習

        Args:
            model (GPModel): ガウス過程回帰モデル
            likelihood (gpytorch.likelihoods.Likelihood): ライクリフッド関数
            train_x (torch.Tensor): トレーニングの入力データ
            train_y (torch.Tensor): トレーニングのターゲットデータ
            n_iter (int, optional): 学習のイテレーション数. Defaults to 100.
            lr (float, optional): 学習率. Defaults to 0.1.
        Returns:
            List[float]: 損失のリスト
        """
        self.model = model
        self.likelihood = likelihood

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # 損失のリストを初期化
        losses = []

        # モデルの学習
        for i in range(n_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            # 損失をリストに追加
            losses.append(loss.item())

        return losses

    def predict(self, test_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ガウス過程回帰モデルの予測

        Args:
            test_x (torch.Tensor): テストの入力データ

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 平均と分散のタプル
        """
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            observed_pred = self.likelihood(self.model(test_x))

        # observed_predから平均と分散を取得
        mean = observed_pred.mean
        variance = observed_pred.variance

        return mean.detach().cpu(), variance.detach().cpu()
