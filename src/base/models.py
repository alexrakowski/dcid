import os
from abc import abstractmethod, ABC

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
import pytorch_lightning as pl


class BaseModel(pl.LightningModule, ABC):
    def __init__(
            self,
            lr: float = 1e-4,
            batch_size: int = 64,
            num_workers=os.environ.get('SLURM_JOB_CPUS_PER_NODE', 1),
            val_ratio=.15,
            log_hyperparams=False,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=log_hyperparams)

        # initialize all needed  objects
        self._create_dataset()
        self._create_splits()
        self._create_backbones()

    @abstractmethod
    def _create_dataset(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_splits(self):
        raise NotImplementedError()

    @abstractmethod
    def _create_backbones(self):
        raise NotImplementedError()

    def _get_dataloader(
            self,
            dataset,
            batch_size=None,
            **kwargs,
    ):
        return torch_data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            **kwargs,
        )

    def train_dataloader(self, shuffle=True):
        dloader = self._get_dataloader(dataset=self.dataset_train, shuffle=shuffle)
        return dloader

    def val_dataloader(self):
        dloader = self._get_dataloader(dataset=self.dataset_val)
        return dloader

    def test_dataloader(self):
        dloader = self._get_dataloader(dataset=self.dataset_test)
        return dloader

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.step(batch, batch_idx, optimizer_idx, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step_name='val')

    def test_step(self, batch, batch_idx, *args, **kwargs):
        return self.step(batch, batch_idx, step_name='test')

    def predict_step(self, batch, batch_idx, *args, **kwargs):
        return self.step(batch, batch_idx, step_name='predict')

    def _step(
            self,
            x,
            y1,
            y2,
            optimizer_idx=0,
            step_name='',
    ):
        logits, _ = self.forward(x)

        mse_1 = F.mse_loss(logits[:, 0], y1)
        mse_2 = F.mse_loss(logits[:, 1], y2)

        loss = (mse_1 + mse_2) / 2
        logs = {
            'loss': loss,
            'mse_1': mse_1,
            'mse_2': mse_2,
        }

        return loss, logs

    def step(
            self,
            batch,
            batch_idx,
            optimizer_idx=0,
            step_name='',
    ):
        x, y = batch

        x = x.float()
        y = y.float()
        y1, y2 = self.make_targets(y)

        if step_name == 'predict':
            return self.forward(x), y

        loss, logs = self._step(
            x=x,
            y1=y1,
            y2=y2,
            optimizer_idx=optimizer_idx,
            step_name=step_name,
        )

        if 'test' not in step_name:
            if loss is None or torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
                return None

        self.log_dict(
            {f'{step_name}/{k}': v for k, v in logs.items()},
            on_step=step_name == 'train',
            on_epoch=step_name != 'train',
            prog_bar=True,
        )

        return logs

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=float(self.hparams.lr),
        )

    @abstractmethod
    def make_targets(self, y):
        raise NotImplementedError()

    @abstractmethod
    def make_S_shared_S_indiv(self, y):
        raise NotImplementedError()

    @abstractmethod
    def compute_dcid_metrics(
            self,
            features_val,
            features_test,
            Y_val,
            S_shared_test,
            S_indiv_test,
            Y_shared_test,
    ):
        raise NotImplementedError()


def multi_task_feature_learning(
        X_val,
        Y_val,
        gamma,
        max_iter=10,
        eps=1e-4,
        eps_W=1e-4,
        max_n=2,
):
    """
    Argyriou, Andreas, Theodoros Evgeniou, and Massimiliano Pontil.
    "Multi-task feature learning." Advances in neural information processing systems 19 (2006).

    Based on the matlab implementation: https://github.com/argyriou/multi_task_learning/
    """

    def f_method(S):
        S = S.copy()
        idxs_bigger = S > eps
        S[idxs_bigger] = 1 / S[idxs_bigger]

        return S

    def D_min_method(D):
        return D / sum(D)

    if max_n is not None and max_n < X_val.shape[0]:
        idxs = np.random.choice(X_val.shape[0], max_n, replace=False)
        X_val, Y_val = X_val[idxs], Y_val[idxs]

    if isinstance(X_val, torch.Tensor):
        X_val = X_val.detach().cpu().numpy()
    if isinstance(Y_val, torch.Tensor):
        Y_val = Y_val.detach().cpu().numpy()

    T = 2
    X = X_val
    Y = Y_val

    d = X.shape[1]
    D = np.eye(d)

    W = np.ones((d, T))

    U, S, _ = np.linalg.svd(D)
    fS = f_method(S)
    fS = np.sqrt(fS)
    idxs_bigger = fS > eps
    fS[idxs_bigger] = 1 / fS[idxs_bigger]
    fD_isqrt = U @ np.diag(fS) @ U.T

    i = 0
    converged = False
    while not converged and i < max_iter:
        W_old = W.copy()

        X_D = X @ fD_isqrt
        K = X_D @ X_D.T
        for t in range(T):
            n = K.shape[0]
            y = Y[:, t]
            A = np.linalg.inv(K + gamma * np.eye(n)) @ y
            W[:, t] = X_D.T @ A

        W_D = W
        W = fD_isqrt @ W

        U, S, _ = np.linalg.svd(W)
        S_ = np.zeros(W.shape[0])
        S_[:S.shape[0]] = S
        S = S_
        Smin = D_min_method(S)
        D = U * np.diag(Smin) * U.T

        fS = f_method(Smin)
        fS = np.sqrt(fS)
        idxs_bigger = fS > eps
        fS[idxs_bigger] = 1 / fS[idxs_bigger]
        fD_isqrt = U @ np.diag(fS) @ U.T

        converged = ((W - W_old) ** 2).sum() < eps_W
        i += 1

    return fD_isqrt, W_D


class BaseDeepCCA:
    def forward(self, x):
        features_1 = self.encoder_1(x)
        preds_1 = self.classifier_1(features_1)

        features_2 = self.encoder_2(x)
        preds_2 = self.classifier_2(features_2)

        return torch.cat([preds_1, preds_2], dim=1), torch.cat([features_1, features_2], dim=1)
