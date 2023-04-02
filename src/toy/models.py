import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F

import base.models as base_models
import base.architectures as base_architectures
import base.metrics as base_metrics
import base.utils as utils
import datasets

NUM_LABELS = 6


class _ScenarioFactory:
    @staticmethod
    def _scenario_1_by_1(scenario_id, **kwargs):
        """
        1 individual variable per Y1/Y2, 1 shared variable.
        Thus we have ${6 \\choose 2} = 15$ different individual variable subsets, times 4 different shared variables.
        """
        assert 0 <= scenario_id <= 59

        idxs = list(range(NUM_LABELS))
        idxs_indiv = list(itertools.combinations(idxs, 2))[scenario_id // 4]
        idxs_shared = [i for i in idxs if i not in idxs_indiv][scenario_id // 15]

        weights = np.zeros((2, NUM_LABELS))
        weights[0][idxs_indiv[0]] = 1
        weights[1][idxs_indiv[1]] = 1
        weights[:, idxs_shared] = 1

        return weights

    @staticmethod
    def _scenario_decreasing_contributions_1(scenario_id, scenario_contribution_ratio, **kwargs):
        """
        1 individual variable per Y1/Y2, 4 shared variables.
        Thus we have ${6 \\choose 2} = 15$ different individual variable subsets.
        The weight for each individual variable is set to 1, the weights for the shared variables are set to
        sqrt(contribution_ratio / 4).
        """
        assert scenario_contribution_ratio is not None
        assert 0 < scenario_contribution_ratio
        assert 0 <= scenario_id <= 14

        idxs = list(range(NUM_LABELS))
        idxs_indiv = list(itertools.combinations(idxs, 2))[scenario_id]
        idxs_shared = [i for i in idxs if i not in idxs_indiv]

        weights = np.zeros((2, NUM_LABELS))
        weights[0][idxs_indiv[0]] = 1
        weights[1][idxs_indiv[1]] = 1
        weights[:, idxs_shared] = np.sqrt(scenario_contribution_ratio / (NUM_LABELS - 2))

        return weights

    @staticmethod
    def _scenario_differing_contributions_1(scenario_id, scenario_contribution_ratio, **kwargs):
        """
        1 individual variable per Y1/Y2, 1 shared variable.
        Thus we have ${6 \\choose 2} = 15$ different individual variable subsets, times 4 different shared variables.
        For Y1 the individual variable explains 1 - scenario_contribution_ratio of the variance, and the shared variable
        explains scenario_contribution_ratio of the variance.
        For Y2 the individual variable explains scenario_contribution_ratio of the variance, and the shared variable
        explains 1 - scenario_contribution_ratio of the variance.
        """
        assert scenario_contribution_ratio is not None
        assert 0 < scenario_contribution_ratio < 1
        assert 0 <= scenario_id <= 59

        idxs = list(range(NUM_LABELS))
        idxs_indiv = list(itertools.combinations(idxs, 2))[scenario_id // 4]
        idxs_shared = [i for i in idxs if i not in idxs_indiv][scenario_id // 15]

        weights = np.zeros((2, NUM_LABELS))
        weights[0][idxs_indiv[0]] = 1 - scenario_contribution_ratio
        weights[1][idxs_indiv[1]] = scenario_contribution_ratio
        weights[0, idxs_shared] = scenario_contribution_ratio
        weights[1, idxs_shared] = 1 - scenario_contribution_ratio
        # take the square root of the weights so that the variance sums to 1
        weights = np.sqrt(weights)

        return weights

    @staticmethod
    def get_scenario_weights(
            scenario_name,
            scenario_id,
            **kwargs,
    ):
        scenario_method = getattr(_ScenarioFactory, f'_scenario_{scenario_name}')
        return scenario_method(scenario_id, **kwargs)


class _BaseToyModel(base_models.BaseModel):
    def __init__(
            self,
            target_weights,
            dim_latent_space=16,
            dataset_in_memory=False,
            dropout=0,
            batch_norm=True,
            # experiment scenario arguments
            scenario_name=None,
            scenario_id=None,
            scenario_contribution_ratio=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(logger=kwargs['log_hyperparams'])

        if scenario_name is not None:
            assert scenario_id is not None
            target_weights = _ScenarioFactory.get_scenario_weights(
                scenario_name=scenario_name,
                scenario_id=scenario_id,
                scenario_contribution_ratio=scenario_contribution_ratio,
            )

        assert target_weights is not None
        self.target_weights = torch.FloatTensor(target_weights).to(self.device)
        assert self.target_weights.shape[0] == 2

    def _create_dataset(self):
        self.dataset = datasets.Shapes3dDataset(in_memory=self.hparams.dataset_in_memory)

    def _create_splits(self):
        self.dataset_train, self.dataset_val, self.dataset_test = torch_data.random_split(
            self.dataset,
            (1 - 2 * self.hparams.val_ratio, self.hparams.val_ratio, self.hparams.val_ratio),
        )

    def make_targets(self, y):
        with torch.no_grad():
            y1 = y @ self.target_weights[0].to(y.device)
            y2 = y @ self.target_weights[1].to(y.device)

        return y1, y2

    def make_S_shared_S_indiv(self, y):
        return y[:, self.idxs_shared], y[:, self.idxs_indiv]

    def make_Y_shared(self, y):
        """Create y1 and y2 with the individual variables set to 0."""
        y_shared = y.clone()
        y_shared[:, self.idxs_indiv] = 0

        return self.make_targets(y_shared)

    @property
    def idxs_shared(self):
        return (self.target_weights != 0).sum(dim=0) == 2

    @property
    def idxs_indiv(self):
        return (self.target_weights != 0).sum(dim=0) < 2


# TODO move shared functionality to base.models
class MultiTask(_BaseToyModel):
    def _create_backbones(self):
        self.encoder = base_architectures.DlibCNN(
            in_channels=3,
            last_channel=self.hparams.dim_latent_space,
        )
        self.classifier = nn.Linear(
            in_features=self.hparams.dim_latent_space,
            out_features=2,
        )

    def forward(self, x):
        features = self.encoder(x)
        preds = self.classifier(features)

        return preds, features

    def compute_dcid_metrics(
            self,
            features_val,
            features_test,
            Y_val,
            S_shared_test,
            S_indiv_test,
            Y_shared_test,
    ):
        W = self.classifier.weight.data.cpu().numpy()
        metrics_W_cutoff = base_metrics.W_cutoff_metrics(
            features_val=features_val,
            features_test=features_test,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
            W=W,
        )

        metrics_mtfl = base_metrics.mtfl_metrics(
            features_val=features_val,
            features_test=features_test,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
        )

        return pd.concat([metrics_W_cutoff, metrics_mtfl])


class DeepCCA(base_models.BaseDeepCCA, _BaseToyModel):
    def _create_backbones(self):
        self.encoder_1 = base_architectures.DlibCNN(
            in_channels=3,
            last_channel=self.hparams.dim_latent_space,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
        )
        self.encoder_2 = base_architectures.DlibCNN(
            in_channels=3,
            last_channel=self.hparams.dim_latent_space,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
        )
        self.classifier_1 = nn.Linear(
            in_features=self.hparams.dim_latent_space,
            out_features=1,
        )
        self.classifier_2 = nn.Linear(
            in_features=self.hparams.dim_latent_space,
            out_features=1,
        )

    def compute_dcid_metrics(
            self,
            features_val,
            features_test,
            Y_val,
            S_shared_test,
            S_indiv_test,
            Y_shared_test,
    ):
        return base_metrics.cca_metrics(
            features_val=features_val,
            features_test=features_test,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
        )


class DeepPCCA(DeepCCA):
    def compute_dcid_metrics(
            self,
            features_val,
            features_test,
            Y_val,
            S_shared_test,
            S_indiv_test,
            Y_shared_test,
    ):
        return base_metrics.pcca_metrics(
            features_val=features_val,
            features_test=features_test,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
        )


class AdversarialMTL(_BaseToyModel):
    def __init__(
            self,
            lr_disc=None,
            num_hidden_disc=1,
            dim_hidden_disc=32,
            coeff_adversarial=1,
            coeff_ortho_shared=1,
            coeff_ortho_indiv=0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if lr_disc is None:
            lr_disc = self.hparams.lr
        self.save_hyperparameters(logger=kwargs['log_hyperparams'])

    def _create_backbones(self):
        self.encoder_1 = base_architectures.DlibCNN(
            in_channels=3,
            last_channel=self.hparams.dim_latent_space,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
        )
        self.encoder_2 = base_architectures.DlibCNN(
            in_channels=3,
            last_channel=self.hparams.dim_latent_space,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
        )
        self.classifier_1 = nn.Linear(
            in_features=self.hparams.dim_latent_space,
            out_features=1,
        )
        self.classifier_2 = nn.Linear(
            in_features=self.hparams.dim_latent_space,
            out_features=1,
        )

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=self.dim_shared_space, out_features=self.hparams.dim_hidden_disc),
            nn.ReLU(),
            *(nn.Sequential(
                nn.Linear(in_features=self.hparams.dim_hidden_disc, out_features=self.hparams.dim_hidden_disc),
                nn.ReLU(),
            ) for _ in range(self.hparams.num_hidden_disc)),
            nn.Linear(in_features=self.hparams.dim_hidden_disc, out_features=1),
        )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            set(
                list(self.encoder_1.parameters()) + list(
                    self.encoder_2.parameters()) + list(
                    self.classifier_1.parameters()) + list(
                    self.classifier_2.parameters()
                ),
            ),
            lr=float(self.hparams.lr),
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=float(self.hparams.lr_disc))
        return [opt_g, opt_d]

    def diff_loss(self, features_shared, features_indiv):
        features_shared = utils.normalize_samples(features_shared)
        features_indiv = utils.normalize_samples(features_indiv)

        return torch.linalg.matrix_norm(features_shared.T @ features_indiv) ** 2

    def _get_shared_features(self, features):
        return features[:, self.dim_shared_space:self.dim_shared_space * 2], features[:, self.dim_shared_space * 3:]

    def _step(
            self,
            x,
            y1,
            y2,
            optimizer_idx=0,
            step_name='',
    ):
        # train the prediction model
        if optimizer_idx == 0:
            logits, features = self.forward(x)

            mse_1 = F.mse_loss(logits[:, 0], y1)
            mse_2 = F.mse_loss(logits[:, 1], y2)

            features_shared_1, features_shared_2 = self._get_shared_features(features)
            features_shared = torch.cat((features_shared_1, features_shared_2), dim=0)
            preds_disc = self.discriminator(features_shared)
            targets = torch.zeros(size=(features_shared.shape[0], 1), device=features_shared.device)
            # set half of the targets to the opposite label
            targets[:targets.shape[0] // 2] += 1
            loss_adv = F.binary_cross_entropy_with_logits(input=preds_disc, target=targets)

            features_indiv_1 = features[:, :self.dim_shared_space]
            features_indiv_2 = features[:, self.dim_shared_space * 2:self.dim_shared_space * 3]
            loss_ortho_shared = self.diff_loss(features_shared_1, features_indiv_1) + self.diff_loss(features_shared_2,
                                                                                                     features_indiv_2)
            loss_ortho_indiv = self.diff_loss(features_indiv_1, features_indiv_2)

            loss = (mse_1 + mse_2) / 2
            loss = loss + self.hparams.coeff_adversarial * loss_adv
            loss = loss + self.hparams.coeff_ortho_shared * loss_ortho_shared
            loss = loss + self.hparams.coeff_ortho_indiv * loss_ortho_indiv
            logs = {
                'loss': loss,
                'mse_1': mse_1,
                'mse_2': mse_2,
                'loss_adv_gen': loss_adv,
                'loss_ortho_shared': loss_ortho_shared,
                'loss_ortho_indiv': loss_ortho_indiv,
            }

        # train the discriminator model
        if optimizer_idx == 1:
            with torch.no_grad():
                _, features = self.forward(x)

                features_shared_1, features_shared_2 = self._get_shared_features(features)
                features_shared = torch.cat((features_shared_1, features_shared_2), dim=0)
                targets = torch.zeros(size=(features_shared.shape[0], 1), device=features_shared.device)
                # set half of the targets to the correct label
                targets[targets.shape[0] // 2:] += 1

            preds_disc = self.discriminator(features_shared)
            loss_adv = F.binary_cross_entropy_with_logits(input=preds_disc, target=targets)

            loss = loss_adv
            logs = {
                'loss': loss,
                'loss_adv_disc': loss_adv,
            }

        return loss, logs

    def forward(self, x):
        features_1 = self.encoder_1(x)
        preds_1 = self.classifier_1(features_1)

        features_2 = self.encoder_2(x)
        preds_2 = self.classifier_2(features_2)

        return torch.cat([preds_1, preds_2], dim=1), torch.cat([features_1, features_2], dim=1)

    @property
    def dim_shared_space(self):
        return self.hparams.dim_latent_space // 2

    def compute_dcid_metrics(
            self,
            features_val,
            features_test,
            Y_val,
            S_shared_test,
            S_indiv_test,
            Y_shared_test,
    ):
        features_shared_1, features_shared_2 = self._get_shared_features(features_test)
        features_shared = (features_shared_1 + features_shared_2) / 2

        W = torch.cat([self.classifier_1.weight.data, self.classifier_2.weight.data], dim=0)
        W = W.detach().cpu().numpy()[:, self.dim_shared_space:]

        metrics = base_metrics.W_cutoff_metrics(
            features_val=features_val,
            features_test=features_shared,
            Y_val=Y_val,
            S_shared_test=S_shared_test,
            S_indiv_test=S_indiv_test,
            Y_shared_test=Y_shared_test,
            W=W,
        )
        metrics['method'] = 'adv_mtl'

        return metrics
