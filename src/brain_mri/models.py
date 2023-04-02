import shutil
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

import datasets
from base import models as base_models
import base.architectures as base_architectures
from base.utils import process_column


class _BaseBrainMRIModel(base_models.BaseModel):
    def __init__(
            self,
            df_path,
            y1_col_name,
            y2_col_name,
            dim_latent_space=16,
            img_size=96,
            slice_dim=None,
            arch_cls='MobileNetV2',
            arch_width_mult=2,
            skip_missing=False,
            dataset_source='adni',
            y1_correct_by=None,
            y2_correct_by=None,
            second_visit_scenario=False,
            cache_dir=None,
            *args,
            **kwargs,
    ):
        cache_dir = None if cache_dir == '' else cache_dir

        super().__init__(*args, **kwargs)

        df_path = Path(df_path)
        assert df_path.exists()

        self.save_hyperparameters(logger=kwargs['log_hyperparams'])

    def _create_dataset(self):
        df = pd.read_csv(self.hparams.df_path)

        if self.hparams.skip_missing:
            df = df.loc[df['path'].apply(lambda p: Path(p).exists())]

        if self.hparams.dataset_source == 'adni':
            df['ABETA'] = df['ABETA'].apply(lambda s: float(str(s).replace('<', '').replace('>', '')))
            df['CDGLOBAL_x'] = df['CDGLOBAL_x'].clip(0, 1)
            df['Sex'] = (df['Sex'] == 'M').astype(int)
            df['TAU'] = df['TAU'].apply(lambda a: str(a).replace('>', '').replace('<', '')).astype(float)
            df['PTAU'] = df['PTAU'].apply(lambda a: str(a).replace('>', '').replace('<', '')).astype(float)
            df['TAU_by_ABETA'] = df['TAU'] / df['ABETA']
            df['CI_ordinal'] = df['Group'].apply(lambda g: 2 if g == 'AD' else (1 if 'MCI' in g else 0))
        elif self.hparams.dataset_source == 'ukbb':
            df['Subject'] = df['eid']
        else:
            raise ValueError(f'Unknown dataset source: {self.hparams.dataset_source}')

        df = df.loc[
            (~df[self.hparams.y1_col_name].isna())
            & (~df[self.hparams.y2_col_name].isna())
            ]

        if self.hparams.y2_correct_by is not None:
            df_cn = df.loc[df['Group'] == 'CN'] if self.hparams.dataset_source == 'adni' else df
            x, y = df_cn[self.hparams.y2_correct_by].values, df_cn[self.hparams.y2_col_name].values
            lm = LinearRegression().fit(x, y)
            df[self.hparams.y2_col_name] -= lm.predict(df[self.hparams.y2_correct_by].values)

        if self.hparams.y1_correct_by is not None:
            df_cn = df.loc[df['Group'] == 'CN'] if self.hparams.dataset_source == 'adni' else df
            x, y = df_cn[self.hparams.y1_correct_by].values, df_cn[self.hparams.y1_col_name].values
            lm = LinearRegression().fit(x, y)
            df[self.hparams.y1_col_name] -= lm.predict(df[self.hparams.y1_correct_by].values)

        df[self.hparams.y1_col_name], self.y1_values = process_column(df[self.hparams.y1_col_name])
        df[self.hparams.y2_col_name], self.y2_values = process_column(df[self.hparams.y2_col_name])

        if self.hparams.cache_dir is not None:
            print(f'Caching dataset to {self.hparams.cache_dir}')
            df['path_cached'] = df['path'].apply(lambda p: Path(self.hparams.cache_dir) / Path(*(Path(p).parts[1:])))
            for _, row in tqdm(df.iterrows(), mininterval=10):
                path_cached = row['path_cached']
                path = Path(row['path'])

                i = 0
                while i < 3:
                    try:
                        if not path_cached.exists() or path_cached.stat().st_size != path.stat().st_size:
                            print(f'Copying {str(path)}')
                            path_cached.parent.mkdir(exist_ok=True, parents=True)
                            shutil.copy(str(path), str(path_cached))
                        break
                    except Exception as e:
                        if i > 3:
                            raise e
                        print(e)
                        sleep(1)
                        i += 1
            df['path'] = df['path_cached']

        self.df = df

    def _create_splits_df(self):
        if self.hparams.second_visit_scenario:
            assert self.hparams.dataset_source == 'ukbb'

            df_test = self.df[~self.df['ventricles-3.0'].isna()]
            df_train = self.df[self.df['ventricles-3.0'].isna()]

            patients = df_test['Subject'].unique()
            np.random.shuffle(patients)
            num_val = len(patients) // 2
            patients_val, patients_test = patients[:num_val], patients[num_val:]

            print(f'Subjects:\nTrain: {len(df_train)}\nVal: {len(patients_val)}\nTest: {len(patients_test)}\n')

            df_test_2nd_visit = df_test.copy()
            df_test_2nd_visit['path'] = df_test_2nd_visit['path'].apply(lambda p: str(p).replace('_2_0', '_3_0'))
            if self.hparams.cache_dir is not None:
                print(f'Caching dataset to {self.hparams.cache_dir}')

                df_test_2nd_visit['path'] = df_test_2nd_visit['path'].apply(
                    lambda p: f"/{str(p).replace(self.hparams.cache_dir, '')}"
                )

                df_test_2nd_visit['path_cached'] = df_test_2nd_visit['path'].apply(
                    lambda p: Path(self.hparams.cache_dir) / Path(*(Path(p).parts[1:])))
                for _, row in tqdm(df_test_2nd_visit.iterrows(), mininterval=10):
                    path_cached = row['path_cached']
                    path = Path(row['path'])
                    if not path_cached.exists() or path_cached.stat().st_size != path.stat().st_size:
                        print(f'Copying {str(path)}')
                        path_cached.parent.mkdir(exist_ok=True, parents=True)
                        shutil.copy(str(path), str(path_cached))
                df_test_2nd_visit['path'] = df_test_2nd_visit['path_cached']

            df_test = pd.concat([df_test, df_test_2nd_visit])

            return (
                df_train,
                df_test.loc[df_test['Subject'].isin(patients_val)],
                df_test.loc[df_test['Subject'].isin(patients_test)],
            )

        patients = self.df['Subject'].unique()
        np.random.shuffle(patients)
        num_val = int(self.hparams.val_ratio * len(patients))
        patients_val, patients_test, patients_train = patients[:num_val], patients[num_val:2 * num_val], patients[
                                                                                                         2 * num_val:]

        print(f'Subjects:\nTrain: {len(patients_train)}\nVal: {len(patients_val)}\nTest: {len(patients_test)}\n')

        return (self.df.loc[self.df['Subject'].isin(patients_train)],
                self.df.loc[self.df['Subject'].isin(patients_val)],
                self.df.loc[self.df['Subject'].isin(patients_test)]
                )

    def _create_splits(self):
        self.df_train, self.df_val, self.df_test = self._create_splits_df()
        print(f'Samples:\nTrain: {len(self.df_train)}\nVal: {len(self.df_val)}\nTest: {len(self.df_test)}')

        dsets = []
        for df in (self.df_train, self.df_val, self.df_test):
            dsets.append(datasets.BrainMRIDataset(
                df=df,
                y1_col_name=self.hparams.y1_col_name,
                y2_col_name=self.hparams.y2_col_name,
                img_size=self.hparams.img_size,
                slice_dim=self.hparams.slice_dim,
            ))
        self.dataset_train, self.dataset_val, self.dataset_test = dsets

    def make_targets(self, y):
        y1, y2 = y[:, 0], y[:, 1]
        return y1, y2

    def make_S_shared_S_indiv(self, y):
        """Dummy method"""
        raise NotImplementedError()


class DeepCCA(base_models.BaseDeepCCA, _BaseBrainMRIModel):
    def _create_backbones(self):
        encoder_cls = getattr(base_architectures, self.hparams.arch_cls)

        self.encoder_1 = encoder_cls(
            in_channels=1,
            last_channel=self.hparams.dim_latent_space,
            width_mult=self.hparams.arch_width_mult,
        )
        self.encoder_2 = encoder_cls(
            in_channels=1,
            last_channel=self.hparams.dim_latent_space,
            width_mult=self.hparams.arch_width_mult,
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
        raise NotImplementedError()
