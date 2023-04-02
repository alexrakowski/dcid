import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import cca_zoo.probabilisticmodels

from base.models import multi_task_feature_learning


def compute_metrics(
        features,
        S_shared,
        S_indiv,
        Y_shared,
):
    R2_S_shared = 0
    R2_S_indiv = 0
    R2_Y_shared_1 = 0
    R2_Y_shared_2 = 0

    if features.shape[1] > 0:
        features = np.nan_to_num(features)
        # apply PCA on features to make them independent
        features = PCA(n_components=.999).fit_transform(features)

        R2_S_shared = R2_score(X=S_shared, y=features)
        R2_S_indiv = R2_score(X=features, y=S_indiv)

        Y_shared_1, Y_shared_2 = Y_shared
        R2_Y_shared_1 = R2_score(X=features, y=Y_shared_1)
        R2_Y_shared_2 = R2_score(X=features, y=Y_shared_2)

    return {
        'R2_S_shared': R2_S_shared,
        'R2_S_indiv': R2_S_indiv,
        'R2_Y_shared_1': R2_Y_shared_1,
        'R2_Y_shared_2': R2_Y_shared_2,
    }


def normalize(X):
    X = X - X.mean(axis=0)
    X = X / X.std(axis=0)
    X = np.nan_to_num(X)

    return X


def R2_score(
        X,
        y,
        normalize_vars=False,
        test_ratio=None,
):
    assert len(X) == len(y)
    if len(X) < 2:
        return np.nan

    if X.ndim == 1:
        X = np.expand_dims(X, 1)
    if normalize_vars:
        X, y = normalize(X), normalize(y)

    X_test, y_test = X, y
    if test_ratio is not None:
        num_test = max(int(test_ratio * len(X)), 1)
        X_test, y_test = X[:num_test], y[:num_test]
        X, y = X[num_test:], y[num_test:]

    reg = LinearRegression().fit(X, y)
    score = reg.score(X_test, y_test)

    return score


def mtfl_metrics(
        features_val,
        features_test,
        Y_val,
        S_shared_test,
        S_indiv_test,
        Y_shared_test,
):
    rows = []
    for gamma in np.logspace(-4, 1, 10):
        fD_isqrt, W_D = multi_task_feature_learning(
            X_val=features_val,
            Y_val=Y_val,
            gamma=gamma,
        )
        W_D = W_D / np.abs(W_D).max()
        for cutoff in np.linspace(0, 1, 10):
            idxs_shared = (np.abs(W_D) > cutoff).min(axis=1)
            learned_S_shared_test = (features_test @ fD_isqrt)[:, idxs_shared]

            row = compute_metrics(
                features=learned_S_shared_test,
                S_shared=S_shared_test,
                S_indiv=S_indiv_test,
                Y_shared=Y_shared_test,
            )
            row['gamma'] = gamma
            row['cutoff_W'] = cutoff
            rows.append(row)

    df = pd.DataFrame(rows)
    df['method'] = 'mtfl'

    return df


def W_cutoff_metrics(
        features_val,
        features_test,
        Y_val,
        S_shared_test,
        S_indiv_test,
        Y_shared_test,
        W,
):
    W = W / np.abs(W).max()

    rows = []
    for cutoff in np.linspace(0, 1, 10):
        idxs_shared = (np.abs(W) > cutoff).sum(axis=0) == 2
        learned_S_shared_test = features_test[:, idxs_shared]

        row = compute_metrics(
            features=learned_S_shared_test,
            S_shared=S_shared_test,
            S_indiv=S_indiv_test,
            Y_shared=Y_shared_test,
        )
        row['cutoff_W'] = cutoff
        rows.append(row)

    df = pd.DataFrame(rows)
    df['method'] = 'W_cutoff'

    return df


def cca_metrics(
        features_val,
        features_test,
        Y_val,
        S_shared_test,
        S_indiv_test,
        Y_shared_test,
):
    assert features_val.shape[1] % 2 == 0
    dim_latent = features_val.shape[1] // 2
    features_1, features_2 = features_val[:, :dim_latent], features_val[:, dim_latent:]

    pca_model_1 = PCA(n_components=.999).fit(features_1)
    pca_model_2 = PCA(n_components=.999).fit(features_2)
    pca_1, pca_2 = pca_model_1.transform(features_1), pca_model_2.transform(features_2)

    min_n_components = min(pca_model_1.n_components_, pca_model_2.n_components_)

    cca = CCA(n_components=min_n_components)
    cca = cca.fit(pca_1, pca_2)
    cca_1, cca_2 = cca.transform(pca_1, pca_2)

    cc = np.corrcoef(cca_1, cca_2, rowvar=False)

    rows = []
    for threshold_cc in np.linspace(0, 1, 10):
        last_big_cc = min_n_components
        for i in range(min_n_components):
            if cc[i, i + min_n_components] < threshold_cc:
                last_big_cc = i
                break

        features_1_test, features_2_test = features_test[:, :dim_latent], features_test[:, dim_latent:]
        pca_1_test, pca_2_test = pca_model_1.transform(features_1_test), pca_model_2.transform(features_2_test)
        cca_1_test, cca_2_test = cca.transform(pca_1_test, pca_2_test)
        cca_avg_pruned = (cca_1_test[:, :last_big_cc] + cca_2_test[:, :last_big_cc]) / 2

        row = compute_metrics(
            features=cca_avg_pruned,
            S_shared=S_shared_test,
            S_indiv=S_indiv_test,
            Y_shared=Y_shared_test,
        )
        row['threshold_cc'] = threshold_cc
        rows.append(row)

    df = pd.DataFrame(rows)
    df['method'] = 'CCA'

    return df


def pcca_metrics(
        features_val,
        features_test,
        Y_val,
        S_shared_test,
        S_indiv_test,
        Y_shared_test,
):
    assert features_val.shape[1] % 2 == 0
    dim_latent = features_val.shape[1] // 2
    features_1, features_2 = features_val[:, :dim_latent], features_val[:, dim_latent:]
    features_1_test, features_2_test = features_test[:, :dim_latent], features_test[:, dim_latent:]
    features_1 = features_1.detach().cpu().numpy()
    features_2 = features_2.detach().cpu().numpy()
    features_1_test = features_1_test.detach().cpu().numpy()
    features_2_test = features_2_test.detach().cpu().numpy()

    min_batch_size = min(features_1.shape[0], features_1_test.shape[0])
    features_1 = features_1[:min_batch_size]
    features_2 = features_2[:min_batch_size]
    features_1_test = features_1_test[:min_batch_size]
    features_2_test = features_2_test[:min_batch_size]

    cca = cca_zoo.probabilisticmodels.ProbabilisticCCA(latent_dims=dim_latent, num_warmup=25, num_samples=5)
    cca.fit([features_1, features_2])
    Z = cca.transform([features_1_test, features_2_test]).mean(0)
    ccs = cca.pairwise_correlations([features_1_test, features_2_test])[0][0]

    rows = []
    for threshold_cc in np.linspace(0, 1, 10):
        idxs = [cc >= threshold_cc for cc in ccs]
        row = compute_metrics(
            features=Z[:, idxs],
            S_shared=S_shared_test,
            S_indiv=S_indiv_test,
            Y_shared=Y_shared_test,
        )
        row['threshold_cc'] = threshold_cc
        rows.append(row)

    df = pd.DataFrame(rows)
    df['method'] = 'PCCA'

    return df
