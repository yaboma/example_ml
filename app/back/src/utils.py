import numpy as np
import pandas as pd
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import RegressorChain
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
import plotly.express as px
from abc import abstractmethod


COLORBLIND_COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def filter_cols(df, name_level_0):
    return [(name_level_0, i) for i in df[name_level_0].columns]


def flatten_columns(*dfs):
    is_multiindex = (
        isinstance(df.columns, pd.core.indexes.multi.MultiIndex) for df in dfs
    )
    if not all(is_multiindex):
        return (df.columns for df in dfs)
    return ([x[0] + "_" + str(x[1]) for x in df.columns] for df in dfs)


def Multirmse(y, y_pred):
    s = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            s += (y[i, j] - y_pred[i, j]) ** 2
    return (s / y.shape[0]) ** (0.5)


def split_data(X, y):
    """train, valid, test split 60/20/20"""
    train, valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.4, random_state=22, shuffle=True
    )
    test, valid, y_test, y_valid = train_test_split(
        valid, y_valid, test_size=0.5, random_state=22, shuffle=True
    )
    print(train.shape, test.shape, valid.shape)
    return train, valid, test, y_train, y_valid, y_test


def get_pools(*args, cat_cols=None):
    train, valid, test, y_train, y_valid, y_test = args
    train_pool = catboost.Pool(train, y_train, cat_features=cat_cols)
    valid_pool = catboost.Pool(valid, y_valid, cat_features=cat_cols)
    test_pool = catboost.Pool(test, y_test, cat_features=cat_cols)
    return train_pool, valid_pool, test_pool


def get_single_preds(multipred, y_test):
    """Returns: pandas.DataFrame with RMSEs for every target
    from multipred and from mean(y_true)
    """
    cols = [i + "_pred" for i in y_test.columns]
    pred = pd.DataFrame(multipred, columns=cols)
    data = []
    for i in y_test.columns:
        const_pred = [y_test[i].mean()] * y_test.shape[0]
        data.append(
            (
                mean_squared_error(
                    y_test[i], pred[i + "_pred"], squared=False
                ),
                mean_squared_error(y_test[i], const_pred, squared=False),
            )
        )
    df = pd.DataFrame(data, columns=["rmse_model", "rmse_mean"])
    df.index = y_test.columns
    return df


@abstractmethod
class CatBoostRegressorChain(RegressorChain):
    def fit(
        self, X, Y, X_valid=None, Y_valid=None, cat_features=None, **fit_params
    ):
        X, Y = np.array(X), np.array(Y)
        if X_valid is not None and Y_valid is not None:
            X_valid, Y_valid = np.array(X_valid), np.array(Y_valid)
        random_state = check_random_state(self.random_state)
        self.order_ = self.order
        if isinstance(self.order_, tuple):
            self.order_ = np.array(self.order_)

        if self.order_ is None:
            self.order_ = np.array(range(Y.shape[1]))
        elif isinstance(self.order_, str):
            if self.order_ == "random":
                self.order_ = random_state.permutation(Y.shape[1])
        elif sorted(self.order_) != list(range(Y.shape[1])):
            raise ValueError("invalid order")

        self.estimators_ = [
            clone(self.base_estimator) for _ in range(Y.shape[1])
        ]

        if self.cv is None:
            Y_pred_chain = Y[:, self.order_]
            X_aug = np.hstack((X, Y_pred_chain))
            if X_valid is not None and Y_valid is not None:
                Y_valid_chain = Y_valid[:, self.order_]
                X_valid_aug = np.hstack((X_valid, Y_valid_chain))

        for chain_idx, estimator in enumerate(self.estimators_):
            y_train = Y[:, self.order_[chain_idx]]
            x_train = X_aug[:, : (X.shape[1] + chain_idx)]
            train_pool = catboost.Pool(
                x_train, y_train, cat_features=cat_features
            )
            if X_valid is not None and Y_valid is not None:
                y_valid = Y_valid[:, self.order_[chain_idx]]
                x_valid = X_valid_aug[:, : (X_valid.shape[1] + chain_idx)]
                valid_pool = [
                    catboost.Pool(x_valid, y_valid, cat_features=cat_features)
                ]
            else:
                valid_pool = None

            estimator.fit(train_pool, eval_set=valid_pool, **fit_params)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = np.array(X)
        Y_pred_chain = np.zeros((X.shape[0], len(self.estimators_)))
        for chain_idx, estimator in enumerate(self.estimators_):
            previous_predictions = Y_pred_chain[:, :chain_idx]
            X_aug = np.hstack((X, previous_predictions))
            Y_pred_chain[:, chain_idx] = estimator.predict(X_aug)

        inv_order = np.empty_like(self.order_)
        inv_order[self.order_] = np.arange(len(self.order_))
        Y_pred = Y_pred_chain[:, inv_order]

        return Y_pred


def draw_experiments_by_rotor(chunck_rotor, y_name="po_bien_val"):
    """
    рисуем несколько экспериментов одного ротора для значений в привязке к частотам
    Ex:
    chunck_rotor = rotors_with_bien.query('nrot == @nrot_draw and test_descr == "НЧ-испытание"')
    draw_experiments_by_rotor(chunck_rotor, 'vt_bien_val')
    """
    fig = px.scatter(
        chunck_rotor,
        x="freq",
        y=y_name,
        color=chunck_rotor["num_experiments"].astype(str),
        hover_name="freq_bien",
        color_discrete_sequence=COLORBLIND_COLORS,
        trendline="lowess",
    )
    fig.show()


def draw_experiments_by_rotor3d(chunck_rotor, y_name="po"):
    """
    рисуем несколько экспериментов одного ротора в координатах х и у и z - частота вращения ротора
    Ex:
    chunck_rotor = rotors_with_bien.query('nrot == @nrot_draw and test_descr == "НЧ-испытание"')
    draw_experiments_by_rotor3d(chunck_rotor, y_name = 'vt')
    """
    fig = px.scatter_3d(
        chunck_rotor,
        x=f"{y_name}_bien_y",
        y=f"{y_name}_bien_x",
        z="freq",
        hover_name="freq_bien",
        color_discrete_sequence=COLORBLIND_COLORS,
        color=chunck_rotor["num_experiments"].astype(str),
    )

    fig.show()


def draw_experiments_by_rotor_x_y(chunck_rotor, name="po_bien"):
    """
    рисуем несколько экспериментов одного ротора в координатах х и у
    Ex:
    chunck_rotor = rotors_with_bien.query('nrot == @nrot_draw and test_descr == "НЧ-испытание"')
    draw_experiments_by_rotor_x_y(chunck_rotor)
    """
    fig = px.scatter(
        chunck_rotor,
        x=f"{name}_x",
        y=f"{name}_y",
        color=chunck_rotor["num_experiments"].astype(str),
        hover_name="freq_bien",
        color_discrete_sequence=COLORBLIND_COLORS,
    )
    fig.show()
