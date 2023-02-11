import logging
import numpy as np

logger = logging.getLogger(__name__)


def MultiRMSE(y, y_pred):
    s = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            s += (y[i, j] - y_pred[i, j]) ** 2
    return (s / y.shape[0]) ** (0.5)


def multiindex__to_index(columns):
    return ["__".join([str(j[0]), str(j[1])]) for j in columns]


def clear_nan_target(X, y, target_name):
    ind_nan = (~y[target_name].isna()).sum(axis=1) == len(
        target_name
    )  # можно вернуть, например, для chain

    return X[ind_nan], y.loc[ind_nan, target_name], ind_nan


def fit_boosting(
    target_name,
    train,
    y_train,
    valid,
    y_valid,
    test,
    y_test,
    dataset_wrap_obj,
    model,
    quality_func,
    **fit_params,
):  # например - plot=True, early_stopping_rounds=50, logging_level='Silent',
    """
    Простая утилита для обучения, например, катбуста длянашей задачи -
    фильтрует таргет по нанам,сохраняет множество, на котором обучился
    и высчитал статистики и модель
    """
    temp = dict()
    train_temp, y_train_temp, ind_nan_train = clear_nan_target(
        train, y_train, target_name
    )
    valid_temp, y_valid_temp, ind_nan_valid = clear_nan_target(
        valid, y_valid, target_name
    )
    test_temp, y_test_temp, ind_nan_test = clear_nan_target(
        test, y_test, target_name
    )

    train_boosting = dataset_wrap_obj(train_temp, y_train_temp)
    valid_boosting = dataset_wrap_obj(valid_temp, y_valid_temp)

    model.fit(train_boosting, eval_set=valid_boosting, **fit_params)
    temp["fitted_model"] = model
    temp["train"] = (train_temp, y_train_temp)
    temp["valid"] = (valid_temp, y_valid_temp)
    temp["test"] = (test_temp, y_test_temp)
    temp["score"] = quality_func(y_test_temp.values, model.predict(test_temp))
    logger.info(f"for {target_name}: {temp['score']}")
    return temp


def inverse_ang(y, x):
    add = np.zeros(y.shape)
    add[((y > 0) & (x < 0)) | ((y < 0) & (x < 0))] = 180
    add[(y < 0) & (x > 0)] = 360
    return np.rad2deg(np.arctan(y / x)) + add


def get_square_by_mas_val(mas_val):
    if mas_val <= 100:
        return 25, 45
    if mas_val > 100 and mas_val <= 200:
        return 35, 60
    if mas_val > 200 and mas_val <= 500:
        return 210, 25
    if mas_val > 500 and mas_val <= 760:
        return 210, 38
    if mas_val > 760 and mas_val <= 1010:
        return 210, 50
    if mas_val > 1010 and mas_val <= 1260:
        return 210, 62
    if mas_val > 1260 and mas_val <= 1510:
        return 210, 75
    if mas_val > 1510 and mas_val <= 1760:
        return 210, 88
    if mas_val > 1760 and mas_val <= 2010:
        return 210, 100
    if mas_val > 2010 and mas_val <= 2260:
        return 210, 112
    if mas_val > 2260 and mas_val <= 2500:
        return 210, 125
    return (
        999,
        999,
    )  # На всякий случай заглушечные значения, потому что теоретически мы их можем выдать
