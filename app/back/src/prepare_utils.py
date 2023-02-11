import numpy as np
import pandas as pd
import multiprocessing as mp
from numpy.polynomial.polynomial import polyfit
from scipy.optimize import curve_fit
import itertools
from inspect import signature
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

tqdm.pandas()


saved_values = [
    "po_bien_x",
    "po_bien_y",
    "vt_bien_x",
    "vt_bien_y",
    "po_bien_val",
    "po_bien_ang",
    "vt_bien_val",
    "vt_bien_ang",
]

OFFSET = 10
KEYS = ["rotor_id", "nrot", "test_descr", "num_experiments", "dt"]
CPU_COUNT = mp.cpu_count()
# Подготовим частотыдля НЧ и ВЧ
lf_list = np.arange(149, 650, 50)
hf_list = np.array([1400, *np.arange(1499, 1800, 50)])
# Служебные константы для делания признаков
TOLERANCE_FREQ = 10
VAL_COLUMNS = ["po_bien_val", "po_bien_ang", "vt_bien_val", "vt_bien_ang"]
MAX_VALID_LEN_ONE_EXPERIMENT = 12


def is_float(x):
    try:
        _ = float(x)
        return True
    except ValueError:
        return False


def first_needed_index(x, values):
    try:
        return np.where(np.isin(x, values))[0][0]
    except IndexError:
        return np.nan


def last_needed_index(x, values):
    try:
        return np.where(np.isin(x, values))[0][-1]
    except IndexError:
        return np.nan


def to_binned(array, base=2):
    return (array // base + 1 * ((array % base) >= base / 2)) * base


def inverse_ang(y, x):
    add = np.zeros(y.shape)
    add[((y > 0) & (x < 0)) | ((y < 0) & (x < 0))] = 180
    add[(y < 0) & (x > 0)] = 360
    return np.rad2deg(np.arctan(y / x)) + add


def degree_func(x, a, b):
    return a * x**b


def exp_func(x, a, b):
    return a * b**x


def get_linear_coef(x, y):
    return polyfit(x, y, deg=1)[-1]


def get_ordered_bool_indexes(size):
    indexes = np.array(list(itertools.product([True, False], repeat=size)))
    sums_nd = np.argsort([-sum(i) for i in indexes])
    return indexes[sums_nd]


INDEXES_STORAGE = {
    i: get_ordered_bool_indexes(i) for i in range(MAX_VALID_LEN_ONE_EXPERIMENT)
}


def get_monotonic_sequence(indexes, one_experiments_df, monotonic_name):
    """
    брутфорсим по упорядоченным по числу оставленных элементов индексам, находим самую длинную последовательность
    (не оченьизящно, зато пишется быстрее динамики)
    возвращаем последовательности для дальнейшего обучения кривых на только монотонных данных
    """
    for i in indexes:
        if one_experiments_df.loc[i, monotonic_name].is_monotonic_increasing:
            res = i
            break
    y = one_experiments_df.loc[res, monotonic_name].values

    x = one_experiments_df.loc[res, "freq"].values
    return y, x, res


def get_curve_coefs(func, x, y):
    L = len(signature(func).parameters) - 1
    popt = [np.nan] * L
    try:
        popt, _ = curve_fit(
            func,
            x,
            y,
        )
    except (RuntimeError, TypeError):
        pass
    return popt


def get_unachived_stats(one_experimetns_rows):
    result = dict()
    lost_freq = np.where(one_experimetns_rows["po_bien_val"].isna())[0]
    get_freq = np.where(~one_experimetns_rows["po_bien_val"].isna())[0]
    result["num_passed_tests"] = len(lost_freq)
    result["num_achieved_tests"] = one_experimetns_rows.shape[0] - len(
        lost_freq
    )
    result["num_tests"] = len(one_experimetns_rows)
    result["lost_freq"] = [
        one_experimetns_rows.iloc[i]["freq"] for i in lost_freq
    ]
    if len(get_freq):
        result["max_achieved_test"] = get_freq[-1] + 1  # чтобы нумерация с 1
        result["max_achieved_freq"] = one_experimetns_rows.iloc[get_freq[-1]][
            "freq"
        ]
    return result


def get_all_coefs(one_experiments_df, name_val, indexes):
    """
    one_experiments_df - датафрейм с частотами изначениями отдельного эксперимента
    name_val - vt или po
    indexes - упорядоченные индексы нужной длины
    """
    result = dict()
    # Здесь нужно оставить только по физике биений биения
    # с ростом частоты растет значение, иначе велик риск получить неадекват
    # наклон мы и так поймем по векторам
    y, x, res = get_monotonic_sequence(
        indexes, one_experiments_df, f"{name_val}_bien_val"
    )
    result[f"monotonic_{name_val}_bien"] = res
    result[f"max_monotonic_{name_val}_bien"] = sum(res)
    popt = get_curve_coefs(degree_func, x, y)
    result[f"{name_val}_bien_val_degree_A"] = popt[0]
    result[f"{name_val}_bien_val_degree_B"] = popt[1]
    # интреполяция по A*(b^x)
    popt = get_curve_coefs(exp_func, x / np.max(x), y)
    result[f"{name_val}_bien_val_exp_A"] = popt[0]
    result[f"{name_val}_bien_val_exp_B"] = popt[1]

    # Получаем линейный коэффициенты
    result[f"linear_coef_{name_val}_bien_val_only_increase"] = get_linear_coef(
        one_experiments_df.loc[res, f"{name_val}_bien_x"],
        one_experiments_df.loc[res, f"{name_val}_bien_y"],
    )
    result[f"linear_coef_{name_val}_bien_val"] = get_linear_coef(
        one_experiments_df[f"{name_val}_bien_x"],
        one_experiments_df[f"{name_val}_bien_y"],
    )

    # Насколько сильно разлетаются углы
    result[f"std_{name_val}_ang"] = one_experiments_df[
        f"{name_val}_bien_ang"
    ].std()

    # Доля сонаправленных углов
    med = one_experiments_df[f"{name_val}_bien_ang"].median()
    result[f"{name_val}_part_nearly_median"] = np.mean(
        (one_experiments_df[f"{name_val}_bien_ang"] < med + OFFSET)
        & (one_experiments_df[f"{name_val}_bien_ang"] > med - OFFSET)
    )
    return result


def calc_stats_bien_for_full_freq(
    one_experimetns_rows, is_need_unachived=False
):
    """
    по каждому испытанию каждого ротора рассчитывает статистики - валидно для НЧ и ВЧ
    """
    result = dict()
    # Достанем нужные как есть фичи
    for i in KEYS:
        result[i] = one_experimetns_rows[i].iloc[0]
    # Блок про недостижимые частоты
    if is_need_unachived:
        result |= get_unachived_stats(one_experimetns_rows)

    one_experiments_df = one_experimetns_rows.dropna(
        subset=["po_bien_val", "po_bien_ang", "vt_bien_val", "vt_bien_ang"]
    )
    if not len(one_experiments_df):
        return result
    result["num_valid_freqs"] = len(one_experiments_df)

    #     indexes = get_ordered_bool_indexes(len(one_experiments_df))
    indexes = INDEXES_STORAGE[len(one_experiments_df)]
    # Получаем все нужные коэффициенты
    for name_val in ["po", "vt"]:
        result |= get_all_coefs(one_experiments_df, name_val, indexes)

    return result


def apply_parallel(dfGrouped, func, total=None, **inner_func_kwargs):
    retLst = Parallel(n_jobs=CPU_COUNT)(
        delayed(func)(group, **inner_func_kwargs)
        for name, group in tqdm(dfGrouped, total=total)
    )
    return pd.DataFrame(retLst)


def get_full_index_names(multiindex, filter_seq):

    if isinstance(filter_seq, str):
        return list(multiindex[multiindex.get_loc(filter_seq)].values)
    elif isinstance(filter_seq, list):
        ind = []
        for i in filter_seq:
            # ToDo некрасиво,
            ind.extend(multiindex[multiindex.get_loc(i)].values)
        return ind
    else:
        raise NotImplementedError("unknown type of filter seq")


def explode_freqs(bien_data):
    # Мокает недостигнутые частоты
    # keys по смыслу должен быть номером испытания
    bien_data = bien_data.copy()

    mock_freq = pd.DataFrame(
        index=bien_data.groupby(level=bien_data.index.names).groups.keys()
    )
    mock_freq.index.names = bien_data.index.names

    mock_freq = mock_freq.assign(
        freq=[
            hf_list if i == "ВЧ-испытание" else lf_list
            for i in mock_freq.index.get_level_values("test_descr")
        ]
    ).explode("freq")
    mock_freq["freq"] = mock_freq["freq"].astype(int)
    mock_freq.sort_values(["freq"], inplace=True)
    bien_data.sort_values(["freq_bien"], inplace=True)
    bien_data["num_rows"] = np.arange(0, bien_data.shape[0])
    bien_data = pd.merge_asof(
        mock_freq,
        bien_data,
        left_on="freq",
        right_on="freq_bien",
        by=bien_data.index.names,
        suffixes=("_rotors", "_bien"),
        direction="nearest",
    )
    # обработка дубликатов
    ind_first_dupl = bien_data.duplicated("num_rows", keep="first")

    bien_data.loc[ind_first_dupl, VAL_COLUMNS] = np.nan
    # Теперь нужно убрать то, что приджонилось странно - например,
    # сейчас мы можем получить 621 приджойненный к 645, а технически это неправильно,
    # Поэтому все, что приджойнилось, но имеет разницу в частотах больше TOLERANCE_FREQ
    # стоит считать не таким же по смыслу
    bien_data.loc[
        (np.abs(bien_data.freq - bien_data.freq_bien) > TOLERANCE_FREQ),
        VAL_COLUMNS,
    ] = np.nan

    return bien_data.drop(columns="num_rows")


def prepare_value_x_y(
    bien_data,
    binned_cols=["po_bien_val", "vt_bien_val"],
    vectorized_cols=["po_bien", "vt_bien", "mas"],
):
    bien_data = bien_data.copy()
    # огрубляем до 2 микрон
    for i in binned_cols:
        bien_data[i] = to_binned(bien_data[i], 2)
    # Преобразуем в векторы
    for j in vectorized_cols:
        bien_data[f"{j}_x"] = bien_data[f"{j}_val"] * np.cos(
            np.radians(bien_data[f"{j}_ang"])
        )
        bien_data[f"{j}_y"] = bien_data[f"{j}_val"] * np.sin(
            np.radians(bien_data[f"{j}_ang"])
        )
    return bien_data


def get_stats(bien_data, is_parallel=False):

    df_group = bien_data.groupby(KEYS)
    if is_parallel:
        result_stats_df = apply_parallel(
            df_group,
            calc_stats_bien_for_full_freq,
            total=len(df_group),
            is_need_unachived=False,
        )
    else:
        result_stats_df = pd.DataFrame(
            [
                calc_stats_bien_for_full_freq(group, is_need_unachived=False)
                for name, group in df_group
            ]
        )

    result_stats_df.set_index(
        KEYS,
        inplace=True,
    )
    result_stats_df.columns = pd.MultiIndex.from_product(
        [["stats"], result_stats_df.columns]
    )
    return result_stats_df


def add_diff_appendix(full_dataset):
    buf = []
    stuff = pd.DataFrame(
        [
            full_dataset.index.get_level_values("num_experiments").values,
            full_dataset.index.get_level_values("test_descr").map(
                {"НЧ-испытание": 0, "ВЧ-испытание": 1}
            ),
        ]
    ).T
    stuff.index = full_dataset.index
    stuff.columns = ["num_tests", "test_code"]
    for i in ["po_bien", "vt_bien"]:
        idxmax = 1799 - full_dataset[i + "_val"].idxmax(axis="columns")
        stuff[i + "_idxmax_freq"] = idxmax
        idxmin = 1799 - full_dataset[i + "_val"].idxmin(axis="columns")
        stuff[i + "_idxmin_freq"] = idxmin

        for j in ["_x", "_y"]:
            name = i + j
            temp = full_dataset[name].diff(axis=1).drop(149, axis=1)
            equal_number = temp.apply(
                lambda x: len(x) - len(set(np.round(np.abs(x), 4))) + 1, axis=1
            )
            stds = temp.std(axis=1)

            mean_low = full_dataset[
                [(name, 149), (name, 199), (name, 249), (name, 299)]
            ].mean(axis=1)
            mean_middle = full_dataset[
                [(name, 349), (name, 399), (name, 449), (name, 499)]
            ].mean(axis=1)
            mean_high = full_dataset[
                [(name, 549), (name, 599), (name, 649)]
            ].mean(axis=1)
            temp["stds"] = stds
            temp["num_equal_diffs"] = equal_number
            temp["mean_low"] = mean_low
            temp["mean_middle"] = mean_middle
            temp["mean_high"] = mean_high
            temp.columns = pd.MultiIndex.from_product(
                [[name + "_diff"], temp.columns]
            )
            buf.append(temp)

    stuff.columns = pd.MultiIndex.from_product([["stuff"], stuff.columns])
    buf.append(stuff)
    return pd.concat(buf, axis=1)
