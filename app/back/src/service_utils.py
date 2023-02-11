import re
import pandas as pd
from datetime import datetime


def get_name(bien_txt_path_name):
    nrot = re.search("\d{4}-\d{4}", bien_txt_path_name)
    return "unknowm" if nrot is None else nrot.group()


def prepare_row_txt(row_txt):
    splitted = [float(i.replace(",", ".")) for i in row_txt.split("\t")]
    return {int(splitted[0]): splitted[1:]}


def get_bien(bien_txt_path_name):
    nrot = get_name(bien_txt_path_name)
    rotor_id = 0  # пока мокаем,
    num_experiments = 0
    test_descr = "НЧ-испытание"  # пока мокаем
    dt = datetime.now()
    res = dict()
    with open(bien_txt_path_name, "r", encoding="cp1251") as fin:
        fin.readline()
        for row in fin:
            res.update(prepare_row_txt(row))
    res = pd.DataFrame.from_dict(res, orient="index").reset_index()
    res.columns = [
        "freq_bien",
        "po_bien_val",
        "po_bien_ang",
        "vt_bien_val",
        "vt_bien_ang",
    ]
    res["nrot"] = nrot
    res["rotor_id"] = rotor_id
    res["num_experiments"] = num_experiments
    res["test_descr"] = test_descr
    res["dt"] = dt

    return res.set_index(
        ["nrot", "rotor_id", "num_experiments", "test_descr", "dt"]
    )
