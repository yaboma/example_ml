import streamlit as st
import requests
import pandas as pd
import re


DF_COLUMNS = [
    "freq_bien",
    "po_bien_val",
    "po_bien_ang",
    "vt_bien_val",
    "vt_bien_ang",
]

SERVER_URL = "http://mas_calculator:8000/calculate"

SELECTBOX_TITLE = """
    Выберите вариант представления загружаемых данных"""

SELECTBOX_OPTIONS = ["Значения с дельтами", "Графики", "Обычная таблица"]

RESULT_TITLE = """
    Результаты расчета корректирующей массы"""


def parse_df(uploaded_file):
    df = pd.read_csv(uploaded_file, sep="\t", encoding="cp1251", decimal=",")
    # узнать, как формируются заголовки в исходном файле и в одинаковом ли порядке выгружаются данные
    df.columns = DF_COLUMNS
    return df


def represent_df(df):
    cols = st.columns(5)
    deltas = df.diff()
    for idx, row in df.iterrows():
        for i, col in enumerate(cols):
            delta = round(deltas.iloc[idx, i], 2)
            col.metric(df.columns[i], row[i], delta)


def parse_rotor_no(uploaded_file):
    try:
        return re.search(r"\d{4}-\d{4}", uploaded_file.name).group(0)
    except AttributeError:
        return None


def show_graphs(df):
    st.header(df.columns[0])
    st.line_chart(df.iloc[:, 0])
    for i, col in enumerate(st.columns(2)):
        with col:
            for j in (i + 1, i + 3):
                st.header(df.columns[j])
                st.line_chart(df.iloc[:, j])


def get_prediction(df, r_no):
    response = requests.post(
        SERVER_URL,
        json={"nrot": r_no, "bien_json": df.to_json()},
        timeout=8000,
        headers={"Content-type": "application/json", "Accept": "text/plain"},
    )
    return response


def run():
    st.sidebar.title("Меню")
    upload_label = "Загрузите файл с результатами испытания"
    uploaded_file = st.sidebar.file_uploader(upload_label)

    if uploaded_file:
        r_no = parse_rotor_no(uploaded_file)
        if r_no is None:
            st.error(
                """
                Не удалось извлечь номер ротора из имени загруженного файла,
                убедитесь, что в названии файла есть имя вида:"\\d{4}-\\d{4}"
                """
            )
        else:
            st.title(f"Испытания ротора № {r_no}")
            df = parse_df(uploaded_file)
            st.table(df)
            if st.sidebar.button("Рассчитать"):
                st.title(RESULT_TITLE)
                result = get_prediction(df, r_no)
                st.table(pd.DataFrame.from_dict(result.json()))
