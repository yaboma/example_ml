import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging
import databases
import os
from src.prepare_utils import (
    get_full_index_names,
    explode_freqs,
    prepare_value_x_y,
    KEYS,
    saved_values,
    get_stats,
    add_diff_appendix,
)
from src.schema import Experiment, Predict, PredictRecord
from src.models_utils import (
    multiindex__to_index,
    inverse_ang,
    get_square_by_mas_val,
)
import asyncio
import concurrent.futures


from fastapi import FastAPI

from fastapi.responses import JSONResponse
from src.model.predicted_mas import mas_table

# Засетапим логгер
logging.basicConfig(
    format="%(name)s - %(asctime)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

PATH_MODEL = "artefacts/model_mas.pkl"

with open(PATH_MODEL, "rb") as f:
    models_mas = pickle.load(f)

# Инициализируем все для коннекта к БД
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_NAME = os.environ.get("DB_NAME")
DB_HOST = os.environ.get("DB_HOST")


SERVER_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"


def create_app(SERVER_URL):
    database = databases.Database(SERVER_URL)

    app = FastAPI(
        title="Mas calculation",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup():
        await database.connect()

    async def create_predict_record(predict: PredictRecord):
        # У нас одинаково называются поля в БД и предикт рекорд, можно хинтануть через **
        query = mas_table.insert().values(**predict.dict())
        result = await database.execute(query)
        return result

    @app.on_event("shutdown")
    async def shutdown():
        await database.disconnect()

    @app.get("/healthcheck")
    def healthcheck():
        return {"STATUS": "OK"}

    @app.post("/calculate", response_model=Predict)
    async def calculation(experiment: Experiment):
        """Get mas value and angle by first experiment"""
        logger.info(f"Read experiments for {experiment.nrot}")
        one_experiment = pd.read_json(experiment.bien_json)
        one_experiment["nrot"] = experiment.nrot
        one_experiment["rotor_id"] = 0
        one_experiment["num_experiments"] = 0
        one_experiment["test_descr"] = "НЧ-испытание"  # пока мокаем
        dt = datetime.now()
        one_experiment["dt"] = dt
        one_experiment.set_index(
            ["nrot", "rotor_id", "num_experiments", "test_descr", "dt"],
            inplace=True,
        )
        logger.info(f"Explode freqs for {experiment.nrot}")
        one_experiment = explode_freqs(one_experiment)
        one_experiment = prepare_value_x_y(
            one_experiment,
            vectorized_cols=[
                "po_bien",
                "vt_bien",
            ],
        )
        logger.info(f"Stats calc for {experiment.nrot}")
        stats_experiment = get_stats(one_experiment)
        lf_bien = pd.pivot(
            one_experiment.query("test_descr == 'НЧ-испытание'"),
            columns="freq",
            values=saved_values,
            index=KEYS,
        )
        dataset = pd.merge(
            lf_bien,
            stats_experiment,
            left_index=True,
            right_index=True,
            how="left",
        )
        dataset.drop(
            [("stats", "monotonic_po_bien"), ("stats", "monotonic_vt_bien")],
            axis=1,
            inplace=True,
        )
        logger.info("Stats diff features")
        dataset = dataset.join(add_diff_appendix(dataset))
        PREDICTOR_COLS = get_full_index_names(
            dataset.columns,
            [
                "po_bien_x",
                "po_bien_y",
                "vt_bien_x",
                "vt_bien_y",
                "stats",
                "po_bien_x_diff",
                "po_bien_y_diff",
                "vt_bien_x_diff",
                "vt_bien_y_diff",
                "stuff",
            ],
        )
        dataset = dataset[PREDICTOR_COLS]
        dataset.columns = multiindex__to_index(dataset.columns)
        # инференс
        logger.info(f"Inference for {experiment.nrot}")
        result = dict()
        loop = asyncio.get_event_loop()
        for target_name, model in models_mas.items():
            name = target_name[0].split("__")[-1].replace("-", "_")

            try:
                with concurrent.futures.ThreadPoolExecutor() as pool:

                    res = await loop.run_in_executor(
                        pool, model.predict, dataset
                    )
                    res = res[0]
            except Exception as e:
                logger.error(str(e))
                res = [[np.nan, np.nan]]

            mas_val = np.sqrt(res[0] ** 2 + res[1] ** 2)
            mas_ang = inverse_ang(res[1], res[0])
            length, height = get_square_by_mas_val(mas_val)
            result[name] = {
                "mas_val": round(mas_val),
                "mas_ang": round(mas_ang) % 360,
                "square": f"{length}*{height}",
            }
        kwargs = dict()
        for names in zip(
            ["Верх", "Низ", "Середина", "Середина_верх", "Середина_низ"],
            ["up", "down", "middle", "middle_up", "middle_down"],
        ):
            mas_val = result.get(names[0], {}).get("mas_val")
            mas_ang = result.get(names[0], {}).get("mas_ang")
            kwargs[f"mas_val_{names[1]}"] = mas_val
            kwargs[f"mas_ang_{names[1]}"] = mas_ang
            if mas_val is not None:
                length, height = get_square_by_mas_val(mas_val)
                kwargs[f"mas_length_{names[1]}"] = length
                kwargs[f"mas_height_{names[1]}"] = height
        db_record = PredictRecord(
            nrot=experiment.nrot, dt=dt, num_experiments=0, **kwargs
        )
        logger.info(f"Writing to DB for {experiment.nrot}")
        _ = await create_predict_record(db_record)
        logger.info(f"Finished for {experiment.nrot}")
        return JSONResponse(content=result)

    return app


app = create_app(SERVER_URL)
