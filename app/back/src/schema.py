from typing import Dict, Optional, Union
from pydantic import BaseModel
from datetime import datetime


class Experiment(BaseModel):
    nrot: str
    bien_json: str


class Predict(BaseModel):
    Верх: Dict[str, Union[str, int]]
    Низ: Dict[str, Union[str, int]]
    Середина: Dict[str, Union[str, int]]
    Середина_низ: Optional[Dict[str, Union[str, int]]]
    Середина_верх: Optional[Dict[str, Union[str, int]]]


class PredictRecord(BaseModel):
    nrot: str
    dt: datetime
    num_experiments: int
    mas_val_up: int
    mas_ang_up: int
    mas_length_up: int
    mas_height_up: int
    mas_val_down: int
    mas_ang_down: int
    mas_length_down: int
    mas_height_down: int
    mas_val_middle: int
    mas_ang_middle: int
    mas_length_middle: int
    mas_height_middle: int
    mas_val_middle_up: Optional[int] = None
    mas_ang_middle_up: Optional[int] = None
    mas_length_middle_up: Optional[int] = None
    mas_height_middle_up: Optional[int] = None
    mas_val_middle_down: Optional[int] = None
    mas_ang_middle_down: Optional[int] = None
    mas_length_middle_down: Optional[int] = None
    mas_height_middle_down: Optional[int] = None
