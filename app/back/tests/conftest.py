import os
from alembic import command
from alembic.config import Config
from sqlalchemy_utils import create_database, drop_database, database_exists
import pytest
from server import create_app, DB_USER, DB_PASS, DB_HOST


ALEMBIC_PATH = "./"


@pytest.fixture(scope="module")  #
def temp_app():
    SERVER_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/test_mas"
    if database_exists(SERVER_URL):
        drop_database(SERVER_URL)
    create_database(SERVER_URL)  # Создаем БД
    os.environ["DB_NAME"] = "test_mas"
    alembic_cfg = Config(
        os.path.join(ALEMBIC_PATH, "alembic.ini")
    )  # Загружаем конфигурацию alembic
    command.upgrade(alembic_cfg, "head")  # выполняем миграции
    app = create_app(SERVER_URL)
    try:
        yield app
    finally:
        drop_database(SERVER_URL)  # удаляем БД
