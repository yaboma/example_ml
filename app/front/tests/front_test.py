import pytest
from src.front_utils import parse_rotor_no
import os

TEST_FILES_PATH = "app/tests/test_files"


@pytest.mark.parametrize(
    "path_file, nrot",
    [
        (
            "2204-0667",
            os.path.join(TEST_FILES_PATH, "Позиция_ 5_Ротор № 2204-0667.txt"),
        ),
        ("2204-0358", os.path.join(TEST_FILES_PATH, "Ротор № 2204-0358.txt")),
        ("2204-1252", os.path.join(TEST_FILES_PATH, "Ротор № 2204-1252.txt")),
        (None, os.path.join(TEST_FILES_PATH, "wrong_name.txt")),
    ],
)
def test_parse_nrot(nrot, path_file):
    assert (
        parse_rotor_no(path_file) is None or parse_rotor_no(path_file) == nrot
    )


@pytest.mark.parametrize(
    "path_file, nrot",
    [
        (
            "2204-0667",
            os.path.join(TEST_FILES_PATH, "Позиция_ 5_Ротор № 2204-0667.txt"),
        ),
        ("2204-0358", os.path.join(TEST_FILES_PATH, "Ротор № 2204-0358.txt")),
        ("2204-1252", os.path.join(TEST_FILES_PATH, "Ротор № 2204-1252.txt")),
        (None, os.path.join(TEST_FILES_PATH, "wrong_name.txt")),
    ],
)
def test_parse_df(nrot, path_file):
    assert (
        parse_rotor_no(path_file) is None or parse_rotor_no(path_file) == nrot
    )
