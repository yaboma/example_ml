import pytest
import json
from fastapi.testclient import TestClient
from server import app


@pytest.mark.asyncio
async def test_health_check():
    with TestClient(app) as client:
        print(client)
        response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"STATUS": "OK"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nrot, bien_json, answer",
    [
        (
            "2205-0001",
            {
                "freq_bien": {
                    "0": 149,
                    "1": 200,
                    "2": 249,
                    "3": 299,
                    "4": 349,
                    "5": 399,
                    "6": 449,
                    "7": 499,
                    "8": 549,
                    "9": 600,
                    "10": 636,
                },
                "po_bien_val": {
                    "0": 38.8,
                    "1": 40.41,
                    "2": 42.94,
                    "3": 46.6,
                    "4": 51.12,
                    "5": 57.6,
                    "6": 66.26,
                    "7": 79.3,
                    "8": 98.92,
                    "9": 134.47,
                    "10": 180.14,
                },
                "po_bien_ang": {
                    "0": 222,
                    "1": 221,
                    "2": 221,
                    "3": 221,
                    "4": 221,
                    "5": 221,
                    "6": 220,
                    "7": 220,
                    "8": 220,
                    "9": 220,
                    "10": 220,
                },
                "vt_bien_val": {
                    "0": 60.1,
                    "1": 61.48,
                    "2": 63.68,
                    "3": 66.6,
                    "4": 70.56,
                    "5": 76.35,
                    "6": 84.49,
                    "7": 96.62,
                    "8": 116.2,
                    "9": 152.53,
                    "10": 201.1,
                },
                "vt_bien_ang": {
                    "0": 268,
                    "1": 266,
                    "2": 265,
                    "3": 263,
                    "4": 261,
                    "5": 258,
                    "6": 254,
                    "7": 251,
                    "8": 246,
                    "9": 242,
                    "10": 237,
                },
            },
            {
                "Верх": {"mas_val": 564, "mas_ang": 290, "square": "210*38"},
                "Низ": {"mas_val": 176, "mas_ang": 242, "square": "35*60"},
                "Середина": {
                    "mas_val": 1369,
                    "mas_ang": 236,
                    "square": "210*75",
                },
            },
        ),
        (
            "2204-0331",
            {
                "freq_bien": {
                    "0": 149,
                    "1": 199,
                    "2": 249,
                    "3": 299,
                    "4": 349,
                    "5": 399,
                    "6": 449,
                    "7": 500,
                    "8": 549,
                    "9": 599,
                    "10": 649,
                },
                "po_bien_val": {
                    "0": 38.17,
                    "1": 38.07,
                    "2": 38.86,
                    "3": 39.98,
                    "4": 41.71,
                    "5": 44.34,
                    "6": 48.65,
                    "7": 55.56,
                    "8": 67.45,
                    "9": 90.02,
                    "10": 139.54,
                },
                "po_bien_ang": {
                    "0": 252,
                    "1": 253,
                    "2": 256,
                    "3": 259,
                    "4": 263,
                    "5": 268,
                    "6": 274,
                    "7": 280,
                    "8": 288,
                    "9": 297,
                    "10": 305,
                },
                "vt_bien_val": {
                    "0": 48.06,
                    "1": 49.35,
                    "2": 51.33,
                    "3": 53.71,
                    "4": 57.17,
                    "5": 61.83,
                    "6": 68.6,
                    "7": 78.12,
                    "8": 93.06,
                    "9": 119.97,
                    "10": 176.32,
                },
                "vt_bien_ang": {
                    "0": 319,
                    "1": 318,
                    "2": 318,
                    "3": 318,
                    "4": 318,
                    "5": 319,
                    "6": 319,
                    "7": 319,
                    "8": 319,
                    "9": 320,
                    "10": 321,
                },
            },
            {
                "Верх": {"mas_val": 351, "mas_ang": 311, "square": "210*25"},
                "Низ": {"mas_val": 367, "mas_ang": 227, "square": "210*25"},
                "Середина": {
                    "mas_val": 1038,
                    "mas_ang": 302,
                    "square": "210*62",
                },
            },
        ),
        (
            "2204-2000",
            {
                "freq_bien": {"0": 150, "1": 170},
                "po_bien_val": {"0": 2.7, "1": 20.7},
                "po_bien_ang": {"0": 3, "1": 30},
                "vt_bien_val": {"0": 4, "1": 40},
                "vt_bien_ang": {"0": 5, "1": 50},
            },
            {
                "Верх": {"mas_val": 208, "mas_ang": 220, "square": "210*25"},
                "Низ": {"mas_val": 58, "mas_ang": 249, "square": "25*45"},
                "Середина": {
                    "mas_val": 475,
                    "mas_ang": 205,
                    "square": "210*25",
                },
            },
        ),
    ],
)
async def test_service(nrot, bien_json, answer, temp_app):

    request_data = {
        "nrot": nrot,
        "bien_json": json.dumps(bien_json),
    }
    with TestClient(temp_app) as client:
        response = client.post(
            "/calculate",
            json=request_data,
            headers={
                "Content-type": "application/json",
                "Accept": "text/plain",
            },
        )
    assert response.status_code == 200
    assert response.json() == answer
