from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import base64
import io

from inference import translation

app = FastAPI()

@app.get("/")
def aa():

    return {'ok': 'ok-'}


@app.get("/predict")
def predict(input: str):

    img = translation(input)
    img.save('example.png', "PNG")
    return {'ok': 'ok'}
