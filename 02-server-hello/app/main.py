from fastapi import FastAPI
from fastapi import responses

import numpy
import pandas

app = FastAPI()

@app.get("/")
def home():
    return "Hola desde el servidor 🤖"

@app.get("/edades")
def obtenerEdades():
    edades = numpy.random.normal(32, 23, 100)
    return responses.JSONResponse(
        content=pandas.DataFrame({
            "edades": edades
        }).to_dict(orient="list")
    )