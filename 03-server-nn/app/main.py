import keras

import numpy

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

modelo = keras.models.load_model("../../modelos/titanic_nn_12i_1o_survived.keras")

# modelo.summary()

class Pasajero(BaseModel):

    sexo: str = "mujer"
    edad: int = 28
    clase: str = "3ra"
    familia: int = 0
    tarifa: float = 7.5
    cabina: bool = False
    puerto: str = "londres"
    sobrevive: float = 0.0

    def to_features(self):
        x1 = int(self.sexo == "mujer")
        x2 = int(self.edad)
        x3 = int(self.clase == "1ra")
        x4 = int(self.clase == "2da")
        x5 = int(self.familia)
        x6 = 0
        x7 = 0
        if self.familia == 1:
            x6 = 1
        if self.familia > 1:
            x7 = 1
        x8 = float(numpy.log1p(0 + self.tarifa))
        x9 = int(self.cabina)
        x10 = int(self.puerto == "francia")
        x11 = int(self.puerto == "irlanda")
        x12 = int(self.puerto == "desconocido")
        return numpy.array([
            x1, x2, x3, x4, 
            x5, x6, x7, x8, 
            x9, x10, x11, x12
        ])

    def predecir(self):
        y = modelo.predict(numpy.stack([self.to_features()]), verbose=0)
        self.sobrevive = float(y[0][0])
        return self

# pasajero = Pasajero.model_construct(
#     sexo="mujer", 
#     edad=36, 
#     clase="2da",
#     familia=2, 
#     tarifa=7.5, 
#     cabina=False, 
#     puerto="londres", 
# )
# pasajero.predecir()
# print(pasajero.model_dump_json())

# print(pasajero.to_features())
# y = modelo.predict(numpy.stack([pasajero.to_features()]), verbose=0)
# sobrevive = y[0][0]
# print(sobrevive)

@app.post("/titanic/predecir")
def titanicPredecir(pasajero: Pasajero):
    pasajero.predecir()
    return pasajero

# Invoke-RestMethod -Uri "http://localhost:5000/titanic/predecir" `
#   -Method POST `
#   -ContentType "application/json" `
#   -Body '{"sexo":"mujer","edad":36,"clase":"2da","familia":2,"tarifa":7.5,"cabina":false,"puerto":"londres"}'

# curl.exe -X POST "http://localhost:5000/titanic/predecir" `
#   -H "Content-Type: application/json" `
#   -d '{ "sexo": "mujer", "edad": 36, "clase": "2da", "familia": 2, "tarifa": 7.5, "cabina": false, "puerto": "londres" }'
