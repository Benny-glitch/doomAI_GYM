from typing import Union
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import os

app = FastAPI()


@app.post("/observations/")
def observationUpdate(obs):
    return


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}