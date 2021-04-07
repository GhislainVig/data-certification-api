
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness, danceability, duration_ms, energy, explicit, id, instrumentalness,\
        key, liveness, loudness, mode, name, release_date, speechiness, tempo, valence, artist):
    x = {
        "acousticness" : float(acousticness),
        "danceability": float(danceability),
        "duration_ms": int(duration_ms),
        "energy": float(energy),
        "explicit": int(explicit),
        "id": id,
        "instrumentalness": float(instrumentalness),
        "key": int(key),
        "liveness": float(liveness),
        "loudness": float(loudness),
        "mode": int(mode),
        "name": artist,
        "release_date": release_date,
        "speechiness": float(speechiness),
        "tempo": float(tempo),
        "valence": float(valence),
        "artist": artist
    }
    df = pd.DataFrame([x.values()], columns = x.keys())
    pipeline = joblib.load('model.joblib')
    y_pred = pipeline.predict(df)[0]
    return {
        "artist": artist,
        "name": name,
        "popularity": y_pred
    }
