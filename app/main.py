from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.model import load_models
from app.predict import prediction
from app.schemas import IrisFeatures, Prediction, PredictionResult

app = FastAPI()
models = load_models()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def predict(features: IrisFeatures):
    """
    Receives Iris data and returns a dummy prediction.
    """
    results = {
        model_name: prediction(features, model=models[model_name])
        for model_name in models.keys()
    }
    return Prediction(results)
