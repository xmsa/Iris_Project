from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import IrisFeatures, Prediction, PredictionResult

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def predict(data: IrisFeatures):
    """
    Receives Iris data and returns a dummy prediction.
    """
    # result = prediction(model, data)
    svm = PredictionResult(
        setosa=0.1,
        versicolor=0.5,
        virginica=0.4,
    )
    DT = PredictionResult(
        setosa=0.1,
        versicolor=0.1,
        virginica=0.8,
    )
    RF = PredictionResult(
        setosa=0.7,
        versicolor=0.1,
        virginica=0.2,
    )
    reg = PredictionResult(
        setosa=0.3,
        versicolor=0.3,
        virginica=0.3,
    )
    results = {"svm": svm, "DT": DT, "RF": RF, "REG": reg}
    return Prediction(results)
