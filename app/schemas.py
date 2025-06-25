from typing import Dict, Tuple

from pydantic import BaseModel, Field, validator


class IrisFeatures(BaseModel):
    sepal: Tuple[float, float] = Field(..., description="(length, width) of sepal")
    petal: Tuple[float, float] = Field(..., description="(length, width) of petal")


class PredictionResult(BaseModel):
    setosa: float = Field(..., ge=0, le=1, description="Probability of Iris-setosa")
    versicolor: float = Field(
        ..., ge=0, le=1, description="Probability of Iris-versicolor"
    )
    virginica: float = Field(
        ..., ge=0, le=1, description="Probability of Iris-virginica"
    )

    @validator("setosa", "versicolor", "virginica", pre=True)
    def round_probabilities(cls, v):
        return round(float(v), 3)


from pydantic import RootModel


class Prediction(RootModel[Dict[str, PredictionResult]]):
    pass
