if __name__ == "__main__":
    from model import load_models
    from schemas import IrisFeatures, PredictionResult
else:
    from app.model import load_models
    from app.schemas import IrisFeatures, PredictionResult


def prediction(features: IrisFeatures, model):
    features = [
        features.sepal[0],
        features.sepal[1],
        features.petal[0],
        features.petal[1],
    ]
    result = model.predict_proba([features])[0]
    result = PredictionResult(
        setosa=result[0], versicolor=result[1], virginica=result[2]
    )
    return result


if __name__ == "__main__":
    features = IrisFeatures(sepal=(0.1, 0.2), petal=(0.1, 0.2))
    models = load_models()
    model_name = "SVM"
    model = models[model_name]

    result = prediction(features, model)
    print(result)
