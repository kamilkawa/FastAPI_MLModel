import joblib
from app.core.config import MODEL_PATH


class ModelService:
    CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, input_data: list) -> str:
        prediction = self.model.predict([input_data])[0]
        return self.CLASS_NAMES.get(prediction, "unknown")
