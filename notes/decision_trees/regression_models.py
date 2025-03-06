from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from notes.data_loader import Dataset


@dataclass
class TreeBasedRegressor:
    model: DecisionTreeRegressor | RandomForestRegressor
    dataset: Dataset

    def train(self):
        self.model.fit(self.dataset.x_train, self.dataset.y_train)

    def validate_predictions(self):
        return self.model.predict(self.dataset.x_val)

    def get_metrics(self, predictions):
        return {
            "mean_absolute_error": mean_absolute_error(self.dataset.y_val, predictions),
        }
