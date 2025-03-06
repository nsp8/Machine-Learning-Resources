from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from notes.data_loader import Dataset


@dataclass
class TreeBasedRegressor:
    model: DecisionTreeRegressor | RandomForestRegressor | XGBRegressor
    dataset: Dataset

    def train(self):
        self.model.fit(self.dataset.x_train, self.dataset.y_train)

    def validate_predictions(self):
        return self.model.predict(self.dataset.x_val)

    def get_metrics(self, predictions):
        return {
            "mean_absolute_error": mean_absolute_error(self.dataset.y_val, predictions),
            "cross_validation_scores": -1 * cross_val_score(
                self.model,
                self.dataset.X,
                self.dataset.y,
                cv=5,
                scoring="neg_mean_absolute_error"
            ),
        }
