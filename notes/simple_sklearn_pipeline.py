from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from notes.data_loader import Dataset


class SimpleModelPipeline:
    def __init__(
        self,
        numerical_columns: list = None,
        categorical_columns: list = None,
    ):
        self.steps = list()
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.pipeline = None

    @property
    def numerical_transformer(self):
        if self.numerical_columns:
            return SimpleImputer(strategy="constant")
        return None

    @property
    def categorical_transformer(self):
        if self.categorical_columns:
            return Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
        return None

    def get_preprocessor(self):
        transformers = list()
        if self.numerical_columns:
            transformers.append(
                ("numerical", self.numerical_transformer, self.numerical_columns)
            )
        if self.categorical_columns:
            transformers.append(
                ("categorical", self.categorical_transformer, self.categorical_columns)
            )
        if transformers:
            return ColumnTransformer(transformers=transformers)
        return None

    def create(self, model):
        preprocessor = self.get_preprocessor()
        steps = list()
        if preprocessor:
            steps.append(("preprocessor", preprocessor))
        steps.append(("model", model))
        self.pipeline = Pipeline(steps=steps)

    def train(self, x_train, y_train):
        if self.pipeline:
            self.pipeline.fit(x_train, y_train)
        self.handle_missing_pipeline()

    def get_predictions(self, x_val):
        if self.pipeline:
            return self.pipeline.predict(x_val)
        self.handle_missing_pipeline()
        return None

    def get_metrics(self, data: Dataset):
        if self.pipeline:
            return {
                "mean_absolute_error": mean_absolute_error(
                    data.y_val, self.get_predictions(data.x_val)
                ),
                """
                Cross Validation: running the model on different subsets of data 
                to get multiple measures of model quality. 
                It divides the data into `cv` number of folds, 
                running an experiment on each fold.
                """
                "cross_validation_scores": -1 * cross_val_score(
                    self.pipeline, data.X, data.y, cv=5, scoring="neg_mean_absolute_error"
                ),
            }
        self.handle_missing_pipeline()
        return None

    @staticmethod
    def handle_missing_pipeline():
        print(
            "Pipeline has not been built yet. "
            "Invoke the create method to create the pipeline."
        )
