from pathlib import Path
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split


def csv_to_dataframe(directory_path: str, relative_file_path: str) -> DataFrame | None:
    relative_file_path = relative_file_path[1:].strip() \
        if relative_file_path.startswith("/") \
        else relative_file_path.strip()
    file_path = Path(f"{directory_path}/{relative_file_path}")
    if file_path.exists() and relative_file_path.endswith(".csv"):
        return read_csv(file_path)
    return None


class Dataset:
    def __init__(
        self,
        train_data: DataFrame,
        test_dataset: DataFrame,
        test_size: float=0.33
    ):
        """`test_size` should be within (0, 1)"""
        test_size = abs(test_size)
        if test_size > 1:
            print("test size should be within the range: (0, 1)")
            print("setting the default to 0.33 (30%)")
            test_size = 0.33
        self.X = train_data
        self.y = test_dataset
        self.x_train, self.y_train, self.x_val, self.y_val = train_test_split(
            train_data, test_dataset, test_size=test_size
        )
