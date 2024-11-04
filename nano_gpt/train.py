from dataclasses import dataclass
import torch


@dataclass
class DataSet:
    train: torch.Tensor
    test: torch.Tensor


class DataLoader:

    def __init__(self, data, split_ratio: float):
        if 0.5 <= split_ratio < 1:
            n = int(split_ratio * len(data))
            self.data = DataSet(train=data[:n], test=data[n:])
        raise ValueError("split ratio must be within 0.5 and 1.")

    def get_chunks(self, source: str, chunk_size: int = 8, batch_size: int = 4):
        def _get_batch(data):
            # random offsets: generate batch_size number of random offsets
            # between 0 and a chunk
            random_position = torch.randint(len(data) - chunk_size, (batch_size,))
            x = torch.stack([data[i:i+chunk_size] for i in random_position])
            y = torch.stack([data[i+1:i+chunk_size+1] for i in random_position])
            return x, y

        match source:
            case "train":
                return _get_batch(self.data.train)
            case "test":
                return _get_batch(self.data.test[:chunk_size])
            case _:
                raise ValueError("invalid source for data")
