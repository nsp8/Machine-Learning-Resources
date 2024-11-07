from dataclasses import dataclass
from enum import StrEnum
from math import e, log
import torch
from torch import Tensor
from torch.nn import Embedding, Module, functional
from nano_gpt.tokenizer import CharacterLevelTokenizer
from nano_gpt.utils import convert_test_file_to_tensors, display_tokenized_chunks


class DataSplitType(StrEnum):
    TEST = "TEST"
    TRAIN = "TRAIN"


@dataclass
class DataSet:
    train: Tensor
    test: Tensor


class DataLoader:

    def __init__(self, data: Tensor, split_ratio: float = 0.75):
        if 0.5 <= split_ratio < 1:
            n = int(split_ratio * len(data))
            self.data = DataSet(train=data[:n], test=data[n:])
        else:
            raise ValueError("split ratio must be within 0.5 and 1.")

    def get_chunks(
            self,
            split: DataSplitType,
            *,
            chunk_size: int = 8,
            batch_size: int = 4,
            visualize_tokenized_chunks: bool = True
    ) -> tuple[Tensor, Tensor]:
        def _get_batch(data):
            # random offsets: generate batch_size number of random offsets
            # between 0 and a chunk
            random_position = torch.randint(len(data) - chunk_size, (batch_size,))
            x = torch.stack([data[i:i+chunk_size] for i in random_position])
            y = torch.stack([data[i+1:i+chunk_size+1] for i in random_position])
            return x, y

        match split:
            case DataSplitType.TRAIN:
                batch = _get_batch(self.data.train)
            case DataSplitType.TEST:
                batch = _get_batch(self.data.test[:chunk_size])
            case _:
                raise ValueError("invalid source for data")
        if visualize_tokenized_chunks:
            display_tokenized_chunks(batch, chunk_size=chunk_size, batch_size=batch_size)
        return batch


class BigramLanguageModel(Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # (batch * time * channel),
        # where time is the chunk size and channel is the vocab size
        logits = self.token_embedding_table(idx)
        # Loss tells the quality of the logits
        # PyTorch expects the multidimensional inputs
        # to CrossEntropyLoss as (batch * channel * time)
        _batch, _time, _channel = logits.shape
        logits = logits.view(_batch * _time, _channel)
        targets = targets.view(_batch * _time)
        loss = functional.cross_entropy(logits, targets)
        return logits, loss


def test_bigram_model():
    data: CharacterLevelTokenizer | None = convert_test_file_to_tensors()
    if data:
        tensors = data.tokens
        if tensors is not None and isinstance(tensors, Tensor):
            data_loader = DataLoader(tensors)
            train_batch = data_loader.get_chunks(
                DataSplitType.TRAIN, visualize_tokenized_chunks=False
            )
            vocab_size = data.n_vocab
            if vocab_size > 0:
                model = BigramLanguageModel(vocab_size=data.n_vocab)
                logits, loss = model(*train_batch)
                # for a vocab_size, the Log Likelihood Estimate:
                # -ln(1/vocab_size)
                expected_log_loss = -log(1 / vocab_size, e)
                print(f"{logits.shape=} {loss=} {expected_log_loss=:.4f}")
            else:
                raise ValueError("Vocab size has to be a positive integer.")
    else:
        print("No data found!")
