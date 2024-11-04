from collections import defaultdict
import string


class CharacterLevelTokenizer:

    def __init__(self):
        self.char_to_int = defaultdict()
        self.int_to_char = defaultdict()
        self.all_characters = sorted(
            f"{string.ascii_uppercase}"
            f"{string.ascii_lowercase}"
            f"{string.digits}"
            f"{string.punctuation}"
            f"{string.whitespace}"
        )
        self.n_vocab = len(self.all_characters)
        for i, char in enumerate(self.all_characters):
            self.char_to_int[char] = i
            self.int_to_char[i] = char

    def encode(self, value: str) -> list:
        return [self.char_to_int[c] for c in value]

    def decode(self, value: list) -> str:
        return "".join([self.int_to_char[c] for c in value])

