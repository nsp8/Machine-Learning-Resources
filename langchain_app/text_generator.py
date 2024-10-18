from transformers import BloomForCausalLM, AutoTokenizer
from dotenv import dotenv_values

env = dotenv_values(".env")


class TextGenerator:
    checkpoint = env.get("PRETRAINED_MODEL")

    def __init__(self):
        self.model = BloomForCausalLM.from_pretrained(TextGenerator.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(TextGenerator.checkpoint)

    def generate_text(self, text: str):
        return self.model.generate(**self.tokenizer(text, return_tensors="pt"), max_length=100)

    def print(self, text: str):
        print(self.tokenizer.decode(text[0]))
