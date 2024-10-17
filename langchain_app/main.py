from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import dotenv_values

env = dotenv_values(".env")


class TextGenerator:
    checkpoint = env.get("CODEGEN_MODEL")

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(TextGenerator.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(TextGenerator.checkpoint)

    def generate_text(self, text: str):
        return self.model.generate(**self.tokenizer(text, return_tensors="pt"))

    def print(self, text: str):
        print(self.tokenizer.decode(text[0]))


if __name__ == "__main__":
    text_generator = TextGenerator()
    generated_text = text_generator.generate_text(input(">_: "))
    text_generator.print(generated_text)
