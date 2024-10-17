from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_text(prompt: str):
    llm = OpenAI(temperature=0.75)
    return llm(prompt)


if __name__ == "__main__":
    print(generate_text(input(">_: ")))
