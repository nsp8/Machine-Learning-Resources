from text_generator import TextGenerator


if __name__ == "__main__":
    text_generator = TextGenerator()
    generated_text = text_generator.generate_text(input(">_: "))
    text_generator.print(generated_text)
    print(generated_text)
