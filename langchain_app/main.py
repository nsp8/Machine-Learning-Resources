from pathlib import Path
from text_generator import TextGenerator
from loader import PythonFileLoader


if __name__ == "__main__":
    file_path = Path(input("Enter directory path: "))
    file_loader = PythonFileLoader(file_path if file_path.is_dir() else file_path.parent)
    text_generator = TextGenerator(file_loader=file_loader)
    generated = text_generator.execute_goal_single_file(input("prompt >_: "))
    try:
        while True:
            print(next(generated))
            input("_")
    except (KeyboardInterrupt, StopIteration):
        print("<EOE>")
