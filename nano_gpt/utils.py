from pathlib import Path
from nano_gpt.tokenizer import CharacterLevelTokenizer
from nano_gpt.constants import TEST_FILE_PATH

TEST_FILE = ""
DEFAULT_FILE_NAME = "input.txt"


def with_file(f):
    def wrapper(*args, **kwargs):
        global TEST_FILE
        directory = Path(__file__).parent.absolute()
        TEST_FILE = TEST_FILE or Path(f"{directory}/{DEFAULT_FILE_NAME}")
        return f(*args, **kwargs)
    return wrapper


@with_file
def download_test_file() -> bool:
    if not Path(TEST_FILE).exists():
        from subprocess import check_output
        print(f"Downloading file into {TEST_FILE} ...")
        check_output(f"wget {TEST_FILE_PATH}", shell=True)
        return True
    else:
        print("Test file already exists!")
        return False


@with_file
def convert_test_file_to_tensors() -> CharacterLevelTokenizer | None:
    if Path(TEST_FILE).exists():
        tokenizer = CharacterLevelTokenizer()
        with open(TEST_FILE) as f:
            text = f.read()
            tokenizer.set_tokens(text)
            return tokenizer
    return None


def display_tokenized_chunks(data: tuple, chunk_size: int, batch_size: int):
    print(f"{'-' * 70}")
    x, y = data
    for batch in range(batch_size):
        for chunk in range(chunk_size):
            context = x[batch, :chunk+1]
            target = y[batch, chunk]
            print(f"When input is {context.tolist()}, the target is: {target}")
