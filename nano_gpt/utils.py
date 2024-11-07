from pathlib import Path
from nano_gpt.tokenizer import CharacterLevelTokenizer
from nano_gpt.constants import TEST_FILE_PATH

TEST_FILE = ""


def download_test_file():
    global TEST_FILE
    file_name = Path(TEST_FILE_PATH).name
    TEST_FILE = f"./{file_name}"
    if not Path(TEST_FILE).exists():
        from subprocess import check_output
        check_output(f"wget {TEST_FILE_PATH}", shell=True)
    else:
        print("Test file already exists!")


def convert_test_file_to_tensors():
    global TEST_FILE
    directory = Path(__file__).parent.absolute()
    TEST_FILE = TEST_FILE or Path(f"{directory}/input.txt")
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
