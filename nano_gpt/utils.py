from pathlib import Path
from torch import tensor, long
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
    if Path(TEST_FILE).exists():
        tokenizer = CharacterLevelTokenizer()
        with open(TEST_FILE) as f:
            text = f.read()
            return tensor(tokenizer.encode(text), dtype=long)
    return None
