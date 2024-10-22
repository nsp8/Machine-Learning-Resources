from langchain_community.document_loaders import DirectoryLoader, PythonLoader, TextLoader


class FileLoader:
    def __init__(self, directory, file_extension: str, loader_type=TextLoader):
        self.glob = f"**/*.{file_extension.replace(".", "")}"
        self.loader = DirectoryLoader(directory, glob=self.glob, loader_cls=loader_type)

    @property
    def documents(self):
        try:
            return self.loader.load()
        except (ValueError, RuntimeError, IOError) as e:
            print(f"Could not load documents because: {e}")
            return None


class PythonFileLoader(FileLoader):
    def __init__(self, directory):
        super().__init__(directory, "py", PythonLoader)
