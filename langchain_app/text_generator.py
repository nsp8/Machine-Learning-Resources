from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
from loader import FileLoader, PythonFileLoader

load_dotenv()


class TextGenerator:

    def __init__(
        self,
        file_loader: FileLoader,
        temperature: float = 0.75,
    ):
        self.llm = OpenAI(temperature=temperature)
        self.file_loader = file_loader
        self.template = """Given the contents of the files below, 
        generate {goal}:
        {file_contents}
        """
        self.prompt = PromptTemplate(
            input_variables=["goal", "file_contents"],
            template=self.template
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def execute_goal_single_file(self, goal: str):
        documents = self.file_loader.documents
        if documents:
            for document in documents:
                yield self.chain.invoke({"goal": goal, "file_contents": document})
