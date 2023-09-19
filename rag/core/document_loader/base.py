from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from typing import List, Optional, Union
from langchain.schema.document import Document


class AutoFileLoader:
    """AutoFileLoader to load files
    using appropriate loaders based on file type."""

    def __init__(
            self,
            file_path: str,
            password: Optional[Union[str, bytes]] = None
    ):
        self.file_path = file_path
        self.password = password

    def load(self) -> List[Document]:
        file_extension = self.get_file_extension()
        if file_extension == ".html":
            loader = UnstructuredHTMLLoader(
                self.file_path,
                mode="elements",
                strategy="fast"
            )
        elif file_extension == ".pdf":
            loader = PyPDFLoader(
                self.file_path,
                password=self.password
            )
        elif (
            file_extension.startswith("http://") or
            file_extension.startswith("https://")
        ):
            loader = UnstructuredURLLoader(
                [self.file_path],
                mode="elements",
                strategy="fast"
            )
        else:
            raise ValueError("Unsupported file type")

        return loader.load()

    def get_file_extension(self) -> str:
        return self.file_path.lower().split(".")[-1]
