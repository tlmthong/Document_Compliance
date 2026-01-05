from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
NOTE: import error, dismissed error raised in util file
"""


class Segmentation:
    def __init__(self):
        pass

    def process_markdown(
        self,
        markdown_document,
        chunk_size=100,
        overlap=30,
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
    ):
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        splits = splitter.split_text(markdown_document)

        for doc in splits:
            print(doc.page_content, doc.metadata)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

        final_chunks_docs = text_splitter.split_documents(splits)
        # Convert Document objects to list of strings (page_content)
        final_chunks = [doc.page_content for doc in final_chunks_docs]
        return final_chunks

    """
    This return a list of segmented document
    """

    def segment_whole_to_chunks(
        self,
        doc,
        chunk_size,
        overlap,
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        return text_splitter.split_text(doc)
