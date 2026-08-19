from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os, sys
from backend.core.config import DOCS_DIR

def load_pdf(file_path):
    loader = DirectoryLoader(file_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True, loader_kwargs={"mode": "single"})
    documents = loader.load()
    return documents

def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunk_data = splitter.split_documents(documents)
    return chunk_data

def overlap_size(a_words, b_words, max_check=20):
    for n in range(max_check, 0, -1):
        if a_words[-n:] == b_words[:n]:
            return n     
    return 0

def check_overlap(chunks):
    for i in range(len(chunks) - 1):
        a_words = chunks[i].page_content.split()
        b_words = chunks[i + 1].page_content.split()
        check_data = overlap_size(a_words, b_words, 20)
        # if check_data == 0:
        #     print("No overlap for this chunk")
        #     print(chunks[i].metadata["source"])


        


if __name__ == "__main__":
    docs = load_pdf(DOCS_DIR)
    
    split_data = split_docs(docs)
    overlap_data_check = check_overlap(split_data)
    # 57 pages, 21 profiles, 36 pages are like some profile have two pages or three or 1
    # 111 is the splitted chunks across each page
    # print(len(docs), len(split_data))
    