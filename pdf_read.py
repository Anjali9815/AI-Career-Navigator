# from langchain_community.document_loaders import PyPDFLoader
# loader = PyPDFLoader("/Users/anjalijha/Python/AI-Career-Navigator/data/docs/Profile (1).pdf")
# docs = loader.load()  # returns one Document per page
# print(docs[0].page_content)
# Run this one-off to see what models you have access to
import google.generativeai as genai
import os
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
quit()
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
import json


loader = DirectoryLoader(
    "data/docs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
pdf_docs = loader.load()

print(sum(len(d.page_content) for d in pdf_docs))
print(len(pdf_docs))
avg_page = sum(len(d.page_content) for d in pdf_docs) // len(pdf_docs)
print(f"Average page length: {avg_page} chars")
# If avg_page is 800 → chunk_size 400-500 is perfect
# If avg_page is 2000 → chunk_size 600-800 works better

quit()


docs_json = []

for doc in pdf_docs:
    docs_json.append({
        "content": doc.page_content,
        "metadata": doc.metadata
    })

# Save to JSON file
with open("output.json", "w", encoding="utf-8") as f:
    json.dump(docs_json, f, indent=4, ensure_ascii=False)

print("Saved to output.json")

# # See ALL 57 docs and where they came from
# for i, doc in enumerate(docs):
#     print(f"Doc {i:3d} | page {doc.metadata['page']} | "
#           f"{doc.metadata['source']} | "
#           f"{len(doc.page_content)} chars")
    
# print(docs)
doc = pdf_docs[0]
print(repr(doc.page_content))
quit()
# print(docs[0].page_content)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # max tokens per chunk
    chunk_overlap=50,      # shared tokens between chunks
    separators=["\n\n", "\n", ".", " ", ""]  # tries these in order
)

chunks = splitter.split_documents(docs)
print("&&&&&&&&")
print(chunks[0].page_content)

print("&&&&&&&&")
print(chunks[1].page_content)
print(len(docs))
print(len(chunks))