# Solves compatibility issues
# Check the issus here:
# [https://stackoverflow.com/questions/65145848/configure-python-sqlite3-module-to-use-different-newer-version-sqlite-system-l]
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import requests
import pymupdf
import glob
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, CharacterTextSplitter
from typing import Optional

class IngestData:
    def __init__(self, lms_client, chroma_client):
        self._lms_client = lms_client
        self._chroma_client = chroma_client

    def _fetch_models(self):
        try:    
            response = requests.get(str(self._lms_client.base_url) + 'models', timeout=5).json()
            models = response.get('data')
        except Exception as e:
            raise ValueError(f"Could not connect to {self._lms_client.base_url} due to {e}")
        
        # Handling error within LM-studio when accessing undefined 
        # endpoints which returns with 'Returning 200 anyway'
        if 'error' in response.keys():
            raise ValueError(f"Could not to {self._lms_client.base_url} due to {response.get('error')}")

        # Check if both the models are specified
        # E.g., LLM model and Embedding model
        if len(models) == 2:
            llm_model = "/".join(model for model in models[0]['id'].split("/")[:2])
            embedding_model = "/".join(model for model in models[1]['id'].split("/")[:2])
        else:
            raise ValueError(f"Not enough models. Require 2 models (e.g., LLM model and Embedding Model). \nOnly found {models[0]['id']}")
        
        return llm_model, embedding_model
    
    def _process_pdf(self, pdf_path):
        doc = pymupdf.open(pdf_path)
        text = " ".join(page.get_text("text") for page in doc)

        # for k, characters in enumerate(text.split(" ")):
        #     if characters.lower() == 'references':
        #         ref_index = k
        #         return " ".join(text for text in text.split(" ")[:ref_index])

        return text.replace("\n", " ")
        
    def _chunk_text(self, text, chunk_size, chunk_overlap, splitter_type='recursive_text'):
        if splitter_type == 'recursive_text':
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        elif splitter_type == 'markdown_text':
            splitter = MarkdownTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3")
                ],
                strip_header=True
            )
        elif splitter_type == 'character_text':
            splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,  # Use method parameters
                chunk_overlap=chunk_overlap,  # Use method parameters
                length_function=len,
                is_separator_regex=False,
            )
        else:
            raise ValueError(f"Unsupported splitter_type: {splitter_type}")

        return splitter.create_documents([text])


    def _get_embedding(self, text, embedding_model):
        return self._lms_client.embeddings.create(input = [text], model=embedding_model).data[0].embedding

    def _create_db(self, pdf_dir_path: str, 
                db_name: str,
                splitter_type: Optional[str] = 'recursive_text', 
                chunk_size: Optional[int] = 1000, 
                chunk_overlap: Optional[int] = 250):
        
        # Perform sanity checks
        if not os.path.exists(pdf_dir_path):
            raise ValueError(f"PDF filepath at {pdf_dir_path} do not exist. Please check the filepath")

        if not sorted(glob.glob(f"{pdf_dir_path}/*.pdf")):
            raise ValueError(f"No PDF files found at {pdf_dir_path}")
        
        if db_name in [collection.name for collection in self._chroma_client.list_collections()]:
            raise ValueError(f"Database {db_name} already exisits. Please use another database name.")
        
        # Fetch models
        _, embedding_model = self._fetch_models()

        # Initialize ChromaDB
        collection = self._chroma_client.create_collection(name=db_name)

        for pdf in sorted(glob.glob(f"{pdf_dir_path}/*.pdf")):
            text = self._process_pdf(pdf)
            print(f"Processing: {pdf}")

            documents = self._chunk_text(text, chunk_size, chunk_overlap, splitter_type)
            for i, doc in enumerate(documents):
                embedding = self._get_embedding(doc.page_content, embedding_model)
                collection.add(
                    # ids: pdf_filename_chunk_k, where k = [0,len(doc)]
                    ids=[f"{pdf.split('/')[-1].split('.')[0].replace(' ','')}_chunk_{i}"],
                    embeddings=[embedding],
                    documents=[doc.page_content]
                )

        return collection

    def _load_db(self, db_name: Optional[str] = 'rag_db'):
        # Perform sanity check
        if db_name not in [collection.name for collection in self._chroma_client.list_collections()]:
            raise ValueError(f"Database {db_name} do not exist. Please verify database name.")
         
        try:
            collection = self._chroma_client.get_collection(name=db_name)
        except Exception as e:
            raise ValueError(f"Could not find database named {db_name}. {e}")
        
        return collection


