__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import argparse
import chromadb
from openai import OpenAI
from RAGify.ingest import IngestData
from RAGify.agents import SafetyAgent, QueryAgent, RAGAgent

# Setup clients for LMS and ChromaDB
lms_client = OpenAI(base_url="http://172.26.208.1:5555/v1", api_key="lm-studio")
chroma_client = chromadb.PersistentClient(path='assets/chroma_db')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="RAG System CLI")
parser.add_argument("--load-db", metavar="DB_NAME", type=str, help="Load the specified database")
parser.add_argument("--create-db", metavar=("PDF_DIR_PATH", "DB_NAME"), type=str, nargs=2, help="Create a new database with the specified PDF directory and database name")

args = parser.parse_args()

# Initialize IngestData
ingest = IngestData(lms_client, chroma_client)

if args.load_db:
    database = ingest._load_db(args.load_db)
elif args.create_db:
    pdf_dir_path, db_name = args.create_db
    database = ingest._create_db(pdf_dir_path, db_name)
else:
    print("Please specify either --load-db <db_name> or --create-db <pdf_dir_path> <db_name>")
    sys.exit(1)

# Interactive Chat Loop
while True:
    query = input("User: ")
    if query.lower() == "exit":
        print("Exiting Chat. Goodbye!")
        break

    if SafetyAgent(lms_client).validate_query(query) == "no":
        query = QueryAgent(lms_client).enhanced_query(query)
        rag = RAGAgent(query, database, lms_client)
        response = rag._generation()
        print(response)
    else:
        print("Terms and Conditions have been violated regarding safety. Terminating chat now.")
        break
