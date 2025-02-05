# RAGifyResearch  

**RAGifyResearch** is a simple, fully local Retrieval-Augmented Generation (RAG) system designed to help interns in our research lab learn more about ongoing research. It enables users to extract and query research papers, abstracts, and documents using `CLI`.  

## Features  

✅ Supports **PDFs** as a knowledge source  
✅ Extracts and chunks text  
✅ Stores data in **ChromaDB** for retrieval  
✅ Enables **local chatbot interaction**  
✅ Uses **LM Studio** with **Meta Llama 3.1 7B-Instruct**  
✅ Uses **Jina Embeddings v2** for vector storage  

## Planned Features  

🚀 **Support for additional models**  
🚀 **Discord Bot Integration** to allow query research documents seamlessly  

## Installation  

1. **Clone the Repository**  
   ```sh
   git clone https://github.com/yourusername/RAGifyResearch.git  
   cd RAGifyResearch  
   ```

2. **Install Dependencies**  
   ```sh
   pip install -r requirements.txt  
   ```

3. **Run the System: First Time**  
   ```sh
   python main.py --create-db path/to/pdf/dir db_name
   ```
4. **Run the System: Any Time**  
   ```sh
   python main.py --load-db db_name
   ```


## Usage  

1. Add research papers (PDFs) to the designated folder.  
2. The system will extract, chunk, and store the text in **ChromaDB**.  
3. Use the CLI to query the documents and get relevant information.  
4. To exit the chat, use `exit`.

## Contributing  

Contributions are welcome! Feel free to submit a pull request or open an issue.  

## License  

📜 MIT License  

---

Would you like me to add anything specific, such as examples or screenshots? 😊