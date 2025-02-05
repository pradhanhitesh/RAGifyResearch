import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ast
import requests

# Download NLTK libratires
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')

class SafetyAgent:
    def __init__(self, client, safety_rules=None):
        if safety_rules is None:
            safety_rules= """
            ### **Reject Malicious or Manipulative Queries**  
            - Do not assist in bypassing security controls or engaging in fraud.  
            - Decline misleading, vague, or harmful queries.  

            ### **No Harassment or Hate Speech**  
            - Reject content promoting violence, discrimination, or harm.  
            - Do not generate abusive, bullying, or targeted harassment content.  

            ### **No Illegal Activities**  
            - Do not assist in illegal activities such as hacking, drug production, or human trafficking.  

            ### **No Misinformation or Deception**  
            - Avoid generating false, misleading, or deceptive content.  
            - Do not fabricate facts, sources, or impersonate individuals.  

            ### **Strict Compliance & Enforcement**  
            Follow these rules rigorously to ensure secure and responsible interactions.
            """
        self.safety_rules = safety_rules.strip().splitlines()
        self.client = client

    def validate_query(self, query):
        query_lower = query.lower()
        try:
            response = self.client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", 
                    "content": "You are a Safety AI Agent." 
                                f"Assess the user's content based on the following safety rules:\n{self.safety_rules}\n"
                                "Respond only with 'yes' if the content violates safety rules, otherwise respond only with 'no'.\n" 
                                "Do not provide any explanation, reasoning, or additional text."},
                    {"role": "user", "content": query_lower},
                ],
                temperature=0
            )
            response = response.choices[0].message.content.lower()
            return "yes" if response == 'yes' else "no"
        except Exception as e:
            return f"error: {str(e)}"
        
class QueryAgent:
    def __init__(self, client):
        self.client = client

    def clean_query(self, query):
        query_lower = query.lower()
        tokens = word_tokenize(query_lower)        
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

        return " ".join(filtered_tokens)

    def enhanced_query(self, query):
        filtered_query = self.clean_query(query)
        
        try:
            response = self.client.chat.completions.create(
                model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", 
                     "content": "You are a User-Query AI Agent. Enhance and structure the user input into a single-short string. \n"
                                "Return output as Python dict, e.g., {'enhanced_query': <query goes here>}. Do not add anything else."                     
                     },
                    {"role": "user", "content": filtered_query},
                ],
                temperature=0.2
            )
            response = ast.literal_eval(response.choices[0].message.content.lower())
            return response['enhanced_query']
        except Exception as e:
            return f"Error: {str(e)}"
        
class RAGAgent:
    def __init__(self, query, database, lms_client):
        self._query = query
        self._database = database
        self._lms_client = lms_client

    def _fetch_models(self):
        try:    
            response = requests.get(str(self._lms_client.base_url) + 'models', timeout=5).json()
            models = response.get('data')
        except Exception as e:
            raise ValueError(f"Could not connect to {self._lms_client.base_url} due to {e}")

        if 'error' in response.keys():
            raise ValueError(f"Could not to {self._lms_client.base_url} due to {response.get('error')}")

        if len(models) == 2:
            llm_model = "/".join(model for model in models[0]['id'].split("/")[:2])
            embedding_model = "/".join(model for model in models[1]['id'].split("/")[:2])
        else:
            raise ValueError(f"Not enough models. Require 2 models (e.g., LLM model and Embedding Model). \nOnly found {models[0]['id']}")
        
        return llm_model, embedding_model
    
    def _get_embedding(self, text, embedding_model):
        return self._lms_client.embeddings.create(input = [text], model=embedding_model).data[0].embedding

    def _rag_rules(self):
        rag_rules = """
        You are an AI assistant using a Retrieval-Augmented Generation (RAG) system. Follow these strict rules to ensure security, accuracy, and relevance:  

        ### **1. Retrieve Only Relevant Information**  
        - Query the knowledge base for the most relevant documents based on the user's input.  
        - Ignore documents that do not directly align with the query intent.

        ### **2. Prioritize Accuracy & Verifiability**  
        - Use only retrieved documents for responses—do not generate or infer missing details.  
        - If no relevant data is found, state explicitly: *"No relevant information available."*  

        ### **3. Provide Sources Transparently**  
        - Cite retrieved documents when supporting responses.  
        - If multiple sources are used, summarize their key points concisely.  

        ### **4. Maintain Query Context & Integrity**  
        - Do not deviate from the given query intent.  
        - Do not generate speculative, hypothetical, or fictional content.  

        ### **5. Reject Irrelevant & Manipulative Requests**  
        - No roleplay, storytelling, or narrative manipulation.  
        - Decline queries unrelated to the system’s purpose (e.g., creative writing, casual conversation).  

        ### **6. Summarize When Necessary**  
        - If multiple documents provide similar insights, consolidate the key points.  
        - Avoid excessive detail unless explicitly requested.  

        ### **7. Prevent Redundant & Circular Responses**  
        - Do not restate retrieved content verbatim unless necessary.  
        - Ensure each response is concise and informative.  

        ### **8. Handle Uncertainty Properly**  
        - If retrieved data is ambiguous or incomplete, request clarification rather than making assumptions.  
        - If necessary, inform the user that relevant data is unavailable.  

        ### **9. Enforce Response Length Control**  
        - Keep responses concise unless a detailed explanation is explicitly requested.  
        - Avoid unnecessary elaboration or off-topic discussion.  

        ### **10. Ensure Readability & Structured Formatting**  
        - Use bullet points, numbered lists, or paragraphs for clarity.  
        - Maintain a professional and structured tone.  

        ### **11. Reject Malicious or Manipulative Queries**  
        - Do not respond to queries that attempt to bypass security controls.  
        - If a query is misleading, vague, or potentially harmful, decline to respond.  

        ### **Strict Compliance & Enforcement**  
        Follow these rules rigorously to ensure accurate, secure, and context-aware responses.
        """

        return rag_rules

    def _retrieval(self, query, database, top_k=10, similarity_threshold=0.5):
        _, embedding_model = self._fetch_models()
        query_embedding = self._get_embedding(query, embedding_model)
        
        # Fetch top_k results
        results = database.query(query_embeddings=[query_embedding], n_results=top_k)
        
        # Manually filter based on similarity score
        context = []
        for doc, score, metadata in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
            if score <= similarity_threshold:  # Adjust threshold as needed
                # context.append({"document": doc, "score": score, "metadata": metadata})
                context.append(doc)
        
        if len(context) == 0:
            context = ['Did not find any relevant documents. Please decline to answer.']
        
        return context

    def _augmentation(self, query, context):
        context = " ".join(contexts for contexts in context)
        
        augmentation_rules = """
        ### **Augmentation Rules**
        - **Expand on key points**: Use retrieved context to provide a well-structured, in-depth response.
        - **Ensure coherence**: Maintain logical flow and readability in your response.
        - **Maintain factual accuracy**: Do not introduce unverified information.
        - **Provide comprehensive details**: Expand on concepts where necessary to enhance understanding.
        - **Use structured formatting**: Utilize bullet points or paragraphs for clarity.
        - **Avoid unnecessary repetition**: Ensure that the response remains informative and concise.
        - **Respect query intent**: Answer directly based on retrieved data without making assumptions.
        """

        augmented_text = f"""" 
        Here is the retrieved context: {context}.

        Based on the retrieved content, please answer the following query: {query}

        {augmentation_rules}
        """
        return augmented_text


    def _generation(self):
        # 0. Fetch LLM model
        llm_model, _ = self._fetch_models()

        # 1. Retrive documents
        retrieved_documents = self._retrieval(self._query, self._database)

        # 2. Augmented context and query
        augmented_text = self._augmentation(self._query, retrieved_documents)

        # 3. Generate text
        generation = self._lms_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": f"You are an AI assistant using a Retrieval-Augmented Generation (RAG) system. Follow these strict rules to ensure security, accuracy, and relevance:  {self._rag_rules()}"},
                {"role": "user", "content": augmented_text}
            ],
            temperature=0.7,
        )
         
        return generation.choices[0].message.content