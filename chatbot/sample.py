import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
# Ensure NLTK tokenizer is available
nltk.download('punkt_tab')
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from flask import Flask, request, render_template, jsonify
from langchain import hub
import fitz  # PyMuPDF
import torch


# Initialize the Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Accessing the various API KEYS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("rag")


# directory = './pdf'

# def read_all_pdf(directory):
#     all_texts=[]
    
#     for filename in os.listdir(directory):
#         if filename.endswith(".pdf"):
#             file_path = os.path.join(directory, filename)
#             print(f"Processing {file_path}")
#             doc = fitz.open(file_path)
#             text = " ".join([page.get_text() for page in doc if page.get_text()])
#             if text:  # Ensure text is not empty
#                 all_texts.append(text)
#             doc.close()
#     return all_texts
            
    
    
# content = read_all_pdf(directory)    

# knowledge_base = {}
# if content:  # Make sure content is not empty
#     knowledge_base[f"doc_{1}"] = " ".join(content) if isinstance(content, list) else content


# def chunk_text(text, chunk_size=300):
#     sentences = sent_tokenize(text)
#     chunks = []
#     chunk = []
#     total_length = 0

#     for sentence in sentences:
#         total_length += len(sentence.split())
#         if total_length > chunk_size:
#             chunks.append(" ".join(chunk))
#             chunk = []
#             total_length = len(sentence.split())
#         chunk.append(sentence)

#     if chunk:
#         chunks.append(" ".join(chunk))
    
#     return chunks

# # Chunk all web page content
# chunked_knowledge_base = {}
# for i, content in knowledge_base.items():
#     chunked_knowledge_base[i] = chunk_text(content)
    

# Load Hugging Face model
model_name = "./llm"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             load_in_8bit=True,
                                             device_map="auto")
model.generation_config.max_new_tokens = 300
model.generation_config.max_length = 1000 
# Check if CUDA is available and set the appropriate device
device = 0 if torch.cuda.is_available() else -1  # -1 refers to CPU
# Create pipeline
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Use HuggingFacePipeline in LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)


embed_model = SentenceTransformer('all-mpnet-base-v2')
# #Embed and store chunks
# for doc_id, chunks in chunked_knowledge_base.items():
#     for i, chunk in enumerate(chunks):
#         embedding = embed_model.encode(chunk).tolist()
#         metadata = {"source": doc_id, "chunk_id": i, "text": chunk}
#         index.upsert([(f"{doc_id}_{i}", embedding, metadata)])

from langchain.vectorstores import Pinecone

# Define retriever
retriever_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
# retriever = PineconeVectorStore(index=index, embedding=retriever_embeddings.embed_query)
# retriever = PineconeVectorStore(embedding=retriever_embeddings.embed_query, index=index)

# retriever = Pinecone.from_documents(
#     documents=chunked_documents,  # Your documents or text chunks
#     embedding=retriever_embeddings.embed_documents,
#     index_name=index  # This should match your Pinecone index
# )

from langchain.prompts import PromptTemplate


#CUSTOM_PROMPT = hub.pull("rlm/rag-prompt")


prompt_template = """
Use the following context to answer the question concisely and directly. 
If you do not know the answer, say "I do not know."

Context:
{context}

Question:
{question}

Answer:
"""

CUSTOM_PROMPT = PromptTemplate(template= prompt_template, input_variables=["context", "question"])



# CUSTOM_PROMPT = PromptTemplate(
#     template=(
#         "You are to answer the question based on the provided context. "
#         "If you do not know the answer, say I do not know"
#         "Answer in three sentences. \n\n"
#         "Context:\n{context}\n\n"
#         "Question:\n{question}\n\n"
#         "Answer:"
#     ),
#     input_variables=["context", "question"]
# )

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

retriever = Pinecone.from_existing_index(
        index_name="rag",
        embedding=retriever_embeddings.embed_query
    )


#Retrieval-Augmented Generation Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt":CUSTOM_PROMPT},
    # output_parser=output_parser,
    return_source_documents=False,

)
# Function to get the RAG answer only
def get_rag_answer(question):
    response = qa_chain.run(question)
    answer = response.split("Answer:")[-1].strip()
    return answer.strip()

# # Example query
# query = "How to extend cloud volume?"
# result = qa_chain.invoke({"query": query})

# # Print answer and source
# print("Answer:", result['result'])
# print("\nSource Documents:")
# for doc in result['source_documents']:
#     print(doc.metadata["text"])


# rag_chain = (
#     {"context": retriever.as_retriever() | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    try:
        # Retrieve JSON data from the request
        data = request.get_json()
        #user_message = data.get("contents")[0]["parts"][0]["text"]
        # Extract the user message
        user_message = data.get("contents", [])[0].get("parts", [])[0].get("text", "")

        # Process the message using your RAG system
        # Replace the following line with the logic for `get_rag_answer`
        response_text = get_rag_answer(user_message)

        # Return the response in the required format
        response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": response_text}]
                }
            }]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": {"message": str(e)}}), 500
    


# Run the Flask app, making it available on port 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
