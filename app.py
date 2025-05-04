from flask import Flask, render_template, request, jsonify
import os
import json
import time

# Langsmith imports
from langsmith import Client
from langchain_core.tracers.langchain import LangChainTracer

# Existing imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Langsmith configuration
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = "UDISE_School_Data_Chatbot"

# Initialize Langsmith Client and Tracer
langsmith_client = Client()
langchain_tracer = LangChainTracer(
    project_name="UDISE_School_Data_Chatbot",
    client=langsmith_client
)

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the LLM with Groq API key
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    You are a highly intelligent AI assistant specialized in analyzing UDISE school data.
    You have access to detailed information about schools, such as their location, infrastructure, facilities, and various operational details. This includes:
    - School Name
    - Location (State, District, Block, Village)
    - Infrastructure (Number of Classrooms, Building Condition)
    - Facilities (Playgrounds, Toilets, Drinking Water, Medical Checkups, etc.)
    - Management and Affiliation (School Management, Type, Affiliation Board, etc.)
    - Special Facilities (Ramp, Handrails, Anganwadi, etc.)

    Answer the user's questions based on the provided school data. If any information is missing or unclear, mention that politely.

    <context>
    {context}
    </context>

    Question: {input}

    Provide your answer in clear, concise, and natural language. If any information is not available in the data, please mention that and provide a relevant response.
    """
)

# Function to load JSON documents
def load_json_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                try:
                    json_data = json.load(file)
                    document_text = json.dumps(json_data, indent=2)
                    doc = Document(page_content=document_text, metadata={'source': filename})
                    documents.append(doc)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
    return documents

# Load documents and initialize FAISS vectors
loader_directory = "research_json"
documents = load_json_documents(loader_directory)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(documents)
vectors = FAISS.from_documents(final_documents, embeddings)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Analyze query route
@app.route('/analyze', methods=['POST'])
def analyze():
    user_prompt = request.form.get('query')

    if user_prompt:
        # Create document chain with Langsmith tracing
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        
        # Invoke chain with Langsmith tracer
        response = retrieval_chain.invoke(
            {'input': user_prompt}, 
            {'callbacks': [langchain_tracer]}
        )
        
        response_time = time.process_time() - start_time

        # Log additional metadata to Langsmith
        langsmith_client.create_run(
            project_name="UDISE_School_Data_Chatbot",
            name="School Data Query",
            run_type="chain",
            inputs={"query": user_prompt},
            outputs={"answer": response['answer'], "response_time": response_time},
            metadata={
                "model": "llama-3.1-8b-instant",
                "response_time": response_time,
                "source": "Flask App"
            }
        )

        return jsonify({
            'answer': response['answer'],
            'response_time': response_time
        })
    else:
        return jsonify({'error': 'No query provided'}), 400

if __name__ == '__main__':
    app.run(debug=True) 