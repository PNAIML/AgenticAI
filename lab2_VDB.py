from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

# -----------------------------
# Sample Documents
# -----------------------------
documents = [
    "Python is a programming language used to build software and AI applications.",
    "Machine Learning allows computers to learn patterns from data.",
    "Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
    "A vector database stores embeddings and retrieves similar information quickly.",
    "LangChain helps developers build applications using Large Language Models."
]

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading Embedding Model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -----------------------------
# Create Vector Database
# -----------------------------
print("Creating Vector Database...")
db = Chroma.from_texts(
    texts=documents,
    embedding=embeddings
)

# -----------------------------
# User Question
# -----------------------------
question = input("\nAsk your question: ")

# -----------------------------
# Retrieve Similar Documents
# -----------------------------
docs = db.similarity_search(question, k=2)

context = "\n".join([doc.page_content for doc in docs])

# -----------------------------
# Load LLM
# -----------------------------
llm = OllamaLLM(model="llama3")

# -----------------------------
# Prompt Template
# -----------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful teacher.

Use ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""
)

# -----------------------------
# Create Chain
# -----------------------------
chain = prompt | llm

print("\nGenerating answer... Please wait...\n")

# -----------------------------
# Generate Response
# -----------------------------
response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)

print("=" * 60)
print(response)
print("=" * 60)