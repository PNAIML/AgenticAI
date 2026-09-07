from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="llama3")

prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are a helpful teaching assistant.\n"
        "Explain the concept of '{topic}' in 3 deepresearch sentences, "
        "suitable for an expert scientist."
    ),
)

chain = prompt | llm
topic = "Artificial Brain and Its Power"

print("=" * 60)
print("🤖 LangChain + Ollama Demo")
print("=" * 60)
print(f"\n📚 Topic: {topic}")
print("\n🚀 Sending your request to the local AI model...")
print("💡 The AI model is analysing your question.")
print("⏳ Please wait...\n")

response = chain.invoke({"topic": topic})

print("✅ Explanation generated successfully!\n")
print(response)