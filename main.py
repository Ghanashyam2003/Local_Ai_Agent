from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import logging

# Setup logging for debugging and better feedback
logging.basicConfig(level=logging.INFO)

# Initialize the local model (Update this to the exact name used in your Ollama app)
model = OllamaLLM(model="llama3:instruct")  # Example name, adjust as needed

# Prompt Template with clear instruction and Mumbai context
template = """
You are a professional local guide and expert in answering questions about tourist attractions, food places, and hidden gems in Mumbai.

Based on the following reviews and knowledge base, answer the user's question in a helpful, friendly, and clear way.

Relevant reviews/documents:
{reviews}

User question:
{question}

Your answer:
"""

# Create the LangChain chain
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Loop for user interaction
print("\nWelcome to the Mumbai Tourist Assistant 🧭\n(Type 'q' to quit)\n")

while True:
    print("\n---------------------------------------------")
    question = input("Ask your question: ").strip()
    print("---------------------------------------------\n")
    
    if question.lower() == "q":
        print("Goodbye! Have a great day! 🌞")
        break

    try:
        # Step 1: Retrieve relevant documents
        reviews = retriever.invoke(question)

        # Step 2: Ask the model
        result = chain.invoke({"reviews": reviews, "question": question})

        # Step 3: Print result
        print("🧠 AI Recommendation:")
        print(result)
    
    except Exception as e:
        logging.error(f"Something went wrong: {e}")
        print("⚠️ Sorry, there was an error processing your request.")
