import os
import re
import traceback
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_fireworks import ChatFireworks
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

# Load API key from .env
load_dotenv()
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
print("[DEBUG] FIREWORKS_API_KEY loaded:", bool(fireworks_api_key))

# Load and process FAQ
loader = TextLoader("data/omi_faq.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

# Embedding & Vector Store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.from_documents(documents, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Chat memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Fireworks DeepSeek LLM
llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3",
    api_key=fireworks_api_key,
    temperature=0.7,
    max_tokens=1024
)

# Greeting detection
greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you"]

# Prompts
CONDENSE_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI that rephrases follow-up questions into standalone questions.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant for an online shopping platform called OMI Live.

ONLY use the information in the context.
If the answer is not present, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
)

CLASSIFY_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant for OMI Live.

Determine if the following user question is related to OMI Live, its features, brands (like Siplit), livestreams, eco-conscious shopping, or anything mentioned in the FAQ.

Even if the question is vague or short (e.g., "By founders?", "Who started it?", "What about them?"), consider it related if it implies interest in OMI Live's leadership, mission, or platform.

Question: {question}

Answer with only "yes" or "no"."""
)

# Build chains
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=CONDENSE_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False
)

# Wrap LLM classification prompt
def classify_question(inputs: dict) -> dict:
    response = llm.invoke(CLASSIFY_PROMPT.format(**inputs))
    print(f"[DEBUG] Raw classification response: {response}")
    return {"text": response.content.strip().lower()}

classifier_chain = RunnableLambda(classify_question)

# Utility function
def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

# Main RAG function
def get_rag_response(question: str, chat_history: list) -> str:
    cleaned_q = clean_text(question)
    print(f"[DEBUG] Cleaned Question: {cleaned_q}")

    if any(cleaned_q.startswith(greet) for greet in greetings):
        return "Hey there! I'm here to help. Ask me anything about OMI Live."

    # Expand vague short queries
    if len(cleaned_q.split()) <= 3 and "founder" in cleaned_q:
        question = "What have the founders said about OMI Live?"

    try:
        classification_result = classifier_chain.invoke({"question": question})
        classification_text = classification_result.get("text", "")
        print(f"[DEBUG] Classification result: {classification_text}")
    except Exception as e:
        print("[ERROR] Classification step failed:", e)
        traceback.print_exc()
        return "⚠️ Error during classification step."

    if classification_text != "yes":
        return "I'm here to help with questions about OMI Live. Try asking about our platform, brands, or features!"

    try:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print(f"[DEBUG] QA Chain result: {result}")
    except Exception as e:
        print("[ERROR] QA step failed:", e)
        traceback.print_exc()
        return "⚠️ Error retrieving an answer."

    answer = result.get("answer", "I don't know.")
    low_answer = answer.lower()
    vague_phrases = ["i don't know", "no information", "not present", "no specific information"]

    if any(p in low_answer for p in vague_phrases) and len(cleaned_q.split()) <= 5:
        return "Could you please clarify your question about the founders or OMI Live?"

    return answer

# Optional test
if __name__ == "__main__":
    print("🔍 Testing LLM connectivity...")
    try:
        test = llm.invoke("Hello, are you online?")
        print("[TEST] DeepSeek LLM working ✅:", test)
    except Exception as e:
        print("[TEST] DeepSeek LLM failed ❌:", e)
