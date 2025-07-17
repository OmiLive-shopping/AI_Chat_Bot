# rag_chain.py

import os, re, traceback, warnings, time
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_fireworks import ChatFireworks
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory

# === Environment Setup ===
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
load_dotenv()

fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
print("[DEBUG] FIREWORKS_API_KEY loaded:", bool(fireworks_api_key))

# === Retriever Setup (with fallback) ===
def get_retriever():
    try:
        # Use smaller embedding model for low memory footprint
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        vectorstore = FAISS.load_local("data/faiss_index", embedding_model)
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    except Exception as e:
        print("[ERROR] Retriever setup failed:", e)
        traceback.print_exc()
        raise RuntimeError("❌ FAISS retriever could not be initialized.")

# === LLM and Memory Setup ===
llm = ChatFireworks(
    model="accounts/fireworks/models/deepseek-v3",
    api_key=fireworks_api_key,
    temperature=0.7,
    max_tokens=1024
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === Prompts ===
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

Determine if the user question relates to OMI Live’s mission, livestream platform, eco-friendly brands, shopping practices, FAQ content, or educational content in the internal knowledge base — including hair care, skincare, body care, sustainable living, food tips, eco-friendly products, tampons, bees, and bamboo.

Even if the question is vague or short (e.g., "hair issues?", "eco laundry?"), consider it related if the topic is found in the FAQ or internal knowledge base.

Question: {question}

Answer with only "yes" or "no"."""
)

# === Classifier Chain ===
def classify_question(inputs: dict) -> dict:
    try:
        response = llm.invoke(CLASSIFY_PROMPT.format(**inputs))
        print(f"[DEBUG] Classification Raw: {response}")
        return {"text": response.content.strip().lower()}
    except Exception as e:
        print("[ERROR] Classifier failed:", e)
        traceback.print_exc()
        return {"text": "no"}

classifier_chain = RunnableLambda(classify_question)

# === Utilities ===
def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you"]

# === RAG Response Logic ===
qa_chain = None

def get_rag_response(question: str, chat_history: list) -> str:
    global qa_chain
    cleaned_q = clean_text(question)
    print(f"[DEBUG] Cleaned Question: {cleaned_q}")

    if any(cleaned_q.startswith(greet) for greet in greetings):
        return "Hey there! I'm here to help. Ask me anything about OMI Live."

    if len(cleaned_q.split()) <= 3 and "founder" in cleaned_q:
        question = "What have the founders said about OMI Live?"

    try:
        classification_result = classifier_chain.invoke({"question": question})
        classification_text = classification_result.get("text", "")
        print(f"[DEBUG] Classification: {classification_text}")
    except Exception as e:
        print("[ERROR] Classification step failed:", e)
        return "⚠️ Error during classification."

    if classification_text != "yes":
        return "I'm here to help with questions about OMI Live. Try asking about our platform, brands, or features!"

    if qa_chain is None:
        try:
            retriever = get_retriever()
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_PROMPT,
                combine_docs_chain_kwargs={"prompt": QA_PROMPT},
                return_source_documents=False
            )
        except Exception as e:
            print("[ERROR] QA chain initialization failed:", e)
            return "⚠️ Internal error during response setup."

    try:
        result = qa_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        print(f"[DEBUG] QA Response: {result}")
    except Exception as e:
        print("[ERROR] QA step failed:", e)
        return "⚠️ Error retrieving an answer."

    answer = result.get("answer", "I don't know.")
    low_answer = answer.lower()
    vague_phrases = ["i don't know", "no information", "not present", "no specific information"]

    if any(p in low_answer for p in vague_phrases) and len(cleaned_q.split()) <= 5:
        return "Could you please clarify your question about the founders or OMI Live?"

    time.sleep(1.0)
    return answer

# === LLM Connectivity Test ===
if __name__ == "__main__":
    print("🔍 Testing LLM connectivity...")
    try:
        test = llm.invoke("Hello, are you online?")
        print("[TEST] DeepSeek working ✅:", test)
    except Exception as e:
        print("[TEST] DeepSeek failed ❌:", e)
