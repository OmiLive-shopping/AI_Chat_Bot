import re
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory

# 1. Load FAQ document
loader = TextLoader("data/omi_faq.txt", encoding="utf-8")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = splitter.split_documents(docs)

# 3. Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.from_documents(documents, embedding_model)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# 4. Chat memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 5. Language model
llm = OllamaLLM(model="mistral")

# 6. Greeting detection
greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "how are you"]

# 7. Prompts
CONDENSE_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI that rephrases follow-up questions into standalone questions.

Given the chat history and the user's next input, rewrite the input into a complete question that clearly refers to any previously mentioned entities or topics, especially "founders" or "Omi Live" if relevant.

Always replace vague references like "they", "it", or "that" with specific names or topics from the conversation.

If the user's input is too vague or just a pronoun, incorporate context to clarify.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant for an online shopping platform called OMI Live.

ONLY use the information in the context.
If the answer is not present, say "I don't know."
Never change brand names or spellings. For example, "Siplit" must remain exactly as "Siplit".

When possible, quote the exact sentence from the context.

Context:
{context}

Question: {question}
Answer:"""
)

CLASSIFY_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant for OMI Live.

Determine if the following user question is related to OMI Live, its features, brands (like Siplit), or shopping experience.

If the question is about Siplit, livestreams, eco-conscious shopping, or anything mentioned in the FAQ, consider it related.

Question: {question}

Answer with only "yes" or "no"."""
)

# 8. Build QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    condense_question_prompt=CONDENSE_PROMPT,
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False
)

# 9. Build classification chain
classifier_chain = LLMChain(llm=llm, prompt=CLASSIFY_PROMPT)

# Utility: clean input text for classification and checks
def clean_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()

# 10. Main wrapper
def get_rag_response(question, chat_history):
    cleaned_q = clean_text(question)

    # Early greeting response
    if any(cleaned_q.startswith(greet) for greet in greetings):
        return "Hey there! I'm here to help. Ask me anything about OMI Live."

    # Classify question relevance
    classification = classifier_chain.run(question=question).strip().lower()
    if classification != "yes":
        return "I'm here to help with questions about OMI Live. Try asking about our platform, brands, or features!"

    # Use RAG chain with question condensation
    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    # Check if answer is meaningful or fallback
    answer = result.get("answer", "I don't know.")
    low_answer = answer.lower()
    vague_answers = [
        "i don't know",
        "there is no mention",
        "not present",
        "no information",
        "no specific information"
    ]
    # If the answer is vague and question is short/vague, ask for clarification
    if any(phrase in low_answer for phrase in vague_answers) and len(cleaned_q.split()) <= 5:
        return "Could you please clarify your question about the founders or Omi Live?"

    return answer
