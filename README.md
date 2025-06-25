# AI_Chat_Bot
AI chat bot to be integrated with the Omi live site

# 1. Install Ollama
brew install ollama

# 2. Start Ollama service
ollama serve

# 3. Pull Mistral model
ollama pull mistral

# 4. (Optional) Pull LLaMA 3 model, but need this model for Text-to-design, it's good to have both the models installed
ollama pull llama3

# 5. Create and activate a Python virtual environment
python3 -m venv rag_env
source rag_env/bin/activate

# 6. Install required Python packages
pip install flask langchain langchain-community langchain-huggingface faiss-cpu

# 7. Start the chatbot
python app.py

# 8. Open in browser
open http://localhost:5050



Notes: 
1. The models are large and require atleast 16GB RAM
2. Sometimes we will face issue in downloading the models, it keeps on spinning, if such case arises, I would suggest using VPN (Urban VPN desktop) which is free and can connect to majority of the countires.
3. Keep ollama running in one terminal, Mistral/Llama3 running in another terminal so that the local host gets running.
