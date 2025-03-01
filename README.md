# Retrieval-Augmented Generation (RAG) Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that enhances generative AI models with information retrieval capabilities. It allows users to upload documents, query the system, and receive responses based on both stored knowledge and real-time retrieval.

## Retrieval Augmented Generation
![image](https://github.com/user-attachments/assets/e7b2ca31-00da-4464-b97a-fbf224a03466)
![image](https://github.com/user-attachments/assets/09d4fdff-b58e-4340-b890-ab315092695b)


- **Text Embedder**: Embeds the user query into a vector representation. --->**Retriever**: Takes the query embedding, compares it with the stored document embeddings, and retrieves relevant documents.​
- **Retriever**: Relevant documents retrieved ---> **Prompt Builder**: Uses the retrieved documents and the original query to create a structured prompt for the language model. ​
- **Prompt Builder**: sends structured prompt to the llm ---> **LLM**: Uses the generated prompt to generate a response based on the context provided by the retrieved documents.​


## Features
- **File Upload**: Users can upload documents to populate the knowledge base.
- **Chat Assistant**: A conversational interface powered by RAG.
- **Prompt Management**: Save and manage prompts for better interactions.
- **Backend API Integration**: Uses a backend to handle file processing and retrieval.
- **User-Friendly Interface**: Simple and intuitive UI built with HTML, CSS, and JavaScript.

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (Flask/FastAPI/Django)
- **Database**: PostgreSQL / JSONPowerDB / SQLite (based on requirement)
- **AI Model**: OpenAI GPT / Llama / Custom Transformer Models
- **Vector Search**: FAISS / Pinecone / Weaviate

## 📂 Project Structure
```
📁 project-root
 ├── 📂 templates       # Contains index.html
 ├── 📄 app.py            # core python code wrapped in flask.
 ├── 📄 requirements.txt  # Dependencies
 ├── 📄 README.md      # Project Documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HarshNevse/RAG_Agent.git
   cd rag-project
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # For macOS/Linux
   env\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your HuggingFace API key:
    ```bash
   set HF_API_KEY=hf_your_api_key_here
   ```
5. Run the backend server:
   ```bash
   python app.py  # Flask example
   ```

## Usage
- **Upload Files**: Click on the upload section to add documents.
- **Ask Questions**: Type queries in the chat interface.
- **Retrieve Information**: The AI model will fetch and generate responses based on the documents.

## Frontend Preview
![image](https://github.com/user-attachments/assets/82f6cdba-507f-4fdf-a9dc-7dc165c112a8)


## Future Enhancements
- Multi-user authentication & access control
- Cloud storage support (AWS S3 / Firebase)
- Advanced document parsing with NLP techniques
- Support for multiple languages


---
**Author:** Harsh Nevse 
**GitHub:** [HarshNevse](https://github.com/HarshNevse)  
**Email:** harshnevse29@gmail.com

