# Retrieval-Augmented Generation (RAG) Agent Project

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
- **Backend**: Python (Flask)
- **Database**: InMemory
- **AI Model**: Mistral Instruct 7b v0.3 via HuggingFace
- **Vector Search**: sentence-transformers/all-MiniLM-L6-v2

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
![9_](https://github.com/user-attachments/assets/788d2d64-c677-4996-9074-35cb717a914a)


## Changelog
- 12/03/2025: Added support for multiple file types (PDF, CSV, Markdown etc.)


## Future Enhancements
- Multi-user authentication & access control
- Cloud storage support (AWS S3 / Firebase)
- Advanced document parsing with NLP techniques
- Support for multiple languages


---
**Author:** Harsh Nevse 
**GitHub:** [HarshNevse](https://github.com/HarshNevse)  
**Email:** harshnevse29@gmail.com

