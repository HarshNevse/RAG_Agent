from flask import Flask, request, jsonify, render_template, Response

app = Flask(__name__)

import os

os.environ["FLASK_RUN_EXTRA_FILES"] = ""

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

from haystack.document_stores.in_memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()

from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
text_embedder.warm_up()



# pre-processing pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters.csv import CSVToDocument

file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown", "text/csv"])
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
pdf_converter = PyPDFToDocument()
csv_converter = CSVToDocument()
document_joiner = DocumentJoiner()

document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_embedder.warm_up()  # Load the embedding model into memory
document_writer = DocumentWriter(document_store)

from haystack import Pipeline

preprocessing_pipeline = Pipeline()
preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
preprocessing_pipeline.add_component(instance=csv_converter, name="csv_converter")
preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
preprocessing_pipeline.connect("file_type_router.text/csv", "csv_converter.sources")
preprocessing_pipeline.connect("text_file_converter", "document_joiner")
preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
preprocessing_pipeline.connect("markdown_converter", "document_joiner")
preprocessing_pipeline.connect("csv_converter", "document_joiner")
preprocessing_pipeline.connect("document_joiner", "document_cleaner")
preprocessing_pipeline.connect("document_cleaner", "document_splitter")
preprocessing_pipeline.connect("document_splitter", "document_embedder")
preprocessing_pipeline.connect("document_embedder", "document_writer")



from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
text_embedder.warm_up()

# Global variable to store the customizable part of the prompt
default_prompt_prefix = """ You are an advanced data analyst.
Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.

When responding to queries:
1. Reference specific evidence from the provided context to support your analysis
2. Explain terminology when introducing new concepts
3. Organize complex findings in a structured format for clarity
4. If you are uncertain about any aspect, clearly state which parts you are confident about versus where you're making educated assessments
5. If the answer cannot be determined from the provided context, respond with "I don't have sufficient information in this context to answer that question, please frame your question better."
"""

default_prompt_suffix = """Handling off-topic questions:
1. Only answer questions directly related to the information provided in the context
2. If asked a question unrelated to the logs (e.g., general knowledge questions, personal queries, or questions about unrelated systems), respond with:
   "I'm designed to analyze only the {data provided in the current context}. Please ask questions related to {this data}. For general information or questions about other topics, please consult a general-purpose assistant."
3. Do not attempt to answer questions that require knowledge outside the provided data, even if you possess such information
4. Redirect the conversation back to the data analysis task when appropriate

Based on the provided context (if any), answer the user's specific question in a direct and succinct manner. 
If there is no context provided, still answer the user's question **directly**. Do **not** create any follow-up questions or continue generating answers after the user’s question is fully addressed.
Ensure that you respond **only** to the user's **specific question** and avoid creating additional questions or answers on your own. Maintain a concise and respectful tone.

Context: {context}

Question: {question}

Answer:"""


# file upload route
import os
import shutil

# Define supported file types
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv", ".md"}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        files = request.files.getlist("files")
        if not files or all(file.filename == "" for file in files):
            return jsonify({"error": "No valid files selected"}), 400

        uploaded_files = []
        file_paths = []
        errors = {}

        # Temporary folder for initial storage
        temp_folder = os.path.join(app.config["UPLOAD_FOLDER"], "temp")
        os.makedirs(temp_folder, exist_ok=True)

        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()  # Get file extension
            if ext not in SUPPORTED_EXTENSIONS:
                errors[file.filename] = "Unsupported file format"
                continue  # Skip saving unsupported files

            try:
                file_path = os.path.join(temp_folder, file.filename)
                file.save(file_path)
                file_paths.append(file_path)  # Save path for processing
                uploaded_files.append(file.filename)
            except Exception as e:
                errors[file.filename] = f"File upload failed: {str(e)}"

        # If no valid files were uploaded, return an error
        if not file_paths:
            return jsonify({"error": "No valid files uploaded", "failed_files": errors}), 400

        try:
            # Run preprocessing pipeline
            preprocessing_pipeline.run({"file_type_router": {"sources": file_paths}})

            # If processing succeeds, move files to permanent location
            for file_path in file_paths:
                shutil.move(file_path, os.path.join(app.config["UPLOAD_FOLDER"], os.path.basename(file_path)))

            processed_files = uploaded_files  # Successfully processed files

        except Exception as e:
            # If processing fails, delete temp files
            for file_path in file_paths:
                os.remove(file_path)

            return jsonify({"error": "File processing failed", "details": str(e)}), 500

        return jsonify({
            "message": "Files uploaded and processed successfully",
            "processed_files": processed_files,
            "failed_files": errors,
        })

    return render_template("index.html")



@app.route("/update_prompt", methods=["POST"])
def update_prompt():
    global default_prompt_prefix
    data = request.json
    new_prompt = data.get("prompt")
    if not new_prompt:
        return jsonify({"error": "No prompt provided"}), 400
    default_prompt_prefix = new_prompt
    return jsonify({"message": "Prompt updated successfully"})


@app.route("/get_prompt_template", methods=["GET"])
def get_prompt_template():
    global default_prompt_prefix
    return jsonify({"template": default_prompt_prefix})


# rag_pipeline
from haystack.core.component import component
from haystack.dataclasses import Document
from typing import List


@component
class CustomPromptBuilder:
    @component.output_types(prompt=str)
    def run(self, documents: List[Document], question: str):
        context = "\n".join([doc.content for doc in documents])
        prompt = f"{default_prompt_prefix}\n{default_prompt_suffix}".replace(
            "{context}", context
        ).replace("{question}", question)
        return {"prompt": prompt}


custom_prompt_builder = CustomPromptBuilder()

from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.utils import Secret

chat_generator = HuggingFaceAPIGenerator(
    api_type="serverless_inference_api",
    api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3"},
    token=Secret.from_token(HF_API_KEY),
    generation_kwargs={"max_new_tokens": 1000},
)

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)

from haystack import Pipeline

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("custom_prompt_builder", custom_prompt_builder)
rag_pipeline.add_component("llm", chat_generator)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "custom_prompt_builder")
rag_pipeline.connect("custom_prompt_builder.prompt", "llm.prompt")


@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Use the already initialized pipeline instead of recreating it
        response = rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "custom_prompt_builder": {"question": question},
            }
        )

        return jsonify({"response": response["llm"]["replies"][0]})

    except Exception as e:
        print("Error:", str(e))  # Logs error in the terminal
        return jsonify({"error": str(e)}), 500  # Always return valid JSON


def generate():
    import time

    for i in range(10):
        yield f"data: Processing {i+1}/10\n\n"
        time.sleep(1)  # Simulate processing time
    yield "data: done\n\n"


@app.route("/progress")
def progress():
    return Response(generate(), mimetype="text/event-stream")


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"message": "API is running"})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
