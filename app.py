from flask import Flask, request, jsonify, render_template, Response

app = Flask(__name__)

import os

os.environ["FLASK_RUN_EXTRA_FILES"] = ""

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

from haystack.document_stores.in_memory import InMemoryDocumentStore

document_store = InMemoryDocumentStore()

from haystack.components.embedders import SentenceTransformersDocumentEmbedder

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
document_embedder.warm_up()  # Load the embedding model into memory

from haystack.components.embedders import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
text_embedder.warm_up()

# Global variable to store the customizable part of the prompt
default_prompt_prefix = """You are an advanced data analyst for analyzing logs with access to external information. 
1. If you are unsure, use the provided context to answer the question.
2. If the answer is not in the context, reply with \"I don't know.\" """

default_prompt_suffix = """Given the following contextual data, completely utilize it and provide a comprehensive answer to the user question.
Context:
{context}

Question: {question}
Answer:"""


# file upload route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "files" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        files = request.files.getlist("files")  # Get multiple files
        if not files or all(file.filename == "" for file in files):
            return jsonify({"error": "No valid files selected"}), 400

        processed_files = []
        errors = {}

        for file in files:
            try:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(file_path)
                processed_files.append(file.filename)

                from haystack.components.converters import TextFileToDocument

                converter = TextFileToDocument()
                text_documents = converter.run(sources=[file_path])["documents"]

                from haystack.components.preprocessors import DocumentSplitter

                document_splitter = DocumentSplitter(
                    split_by="word", split_length=250, split_overlap=25
                )
                splitted_docs = document_splitter.run(text_documents)["documents"]

                # Embed documents using global document_embedder
                embedded_docs = document_embedder.run(splitted_docs)

                from haystack.components.writers import DocumentWriter

                document_writer = DocumentWriter(document_store)
                document_writer.run(embedded_docs["documents"])

            except Exception as e:
                errors[file.filename] = str(e)

        # If all files failed, return an error
        if len(errors) == len(files):
            return (
                jsonify({"error": "All files failed to process", "details": errors}),
                500,
            )

        # Return JSON response with success and failure details
        return jsonify(
            {
                "message": "Files uploaded and indexed successfully",
                "processed_files": processed_files,
                "failed_files": errors,
            }
        )

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
