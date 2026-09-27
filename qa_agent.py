from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# =============================================================
# CONFIGURATION
# =============================================================

PRODUCT_DOCUMENT = "product_overview.md"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:latest"

TOP_K = 3


# =============================================================
# DOCUMENT PARSING
# =============================================================

def parse_markdown_products(file_path):
    """
    Read the Markdown document and create one structured
    chunk per product.
    """

    document_path = Path(file_path)

    if not document_path.exists():
        raise FileNotFoundError(
            f"Product document not found: {document_path}"
        )

    text = document_path.read_text(
        encoding="utf-8"
    )

    lines = text.splitlines()

    chunks = []

    current_section = None
    current_product = None
    current_content = []
    section_context = []

    # One heading in the supplied document does not contain
    # the normal Markdown "##" prefix.
    known_sections = {
        "Acoustic Leak Detection Microphones"
    }

    def save_current_product():
        """Save the current product as a structured chunk."""

        if current_product is None:
            return

        chunks.append(
            {
                "section": current_section,
                "product": current_product,
                "section_context": "\n".join(
                    section_context
                ).strip(),
                "content": "\n".join(
                    current_content
                ).strip(),
            }
        )

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        line = (
            line.replace("\ufeff", "")
                .replace("\u200b", "")
                .strip()
        )

        # Ignore comments and image references.
        if line.startswith("<!--"):
            continue

        if line.startswith("!["):
            continue

        # Detect normal section headings.
        if line.startswith("## "):

            save_current_product()

            current_section = line[3:].strip()
            current_product = None
            current_content = []
            section_context = []

        # Detect the special section without "##".
        elif line in known_sections:

            save_current_product()

            current_section = line
            current_product = None
            current_content = []
            section_context = []

        # Detect product headings.
        elif line.startswith("### "):

            save_current_product()

            current_product = line[4:].strip()
            current_content = []

        # Product information.
        elif current_product is not None:

            current_content.append(line)

        # Section-level information.
        else:

            section_context.append(line)

    # Save the final product.
    save_current_product()

    return chunks


# =============================================================
# TEXT PREPARATION
# =============================================================

def prepare_chunk_text(chunk):
    """
    Convert a structured product chunk into a single
    text block suitable for embedding.
    """

    return f"""
Section: {chunk["section"]}

Product: {chunk["product"]}

Section Context:
{chunk["section_context"]}

Product Content:
{chunk["content"]}
""".strip()


# =============================================================
# EMBEDDINGS
# =============================================================

def create_embeddings(chunks):
    """
    Create one normalized embedding vector per product.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        prepare_chunk_text(chunk)
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} product chunks..."
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return model, np.asarray(
        embeddings,
        dtype=float
    )


# =============================================================
# HYBRID RETRIEVAL
# =============================================================

def retrieve_products(
    query,
    model,
    chunks,
    embeddings,
    top_k=TOP_K
):
    """
    Retrieve relevant products using:

    65% semantic similarity
    35% lexical overlap
    """

    # Embed the question.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Semantic similarity.
    # Normalized vectors make dot product equivalent
    # to cosine similarity.
    semantic_scores = (
        embeddings @ query_embedding
    )

    stop_words = {
        "i",
        "need",
        "to",
        "find",
        "a",
        "the",
        "on",
        "for",
        "what",
        "do",
        "you",
        "recommend",
        "can",
        "please",
        "we",
        "our",
        "is",
        "are",
        "and",
        "does",
        "which",
        "has",
        "have",
        "it",
        "this",
        "that"
    }

    query_words = {
        word.lower().strip(".,!?;:")
        for word in query.split()
        if word.lower().strip(".,!?;:")
        not in stop_words
    }

    lexical_scores = []

    for chunk in chunks:

        chunk_text = " ".join(
            [
                chunk["section"],
                chunk["product"],
                chunk["section_context"],
                chunk["content"]
            ]
        ).lower()

        matched_words = sum(
            1
            for word in query_words
            if word in chunk_text
        )

        if query_words:
            lexical_score = (
                matched_words /
                len(query_words)
            )
        else:
            lexical_score = 0.0

        lexical_scores.append(
            lexical_score
        )

    lexical_scores = np.asarray(
        lexical_scores,
        dtype=float
    )

    # Hybrid retrieval score.
    hybrid_scores = (
        0.65 * semantic_scores
        + 0.35 * lexical_scores
    )

    ranked_indices = np.argsort(
        hybrid_scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "chunk": chunks[index],
                "semantic_score": float(
                    semantic_scores[index]
                ),
                "lexical_score": float(
                    lexical_scores[index]
                ),
                "score": float(
                    hybrid_scores[index]
                )
            }
        )

    return results


# =============================================================
# OLLAMA ANSWER GENERATION
# =============================================================

def generate_answer(query, results):
    """
    Generate a grounded answer using only the retrieved
    product-document context.
    """

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        context_parts.append(
            f"""
SOURCE {number}

Section:
{chunk["section"]}

Product:
{chunk["product"]}

Section Context:
{chunk["section_context"]}

Product Content:
{chunk["content"]}
""".strip()
        )

    context = (
        "\n\n-------------------------\n\n"
        .join(context_parts)
    )

    prompt = f"""
You are a product knowledge assistant for Gutermann.

Answer the USER QUESTION using ONLY the PRODUCT DOCUMENT
CONTEXT below.

You must not use outside knowledge, assumptions, or guesses.

GROUNDING RULES:

1. Use all relevant information in every supplied source,
   including Section Context.

2. A product can inherit characteristics stated in its
   parent section when the document structure places that
   product under that section.

3. For "what fits?" questions, compare the user's
   requirements with the documented product characteristics.

4. Do not claim that a product has a feature merely because
   another product has that feature.

5. Do not say information is missing when it is explicitly
   present in the context.

6. Do not infer that a product lacks a feature simply because
   the document does not mention that feature for the product.

7. For comparison questions, report only documented
   differences.

8. For current availability or ordering questions:
   - do not invent current availability
   - do not answer YES unless the document confirms it
   - do not answer NO unless the document confirms it
   - report documented launch or availability information
     and clearly state when current status is not confirmed

9. If the document provides partial information, provide
   those documented facts and explain the limitation.

10. If the document contains no relevant information, say:

"The product document does not provide enough information
to answer that."

PRODUCT DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

Answer using only the supplied context.
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"].strip()


# =============================================================
# INTERACTIVE CLI
# =============================================================

def main():

    print(
        "Starting Gutermann Product Q&A Agent...\n"
    )

    # Load the knowledge base.
    print("Reading product document...")

    chunks = parse_markdown_products(
        PRODUCT_DOCUMENT
    )

    if not chunks:
        raise RuntimeError(
            "Product parsing did not return any chunks."
        )

    print(
        f"Product parsing completed. "
        f"Found {len(chunks)} product chunks."
    )

    # Create embeddings once at startup.
    model, embeddings = create_embeddings(
        chunks
    )

    print(
        "\nEmbedding creation completed."
    )

    print(
        "Embedding matrix shape:",
        embeddings.shape
    )

    print(
        "\nGutermann Product Q&A Agent is ready."
    )

    print(
        "Ask a question about the products."
    )

    print(
        "Type 'exit' or 'quit' to end the session.\n"
    )

    while True:

        try:
            query = input("> ").strip()

        except KeyboardInterrupt:
            print("\n\nGoodbye.")
            break

        except EOFError:
            print("\n\nGoodbye.")
            break

        # Ignore empty input.
        if not query:
            continue

        # Exit commands.
        if query.lower() in {
            "exit",
            "quit"
        }:
            print("Goodbye.")
            break

        # Retrieve relevant products.
        results = retrieve_products(
            query=query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=TOP_K
        )

        # Generate answer.
        try:

            answer = generate_answer(
                query,
                results
            )

        except requests.exceptions.ConnectionError:

            print(
                "\nAgent: Unable to connect to Ollama."
            )

            print(
                "Make sure Ollama is running and try again.\n"
            )

            continue

        except requests.exceptions.Timeout:

            print(
                "\nAgent: Ollama took too long to respond."
            )

            print(
                "Please try the question again.\n"
            )

            continue

        except Exception as error:

            print(
                "\nAgent: An error occurred while "
                "generating the answer."
            )

            print(
                f"Details: {error}\n"
            )

            continue

        # Display answer.
        print("\nAgent:")
        print(answer)

        # Display retrieval sources.
        print("\nRetrieved sources:")

        for result in results:

            chunk = result["chunk"]

            print(
                f"- {chunk['product']} "
                f"({chunk['section']})"
            )

        print()


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()