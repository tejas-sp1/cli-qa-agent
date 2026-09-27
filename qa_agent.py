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
# 1. PARSE THE PRODUCT DOCUMENT
# =============================================================

def parse_markdown_products(file_path):
    """
    Read the Markdown product document and convert it into
    one structured chunk for each product.
    """

    document_path = Path(file_path)

    # Check that the product document exists
    if not document_path.exists():
        raise FileNotFoundError(
            f"Product document not found: {document_path}"
        )

    # Read the complete Markdown file
    text = document_path.read_text(encoding="utf-8")

    # Split the document into individual lines
    lines = text.splitlines()

    # Store all product chunks
    chunks = []

    # Track our current location
    current_section = None
    current_product = None

    # Product-specific information
    current_content = []

    # Information shared by a section
    section_context = []

    # One section in the supplied document does not have
    # the normal Markdown "##" marker.
    known_sections = {
        "Acoustic Leak Detection Microphones"
    }

    # ---------------------------------------------------------
    # Helper function
    # ---------------------------------------------------------

    def save_current_product():
        """
        Save the currently active product as one chunk.
        """

        if current_product is None:
            return

        product_text = "\n".join(
            current_content
        ).strip()

        shared_context = "\n".join(
            section_context
        ).strip()

        chunks.append(
            {
                "section": current_section,
                "product": current_product,
                "section_context": shared_context,
                "content": product_text
            }
        )

    # ---------------------------------------------------------
    # Process the document line by line
    # ---------------------------------------------------------

    for raw_line in lines:

        # Remove spaces around the line
        line = raw_line.strip()

        # Ignore empty lines
        if not line:
            continue

        # Remove invisible Unicode characters
        line = (
            line.replace("\ufeff", "")
                .replace("\u200b", "")
                .strip()
        )

        # Ignore HTML comments
        if line.startswith("<!--"):
            continue

        # Ignore Markdown image references
        if line.startswith("!["):
            continue

        # -----------------------------------------------------
        # Normal section
        #
        # Example:
        # ## Real-Time Correlators For All Conditions
        # -----------------------------------------------------

        if line.startswith("## "):

            save_current_product()

            current_section = line[3:].strip()

            current_product = None
            current_content = []
            section_context = []

        # -----------------------------------------------------
        # Special section without "##"
        #
        # Example:
        # Acoustic Leak Detection Microphones
        # -----------------------------------------------------

        elif line in known_sections:

            save_current_product()

            current_section = line

            current_product = None
            current_content = []
            section_context = []

        # -----------------------------------------------------
        # Product heading
        #
        # Example:
        # ### AQUASCOPE 3
        # -----------------------------------------------------

        elif line.startswith("### "):

            save_current_product()

            current_product = line[4:].strip()

            current_content = []

        # -----------------------------------------------------
        # Product content
        # -----------------------------------------------------

        elif current_product is not None:

            current_content.append(line)

        # -----------------------------------------------------
        # Section-level information
        # -----------------------------------------------------

        else:

            section_context.append(line)

    # Save the final product
    save_current_product()

    return chunks


# =============================================================
# 2. PREPARE CHUNK TEXT
# =============================================================

def prepare_chunk_text(chunk):
    """
    Convert a structured product chunk into one text block
    for the embedding model.
    """

    text = f"""
Section: {chunk["section"]}

Product: {chunk["product"]}

Section Context:
{chunk["section_context"]}

Product Content:
{chunk["content"]}
""".strip()

    return text


# =============================================================
# 3. CREATE EMBEDDINGS
# =============================================================

def create_embeddings(chunks):
    """
    Create one embedding vector for every product chunk.
    """

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Convert structured chunks into plain text
    texts = [
        prepare_chunk_text(chunk)
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} product chunks..."
    )

    # Convert text into numerical vectors
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=float
    )

    return model, embeddings


# =============================================================
# 4. HYBRID RETRIEVAL
# =============================================================

def retrieve_products(
    query,
    model,
    chunks,
    embeddings,
    top_k=TOP_K
):
    """
    Retrieve the most relevant products using a hybrid score.

    Hybrid score:
        65% semantic similarity
        35% lexical overlap
    """

    # ---------------------------------------------------------
    # STEP 1: Embed the user's question
    # ---------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # ---------------------------------------------------------
    # STEP 2: Semantic similarity
    # ---------------------------------------------------------

    # Because both embeddings are normalized,
    # dot product is equivalent to cosine similarity.
    semantic_scores = (
        embeddings @ query_embedding
    )

    # ---------------------------------------------------------
    # STEP 3: Stop words
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # STEP 4: Extract meaningful query words
    # ---------------------------------------------------------

    query_words = {
        word.lower().strip(".,!?;:")
        for word in query.split()
        if word.lower().strip(".,!?;:")
        not in stop_words
    }

    # ---------------------------------------------------------
    # STEP 5: Lexical overlap
    # ---------------------------------------------------------

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
                matched_words / len(query_words)
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

    # ---------------------------------------------------------
    # STEP 6: Calculate hybrid score
    # ---------------------------------------------------------

    hybrid_scores = (
        0.65 * semantic_scores
        + 0.35 * lexical_scores
    )

    # ---------------------------------------------------------
    # STEP 7: Rank products
    # ---------------------------------------------------------

    ranked_indices = np.argsort(
        hybrid_scores
    )[::-1]

    # ---------------------------------------------------------
    # STEP 8: Build result objects
    # ---------------------------------------------------------

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
# 5. DISPLAY RETRIEVAL RESULTS
# =============================================================

def display_results(query, results):
    """
    Display retrieved products and their scores.
    """

    print("\n" + "=" * 70)

    print("User question:")
    print(query)

    print("\nTop retrieved products:\n")

    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        print(
            f"{rank}. {chunk['product']}"
        )

        print(
            f"   Section: {chunk['section']}"
        )

        print(
            f"   Hybrid score: "
            f"{result['score']:.4f}"
        )

        print(
            f"   Semantic score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"   Lexical score: "
            f"{result['lexical_score']:.4f}"
        )

        print(
            f"   Content: "
            f"{chunk['content'][:300]}"
        )

        print()


# =============================================================
# 6. GENERATE GROUNDED ANSWER WITH OLLAMA
# =============================================================

# =============================================================
# 6. GENERATE GROUNDED ANSWER WITH OLLAMA
# =============================================================

# =============================================================
# 6. GENERATE GROUNDED ANSWER WITH OLLAMA
# =============================================================

def generate_answer(query, results):
    """
    Generate a grounded answer using only the retrieved
    product-document context.
    """

    # ---------------------------------------------------------
    # STEP 1: Build context from retrieved chunks
    # ---------------------------------------------------------

    context_parts = []

    for number, result in enumerate(results, start=1):

        chunk = result["chunk"]

        context = f"""
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

        context_parts.append(context)

    context = (
        "\n\n-------------------------\n\n"
        .join(context_parts)
    )

    # ---------------------------------------------------------
    # STEP 2: Grounding prompt
    # ---------------------------------------------------------

    prompt = f"""
You are a product knowledge assistant for Gutermann.

Answer the USER QUESTION using ONLY the PRODUCT DOCUMENT
CONTEXT below.

You must not use outside knowledge, assumptions, or guesses.

IMPORTANT GROUNDING RULES:

1. Use all relevant information in every supplied source,
   including Section Context.

2. A product can inherit characteristics stated in its
   parent product section when the document structure places
   that product under that section.

3. For requirement or "what fits?" questions:
   compare the user's requirements against the documented
   product characteristics.

   If the document explicitly provides the required
   characteristics, you may state that the product matches
   those requirements.

   You do NOT need the document to literally say:
   "this product is suitable for this use case."

4. Do not claim a product has a feature merely because
   another unrelated product has that feature.

5. Do not say information is missing when it is explicitly
   present in the supplied context.

6. Do not infer that a product lacks a feature merely because
   that feature is not mentioned in its individual description.

   Instead say:
   "The document explicitly mentions this for X, but does not
   mention it for Y."

7. For comparison questions, report only documented differences.

8. For current availability or ordering questions:
   do not invent current availability.
   If the document gives launch or pre-series information but
   does not confirm current ordering status, report those facts
   and clearly state that current ordering is not confirmed.

9. If the document provides some relevant information but does
   not completely answer the question, provide the documented
   facts and clearly state the remaining limitation.

10. If the document genuinely contains no relevant information,
    say:

    "The product document does not provide enough information
    to answer that."

IMPORTANT EXAMPLE:

Question:
"We want permanent monitoring in underground chambers
with no drilling. What fits?"

The document section states that the products are:
- permanently installed correlating leak-noise loggers
- connected from underground chambers to the cloud
- designed so no drilling of holes is required

ZONESCAN AI and ZONESCAN HYDRO are listed under that
permanent monitoring section.

Therefore, answer that both products match the documented
requirements, and explain the relevant documented facts.

IMPORTANT EXAMPLE:

Question:
"Does the ZONESCAN AI use hydrophone technology?"

If the document describes hydrophone technology for
ZONESCAN HYDRO but describes an internal accelerometer
for ZONESCAN AI, do NOT transfer the hydrophone capability
to ZONESCAN AI.

PRODUCT DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

Answer using only the supplied context.
    """.strip()

    # ---------------------------------------------------------
    # STEP 3: Send request to Ollama
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # STEP 4: Check response
    # ---------------------------------------------------------

    response.raise_for_status()

    # ---------------------------------------------------------
    # STEP 5: Extract answer
    # ---------------------------------------------------------

    data = response.json()

    return data["message"]["content"].strip()

# =============================================================
# 7. MAIN PROGRAM
# =============================================================

# =============================================================
# 6. MAIN PROGRAM
# =============================================================

def main():

    print(
        "Starting Gutermann Product Q&A Agent...\n"
    )

    # ---------------------------------------------------------
    # STEP 1: Read and parse the product document
    # ---------------------------------------------------------

    print("Reading product document...")

    chunks = parse_markdown_products(
        PRODUCT_DOCUMENT
    )

    print(
        f"Product parsing completed. "
        f"Found {len(chunks)} product chunks."
    )

    # Safety check
    if not chunks:
        raise RuntimeError(
            "Product parsing did not return any chunks."
        )

    # ---------------------------------------------------------
    # STEP 2: Create embeddings
    # ---------------------------------------------------------

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
        "Each product is represented by a "
        f"{embeddings.shape[1]}-dimensional vector."
    )

    # ---------------------------------------------------------
    # STEP 3: Start interactive CLI
    # ---------------------------------------------------------

    print(
        "\nGutermann Product Q&A Agent is ready."
    )

    print(
        "Ask a question about the products."
    )

    print(
        "Type 'exit' or 'quit' to end the session.\n"
    )

    # ---------------------------------------------------------
    # STEP 4: Interactive question loop
    # ---------------------------------------------------------

    while True:

        try:
            query = input("> ").strip()

        except KeyboardInterrupt:
            print("\n\nGoodbye.")
            break

        except EOFError:
            print("\n\nGoodbye.")
            break

        # -----------------------------------------------------
        # Ignore empty input
        # -----------------------------------------------------

        if not query:
            continue

        # -----------------------------------------------------
        # Exit commands
        # -----------------------------------------------------

        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # -----------------------------------------------------
        # Retrieve relevant products
        # -----------------------------------------------------

        results = retrieve_products(
            query=query,
            model=model,
            chunks=chunks,
            embeddings=embeddings,
            top_k=TOP_K
        )

        # -----------------------------------------------------
        # Generate grounded answer
        # -----------------------------------------------------

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
                "\nAgent: An error occurred while generating "
                "the answer."
            )

            print(
                f"Details: {error}\n"
            )

            continue

        # -----------------------------------------------------
        # Display answer
        # -----------------------------------------------------

        print("\nAgent:")
        print(answer)

        # -----------------------------------------------------
        # Optional source display
        # -----------------------------------------------------

        print("\nSources:")

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