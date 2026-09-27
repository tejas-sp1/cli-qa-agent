# Gutermann CLI Q&A Agent

A Python-based Retrieval-Augmented Generation (RAG) command-line
assistant for answering questions about Gutermann water leak detection
products using the supplied product knowledge document.

The application runs entirely from the command line and uses:

- Markdown document ingestion
- Product-level chunking
- Sentence Transformer embeddings
- Hybrid semantic + lexical retrieval
- Local Ollama LLM
- Grounded answer generation

---

## 1. Problem Statement

The goal of this project is to build a CLI-based product knowledge
assistant that can answer questions using the supplied Gutermann
product documentation.

The application must:

1. Parse the supplied Markdown knowledge base.
2. Split the document into useful chunks.
3. Generate embeddings for the chunks.
4. Retrieve the most relevant chunks for each user query.
5. Pass the retrieved information to an LLM.
6. Generate answers grounded only in the supplied document.
7. Clearly indicate when the document does not provide enough
   information.
8. Support an interactive CLI with `exit` and `quit`.

---

## 2. Features

- Parses `product_overview.md` at startup
- Creates 14 product-level knowledge chunks
- Preserves parent section context
- Removes Markdown image references and HTML comments
- Generates 384-dimensional embeddings
- Stores embeddings in memory
- Uses cosine similarity for semantic retrieval
- Uses lexical keyword overlap as a second retrieval signal
- Combines semantic and lexical scores
- Uses a local Ollama `llama3.2` model for generation
- Restricts answers to retrieved document context
- Handles incomplete information
- Protects against unsupported product-feature transfer
- Displays retrieved product sources
- Provides an interactive command-line interface

---

## 3. Architecture

```text
                    product_overview.md
                            |
                            v
                   Markdown Parser
                            |
                            v
                  Product-level Chunks
                            |
                            v
                Sentence Transformer
                            |
                            v
                       Embeddings
                            |
                            v
                  In-memory Storage
                            |
                            |
                     User Question
                            |
                            v
                  Question Embedding
                            |
                            v
                   Hybrid Retrieval
                     /           \
                    /             \
           Semantic Score    Lexical Score
                    \             /
                     \           /
                      v         v
                     Hybrid Score
                            |
                            v
                       Top-K Chunks
                            |
                            v
                   Retrieved Context
                            |
                            v
                   Ollama / Llama 3.2
                            |
                            v
                     Grounded Answer
                            |
                            v
                          CLI