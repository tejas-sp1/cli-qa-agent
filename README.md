# Gutermann CLI Q&A Agent

A Python-based Retrieval-Augmented Generation (RAG) CLI agent
for answering questions about Gutermann water leak detection
products using the supplied product knowledge document.

The application runs entirely from the command line and uses
local embeddings and a local Ollama LLM.

---

## Features

- Parses the supplied Markdown product document
- Creates product-level knowledge chunks
- Preserves parent-section context
- Generates semantic embeddings using Sentence Transformers
- Performs hybrid retrieval using:
  - semantic similarity
  - lexical keyword overlap
- Uses a local Ollama LLM for answer generation
- Grounds answers in the retrieved document context
- Handles insufficient or incomplete information explicitly
- Displays retrieved product sources
- Supports an interactive CLI
- Supports `exit` and `quit` commands

---

## Architecture

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
                     /            \
                    /              \
           Semantic Score     Lexical Score
                    \              /
                     \            /
                      v          v
                       Hybrid Score
                            |
                            v
                     Top-K Products
                            |
                            v
                  Grounded Context
                            |
                            v
                    Ollama / Llama 3.2
                            |
                            v
                       CLI Answer