# Doctor assistant

Doctor assistant is a medical chatbot that answers questions using information from medical documents (PDFs). It uses AI to search through the documents and give short, clear answers.

## How it works

1. Medical PDF files are loaded from the `data` folder.
2. The text is split into small chunks.
3. Each chunk is turned into an embedding (a number representation of the text) using a HuggingFace model.
4. The embeddings are stored in a Pinecone vector database.
5. When a question is asked, the bot searches Pinecone for the most relevant chunks and uses an OpenAI language model to write an answer based on them.

## Tech stack

- Python
- LangChain
- Pinecone (vector database)
- HuggingFace Sentence Transformers (embeddings)
- OpenAI (language model)
- FastAPI (planned for the web app)

## Project structure

```
medibot/
├── data/            # Medical PDF files
├── src/
│   ├── helper.py    # Functions to load PDFs, split text, and create embeddings
│   └── prompt.py     # The instructions given to the AI
├── research/        # Notebook for experiments
├── store_index.py   # Script to build the Pinecone index from the PDFs
└── app.py           # The chatbot app (in progress)
```

## Setup

1. Clone this repository and go into the project folder.
2. Create a virtual environment and install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   PINECONE_API_KEY=your_pinecone_key
   OPENAI_API_KEY=your_openai_key
   ```
4. Put your medical PDF files in the `data` folder.
5. Build the Pinecone index:
   ```
   python store_index.py
   ```

## Status

This project is a work in progress. The document search and indexing part works. The chatbot app (`app.py`) is still being built.

