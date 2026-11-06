# 🏮 Thai Festivals & Traditions Chatbot

**Topic:** We give an AI a collection of local documents about Thai festivals (Songkran, Loy Krathong, Yi Peng, Bun Bang Fai, Phi Ta Khon) and study how accurately it answers questions about their **history, rituals, dates and regional differences**, compared with the same AI **without** those documents.

- **Interface:** Python + Streamlit
- **AI:** Local LLM through [Ollama](https://ollama.com) (no API key needed, runs on your own computer)
- **Method:** RAG (Retrieval-Augmented Generation): find the most relevant paragraphs in `docs/` and give them to the AI with the question

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit chatbot (3 modes: with documents / without documents / compare both) |
| `rag.py` | Loads documents, searches for relevant paragraphs, calls Ollama |
| `docs/*.txt` | Our knowledge base: one file per festival |
| `questions.csv` | Test questions + expected answers (dates, history, rituals, regional, out-of-scope) |
| `evaluate.py` | Runs all test questions in both modes and saves `results.csv` |
| `requirements.txt` | Python libraries |

## How to run

1. Install Ollama from https://ollama.com and download a model:
   ```bash
   ollama pull qwen2.5:7b
   ```
   (Slower or older laptop? Use `qwen2.5:3b` and type it in the app sidebar.)
2. Install the Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the chatbot:
   ```bash
   streamlit run app.py
   ```
4. Run the comparison test:
   ```bash
   python evaluate.py
   ```
   Then open `results.csv` and score each answer: **2 = correct, 1 = partly correct, 0 = wrong / made up**.

## How it works

1. Each file in `docs/` is split into paragraphs.
2. When a user asks a question, the app finds the top 4 most similar paragraphs (TF-IDF with character n-grams, which works for both English and Thai).
3. **With documents:** the AI gets those paragraphs + the question and is told to answer only from them, and to say "I don't know" if the answer is not there.
4. **Without documents:** the AI gets only the question and answers from its own knowledge.
5. We compare both answers to the expected answers in `questions.csv`.

## Adding more documents

Add a new `.txt` file to `docs/` (first line = festival name, blank line between paragraphs) and restart the app.
