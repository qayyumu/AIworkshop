import gradio as gr

if 0:
   gr.load_chat("http://localhost:11434/v1/", model="mistral", token="***").launch()
else:
    import json
    import re
    from collections import Counter
    from pathlib import Path
    from urllib import request

    import gradio as gr

    OLLAMA_CHAT_URL = "http://localhost:11434/v1/chat/completions"
    MODEL_NAME = "mistral"
    RAG_FILE = Path(__file__).with_name("rag_document.txt")


    def chunk_text(text: str, words_per_chunk: int = 120) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        for i in range(0, len(words), words_per_chunk):
            chunk = " ".join(words[i : i + words_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks


    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())


    def retrieve_context(query: str, chunks: list[str], top_k: int = 2) -> str:
        query_tokens = tokenize(query)
        if not query_tokens or not chunks:
            return ""

        query_counts = Counter(query_tokens)
        scored_chunks: list[tuple[int, str]] = []

        for chunk in chunks:
            chunk_counts = Counter(tokenize(chunk))
            score = sum(query_counts[token] * chunk_counts[token] for token in query_counts)
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        selected = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
        return "\n\n".join(selected)


    def ask_mistral(messages: list[dict]) -> str:
        payload = {"model": MODEL_NAME, "messages": messages, "temperature": 0.2}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            OLLAMA_CHAT_URL,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["choices"][0]["message"]["content"]


    RAG_TEXT = RAG_FILE.read_text(encoding="utf-8") if RAG_FILE.exists() else ""
    RAG_CHUNKS = chunk_text(RAG_TEXT)


    def chat_fn(message: str, history: list[dict]) -> str:
        context = retrieve_context(message, RAG_CHUNKS, top_k=2)

        system_prompt = (
            "You are a helpful assistant. Use the retrieved context when relevant. "
            "If the context is not enough, say what is missing."
        )
        if context:
            system_prompt += f"\n\nRetrieved context:\n{context}"

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": message})

        try:
            return ask_mistral(messages)
        except Exception as error:
            return f"Error talking to Ollama/Mistral: {error}"


    gr.ChatInterface(
        fn=chat_fn,
        title="Mistral + Local RAG",
        description="Answers grounded on Session-03/rag_document.txt",
    ).launch()




