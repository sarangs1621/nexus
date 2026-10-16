import re
from typing import Protocol

SYSTEM_ANSWER = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite every claim with the passage number in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, say exactly: "
    "'The provided documents do not contain this information.'"
)
SYSTEM_REWRITE = (
    "Rewrite the user's question as a concise search query for a document index: "
    "keep key nouns and technical terms, drop filler. Return only the query."
)


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        msg = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class Generator:
    """Query rewriting + grounded answering. Works without an LLM (extractive fallback)."""

    def __init__(self, llm: LLM | None):
        self.llm = llm

    @property
    def enabled(self) -> bool:
        return self.llm is not None

    def rewrite(self, query: str) -> str | None:
        if not self.llm:
            return None
        out = self.llm.complete(SYSTEM_REWRITE, query, max_tokens=60).strip().strip('"')
        return out if out and out.lower() != query.lower() else None

    def answer(self, query: str, chunks: list[dict]) -> tuple[str, list[int]]:
        if not chunks:
            return "The provided documents do not contain this information.", []
        if not self.llm:
            # extractive fallback: top passage, cited as [1]
            return chunks[0]["text"][:600] + " [1]", [1]
        context = "\n\n".join(f"[{i + 1}] (source: {c['doc_id']})\n{c['text']}" for i, c in enumerate(chunks))
        user = f"Context passages:\n\n{context}\n\nQuestion: {query}"
        text = self.llm.complete(SYSTEM_ANSWER, user, max_tokens=600)
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(chunks)})
        return text, cited
