import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = Path(__file__).parent / "faqs.json"
THRESHOLD = 0.30
LLM_MODEL = "gemini-3.8-flash"


@dataclass
class Result:
    query: str
    status: str
    confidence: float
    draft: str
    matches: list = field(default_factory=list)
    reason: str = ""
    source: str = ""


class FAQBot:
    def __init__(self, path=FAQ_PATH, threshold=THRESHOLD):
        self.faqs = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        self.threshold = threshold

        corpus = [
            f["question"] + " " + f["answer"]
            for f in self.faqs
        ]

        self.vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.matrix = self.vec.fit_transform(corpus)

    def retrieve(self, query, k=3):
        scores = cosine_similarity(
            self.vec.transform([query]),
            self.matrix
        )[0]

        ranked = sorted(
            zip(self.faqs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            (faq, float(score))
            for faq, score in ranked[:k]
        ]

    def respond(self, query, threshold=None):
        threshold = (
            self.threshold
            if threshold is None
            else threshold
        )

        query = query.strip()

        if not query:
            return Result(
                query=query,
                status="REFER_TO_HUMAN",
                confidence=0.0,
                draft="",
                reason="Empty query",
                source="Human referral"
            )

        matches = self.retrieve(query)
        top_faq, top_score = matches[0]

        # Gate 1: FAQ retrieval confidence
        if top_score < threshold:
            return Result(
                query=query,
                status="REFER_TO_HUMAN",
                confidence=top_score,
                draft=(
                    "Thanks for reaching out. "
                    "To make sure you get accurate help, "
                    "I'm passing your question to our "
                    "support team. They will get back "
                    "to you shortly."
                ),
                matches=matches,
                reason=(
                    f"Best match score {top_score:.2f} "
                    f"< threshold {threshold:.2f}"
                ),
                source="Human referral"
            )

        # Gate 2: Try Gemini
        print("[System] Calling Gemini...", flush=True)

        llm_draft = self._llm_draft(query, matches)

        if llm_draft:
            print(
                "[System] Gemini returned a response",
                flush=True
            )

            draft = llm_draft
            source = "Gemini AI"

        else:
            print(
                "[System] Using FAQ template fallback",
                flush=True
            )

            draft = self._template_draft(top_faq)
            source = "FAQ Template"

        # Gate 3: Gemini requests human referral
        if draft.strip() == "REFER_TO_HUMAN":
            return Result(
                query=query,
                status="REFER_TO_HUMAN",
                confidence=top_score,
                draft=(
                    "Thanks for reaching out. "
                    "I'm passing this to our support "
                    "team so they can help you properly."
                ),
                matches=matches,
                reason="LLM judged the FAQs insufficient",
                source="Gemini AI"
            )

        return Result(
            query=query,
            status="DRAFTED",
            confidence=top_score,
            draft=draft,
            matches=matches,
            source=source
        )

    @staticmethod
    def _template_draft(faq):
        return (
            f"Hi there,\n\n"
            f"{faq['answer']}\n\n"
            "Let us know if there's anything "
            "else we can help with!"
        )

    @staticmethod
    def _llm_draft(query, matches):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print(
                "[Gemini] API key not found",
                flush=True
            )
            return None

        print(
            "[Gemini] API key detected",
            flush=True
        )

        try:
            from google import genai

            context = "\n\n".join(
                f"Q: {faq['question']}\nA: {faq['answer']}"
                for faq, _ in matches
            )

            prompt = (
                "You are a customer support assistant. "
                "Draft a short, friendly reply using "
                "ONLY the FAQ entries below. "
                "Do not invent policies, numbers or links. "
                "If the FAQs do not contain enough "
                "information to answer, reply with "
                "exactly: REFER_TO_HUMAN\n\n"
                f"FAQ entries:\n{context}\n\n"
                f"Customer question: {query}\n\n"
                "Draft reply:"
            )

           from google.genai import types

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000)
)

response = client.models.generate_content(
    model=LLM_MODEL,
    contents=prompt
)

            if not response.text:
                print(
                    "[Gemini] Empty response received",
                    flush=True
                )
                return None

            print(
                "[Gemini] API request successful",
                flush=True
            )

            return response.text.strip()

        except Exception as e:
            print(
                f"[Gemini] API error: {e}",
                flush=True
            )
            return None
