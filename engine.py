import json, os
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = Path(__file__).parent / "faqs.json"
THRESHOLD = 0.30                 # tune in Step 6
LLM_MODEL = "gemini-3.8-flash"  # if you get a 404, check the current name in AI Studio


@dataclass
class Result:
    query: str
    status: str                  # "DRAFTED" or "REFER_TO_HUMAN"
    confidence: float
    draft: str
    matches: list = field(default_factory=list)
    reason: str = ""


class FAQBot:
    def __init__(self, path=FAQ_PATH, threshold=THRESHOLD):
        self.faqs = json.loads(Path(path).read_text(encoding="utf-8"))
        self.threshold = threshold
        corpus = [f["question"] + " " + f["answer"] for f in self.faqs]
        self.vec = TfidfVectorizer(stop_words="english",
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(corpus)

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        ranked = sorted(zip(self.faqs, scores), key=lambda x: x[1], reverse=True)
        return [(f, float(s)) for f, s in ranked[:k]]

    def respond(self, query, threshold=None):
        threshold = self.threshold if threshold is None else threshold
        query = query.strip()
        if not query:
            return Result(query, "REFER_TO_HUMAN", 0.0, "", reason="Empty query")

        matches = self.retrieve(query)
        top_faq, top_score = matches[0]

        # Gate 1: retrieval confidence
        if top_score < threshold:
            return Result(
                query, "REFER_TO_HUMAN", top_score,
                "Thanks for reaching out. To make sure you get accurate help, "
                "I'm passing your question to our support team. "
                "They will get back to you shortly.",
                matches,
                reason=f"Best match score {top_score:.2f} < threshold {threshold:.2f}")

        draft = self._llm_draft(query, matches) or self._template_draft(top_faq)

        # Gate 2: Gemini says the FAQs are not enough
        if draft.strip() == "REFER_TO_HUMAN":
            return Result(
                query, "REFER_TO_HUMAN", top_score,
                "Thanks for reaching out. I'm passing this to our support team "
                "so they can help you properly.",
                matches, reason="LLM judged the FAQs insufficient")

        return Result(query, "DRAFTED", top_score, draft, matches)

    @staticmethod
    def _template_draft(faq):
        return (f"Hi there,\n\n{faq['answer']}\n\n"
                "Let us know if there's anything else we can help with!")

    
    @staticmethod
    def _llm_draft(query, matches):
        if not os.getenv("GEMINI_API_KEY"):
            print("[Gemini] API key not found")
            return None

        print("[Gemini] API key detected")

        try:
            from google import genai

            context = "\n\n".join(
                f"Q: {f['question']}\nA: {f['answer']}"
                for f, _ in matches
            )

            prompt = (
                "You are a customer support assistant. Draft a short, friendly "
                "reply using ONLY the FAQ entries below. Do not invent policies, "
                "numbers or links. If the FAQs do not contain enough information "
                "to answer, reply with exactly: REFER_TO_HUMAN\n\n"
                f"FAQ entries:\n{context}\n\n"
                f"Customer question: {query}\n\nDraft reply:"
            )

            client = genai.Client()
            resp = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )

            print("[Gemini] API request successful")

            if not resp.text:
                print("[Gemini] Empty response received")
                return None

            return resp.text.strip()

        except Exception as e:
            print(f"[LLM fallback to template] {e}")
            return None
