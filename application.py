import requests
import streamlit as st

st.set_page_config(page_title="Handbook Assistant", page_icon="📘", layout="centered")
st.title("Handbook Assistant")
st.caption("Ask policy-related or general questions about the company handbook.")


@st.cache_data
def get_handbook_chunks():
    return [
        "Leave Policy: Employees are entitled to 18 paid leave days per calendar year, including casual and sick leave combined.",
        "Laptop Policy: Company laptops are provided to all full-time employees and must be returned upon exit. Personal use is permitted within reasonable limits.",
        "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval, submitted via the HR portal.",
        "Expense Policy: Business expenses including client meals and travel are reimbursable with receipts submitted within 30 days.",
        "Probation Policy: New employees undergo a 6-month probation period, reviewed at the 3-month and 6-month marks.",
        "Notice Period: Employees must serve a notice period of 60 days upon resignation, unless otherwise agreed with HR.",
        "Health Insurance: All employees are covered under group health insurance from day one, extending to immediate family.",
        "Working Hours: Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday, with flexible start times within a 1-hour window.",
        "Grievance Redressal: Employees can raise workplace grievances confidentially through the HR helpline, acknowledged within 2 working days.",
        "Exit Process: Employees must complete a knowledge transfer plan before their last working day; full settlement is processed within 45 days.",
    ]


api_key = "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"


def call_groq(prompt: str, max_tokens: int = 300) -> str:
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

CLASSIFY_PROMPT = """Classify the user's question as exactly one word:
either \"policy\" or \"general\".
policy = questions about company rules, leave, expenses, equipment,
benefits, conduct, or HR processes.
general = anything else (small talk, general knowledge, unrelated topics).
Examples:
Question: How many paid leave days do I get per year?
Answer: policy
Question: What's a good recipe for banana bread?
Answer: general
Question: Can I expense a client dinner?
Answer: policy
Now classify this question. Answer with exactly one word, nothing else.
Question: {question}
Answer:"""


def classify_question(question: str) -> str:
    prompt = CLASSIFY_PROMPT.format(question=question)
    response = call_groq(prompt).strip().lower()
    return "policy" if "policy" in response else "general"


def retrieve(query: str, k: int = 2):
    import re

    chunks = get_handbook_chunks()
    stop_words = {
        "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because",
        "before", "between", "can", "company", "context", "does", "doesn't", "do", "for", "from",
        "how", "i", "in", "is", "it", "leave", "mention", "my", "no", "not", "of", "on", "or",
        "policy", "question", "the", "this", "to", "what", "when", "where", "which", "with",
        "would", "you"
    }

    def normalize_terms(text: str):
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return [term for term in cleaned.split() if len(term) > 2 and term not in stop_words]

    query_terms = set(normalize_terms(query))
    if not query_terms:
        return []

    scored_chunks = []
    for chunk in chunks:
        chunk_terms = set(normalize_terms(chunk))
        overlap = len(query_terms & chunk_terms)
        if overlap > 0:
            scored_chunks.append((overlap, chunk))

    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]


def run_agent(question: str):
    category = classify_question(question)

    if category == "policy":
        chunks = retrieve(question, k=2)
        if not chunks:
            return category, "", "I couldn't find a relevant handbook policy for that question."

        context = "\n".join(chunks)
        prompt = f"""Answer the question using ONLY the context below. Be concise.
Context:
{context}
Question: {question}
Answer:"""
        answer = call_groq(prompt)
        return category, context, answer

    answer = call_groq(question)
    return category, "(no retrieval -- general question)", answer


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            category, context, answer = run_agent(prompt)

        st.markdown(answer)
        st.caption(f"Category: {category}")
        if context and context != "(no retrieval -- general question)":
            with st.expander("Retrieved context"):
                st.write(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
