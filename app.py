import os
import re
from flask import Flask, render_template, request, jsonify, session
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from datetime import datetime
from dotenv import load_dotenv
import time
import json

# ---- Load .env file for local dev (ignored on Azure if no .env file exists) ----
load_dotenv()

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}


DEFAULT_SERVICE_LINE = "General"
# Load from environment variables
SERVICE_LINE_CONTACTS = json.loads(
    os.environ.get("SERVICE_LINE_CONTACTS", "{}")
)

SERVICE_LINE_KEYWORDS = json.loads(
    os.environ.get("SERVICE_LINE_KEYWORDS", "{}")
)

def classify_service_line(user_question: str) -> str:
    q = (user_question or "").lower()

    # simple scoring: count keyword matches
    scores = {k: 0 for k in SERVICE_LINE_KEYWORDS.keys()}
    for service_line, words in SERVICE_LINE_KEYWORDS.items():
        for w in words:
            # word boundary for short tokens; contains for multi-word phrases
            if " " in w or any(ch.isdigit() for ch in w):
                if w in q:
                    scores[service_line] += 1
            else:
                if re.search(rf"\b{re.escape(w)}\b", q):
                    scores[service_line] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else DEFAULT_SERVICE_LINE


def append_contact_note(reply_text: str, service_line: str, contact_map: dict) -> str:
    contact = contact_map.get(service_line) or contact_map.get("General", "Sarah")
    note = f"\n\n**Note:** For more information, contact **{contact}**."
    return (reply_text or "").rstrip() + note

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    feedback_type = data.get("feedback")
    response_text = data.get("response")

    print("FEEDBACK RECEIVED")
    print("Type:", feedback_type)
    print("Response snippet:", response_text[:200])

    return {"status": "ok"}

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# ---- Azure AI Foundry setup (loaded from .env locally / App Settings on Azure) ----
PROJECT_ENDPOINT = os.getenv("PROJECT_ENDPOINT")
AGENT_ID = os.getenv("AGENT_ID")

if not PROJECT_ENDPOINT:
    raise ValueError("Missing required config: PROJECT_ENDPOINT is not set.")
if not AGENT_ID:
    raise ValueError("Missing required config: AGENT_ID is not set.")

project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=PROJECT_ENDPOINT
)

agent = project.agents.get_agent(AGENT_ID)


def get_or_create_thread_id() -> str:
    """Create one thread per user (browser session)."""
    if "thread_id" not in session:
        thread = project.agents.threads.create()
        session["thread_id"] = thread.id
    return session["thread_id"]


def clean_agent_response(text: str) -> str:
    """
    Cleans common markdown and AI Foundry citation markers so output is business-friendly:
    - Removes citation markers
    - Removes markdown headings like ###
    - Removes **bold** and *italic*
    - Normalizes excessive blank lines
    """
    if not text:
        return ""

    # Remove citation markers like  or multiple of them
    time.sleep(3)
    # text = re.sub(r"【\d+:\d+†source】", "", text)    
    # text = re.sub(r"【\d+:\d+†source】", "", text)
    # text = re.sub(r"【\d+:\d+†.*?】", "", text)    
    text = re.sub(r"【[^】]*】", "", text)

    # # Remove markdown headings (###, ##, #) at line starts
    # text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # # Remove bold/italic markdown tokens
    # text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # text = re.sub(r"\*(.*?)\*", r"\1", text)

    # # Remove leftover double spaces from citation removal
    # text = re.sub(r"[ \t]{2,}", " ", text)

    # # Normalize blank lines
    # text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("thread_id", None)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    thread_id = get_or_create_thread_id()

    # Add user message
    project.agents.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message + " Please do not hallucinate and give accurate answers."
    )

    # Run agent
    run = project.agents.runs.create_and_process(
        thread_id=thread_id,
        agent_id=agent.id
    )

    if run.status == "failed":
        return jsonify({"error": str(run.last_error)}), 500

    # Get latest assistant reply
    messages = list(project.agents.messages.list(
        thread_id=thread_id,
        order=ListSortOrder.ASCENDING
    ))

    raw_reply = ""
    for msg in reversed(messages):
        if msg.role == "assistant" and msg.text_messages:
            raw_reply = msg.text_messages[-1].text.value
            break

    assistant_reply = clean_agent_response(raw_reply)
    # 2) Decide service line
    service_line = classify_service_line(user_message) 
    # 4) Append note ALWAYS
    assistant_reply = append_contact_note(assistant_reply, service_line,SERVICE_LINE_CONTACTS)    

    return jsonify({"reply": assistant_reply}),200


if __name__ == "__main__":
    app.run()