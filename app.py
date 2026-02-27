import os
import re
from flask import Flask, render_template, request, jsonify, session
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from datetime import datetime
from dotenv import load_dotenv

# ---- Load .env file for local dev (ignored on Azure if no .env file exists) ----
load_dotenv()

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}


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
    text = re.sub(r"【\d+:\d+†source】", "", text)

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
        content=user_message
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
    return jsonify({"reply": assistant_reply})


if __name__ == "__main__":
    app.run()