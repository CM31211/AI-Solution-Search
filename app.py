import os
import re
from flask import Flask, render_template, request, jsonify, session
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from azure.storage.blob import BlobServiceClient
from datetime import datetime
from dotenv import load_dotenv
import time
import json
import threading

# ---- Load .env file for local dev (ignored on Azure if no .env file exists) ----
load_dotenv()

app = Flask(__name__)

@app.get("/health")
def health():
    return {"status": "ok"}


DEFAULT_SERVICE_LINE = "Default"

# Azure Blob Storage config (no network calls at startup)
_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
_BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER")

if _STORAGE_CONNECTION_STRING:
    _blob_service = BlobServiceClient.from_connection_string(_STORAGE_CONNECTION_STRING)
else:
    _blob_service = BlobServiceClient(account_url=_STORAGE_ACCOUNT_URL, credential=DefaultAzureCredential())

def _load_json_from_blob(blob_name: str) -> dict:
    blob_client = _blob_service.get_blob_client(container=_BLOB_CONTAINER, blob=blob_name)
    data = blob_client.download_blob().readall()
    return json.loads(data.decode("utf-8"))

# Load from environment variables
SERVICE_LINE_CONTACTS = json.loads(os.environ.get("SERVICE_LINE_CONTACTS", "{}"))
SERVICE_LINE_KEYWORDS = json.loads(os.environ.get("SERVICE_LINE_KEYWORDS", "{}"))

# Globals populated by background startup thread
PARTNERS_DIRECTORY = None
SKILLS_DATABASE = None
ALL_NAMES = set()

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


_SCENARIO_RESPONSE_MARKERS = re.compile(
    r'how eisneramper can help|recommended smes|suggested next steps|'
    r'relevant past experience|summary of client problem',
    re.IGNORECASE
)

def append_contact_note(reply_text: str, service_line: str, contact_map: dict) -> str:
    # Only append contact note for structured scenario responses, not greetings/conversational replies
    if not _SCENARIO_RESPONSE_MARKERS.search(reply_text or ""):
        return (reply_text or "").rstrip()
    contact = contact_map.get(service_line) or contact_map.get("Default", "Kristen Lewis")
    note = f"\n\nFor further discussion or service-specific inquiries, please contact **{contact}**."
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

# Globals for agent (populated by background startup thread)
agent = None
_startup_ready = threading.Event()
_startup_error = None

def _background_startup():
    global PARTNERS_DIRECTORY, SKILLS_DATABASE, ALL_NAMES, agent, _startup_error
    try:
        PARTNERS_DIRECTORY = _load_json_from_blob(os.environ.get("PARTNERS_DIRECTORY_BLOB"))
        SKILLS_DATABASE    = _load_json_from_blob(os.environ.get("SKILLS_DATABASE_BLOB"))
        ALL_NAMES.update(
            [p["Name"] for p in PARTNERS_DIRECTORY["partners"] if "Name" in p] +
            [s["Legal_Full_Name"] for s in SKILLS_DATABASE["Report_Entry"] if "Legal_Full_Name" in s]
        )
        agent = project.agents.get_agent(AGENT_ID)
        print("Background startup complete.")
    except Exception as e:
        _startup_error = e
        print(f"Background startup failed: {e}")
    finally:
        _startup_ready.set()

threading.Thread(target=_background_startup, daemon=True).start()


def get_or_create_thread_id() -> str:
    """Create one thread per user (browser session)."""
    if "thread_id" not in session:
        thread = project.agents.threads.create()
        session["thread_id"] = thread.id
    return session["thread_id"]


def validate_sme_names(text: str) -> str:
    """
    For each Sales / Technical SME section independently:
      - Splits the section body into segments (blank-line separated).
      - Each segment is classified as valid or invalid by:
          * Name: blocks  → looks up the name in ALL_NAMES
          * Plain paragraphs → scans text for any known name from ALL_NAMES
          * Agent's own "No specific SME..." placeholder → always invalid
      - If at least one segment is valid: keeps only valid segments.
      - If none are valid: replaces entire section with a single not-found message.
    """
    _all_names_lower = {n.lower(): n for n in ALL_NAMES}
    _not_found_msg = "No verified SME found in our directory for this category."

    def classify_segment(seg: str):
        """Return True if this segment describes a valid SME, False otherwise."""
        stripped = seg.strip()
        if not stripped:
            return None  # blank — skip

        # Check for Name: prefix
        name_match = re.match(r'-?\s*Name:\s*([^\n]+)', stripped, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            if name.lower().startswith("no specific sme"):
                return False  # agent's own placeholder
            return name.lower() in _all_names_lower

        # Plain paragraph — scan for any known name
        seg_lower = stripped.lower()
        return any(n in seg_lower for n in _all_names_lower)

    # Regex to detect the start of a main response section — these must never be removed
    _main_section_re = re.compile(
        r'summary of client problem|how eisneramper can help|'
        r'relevant past experience|case studies|'
        r'recommended smes or teams|suggested next steps',
        re.IGNORECASE
    )

    def process_sme_body(body: str) -> str:
        """Filter all SME entries in one section body, preserving any trailing main sections."""
        segments = re.split(r'\n{2,}', body)

        sme_segs = []   # segments belonging to this SME section
        tail_segs = []  # everything from the next main section header onwards
        in_tail = False

        for seg in segments:
            stripped = seg.strip()
            if not stripped:
                continue
            # Once we hit a main section header, stop SME processing
            if not in_tail and _main_section_re.search(stripped):
                in_tail = True
            if in_tail:
                tail_segs.append(seg)
            else:
                result = classify_segment(seg)
                if result is not None:
                    sme_segs.append((seg, result))

        # Build the filtered SME part
        if not sme_segs:
            sme_part = ''
        else:
            valid_segs = [seg for seg, is_valid in sme_segs if is_valid]
            if not valid_segs:
                sme_part = f"\nName: {_not_found_msg}\n"
            else:
                sme_part = '\n\n'.join(valid_segs) + '\n'

        # Re-attach any main sections that were cut off
        tail_part = ('\n\n' + '\n\n'.join(tail_segs)) if tail_segs else ''

        return sme_part + tail_part

    # Split text on Sales/Technical section headers, keeping headers as delimiters.
    # Handles: "Sales:", "Technical:", "Sales SMEs:", "**Technical SMEs:**", "### Sales:" etc.
    header_re = re.compile(
        r'(\*{0,3}#{0,3}\s*(?:Sales|Technical)(?:\s+SME[s]?)?\s*(?:\*{0,3})?\s*:\s*(?:\*{0,3})?\s*\n)',
        re.IGNORECASE
    )
    parts = header_re.split(text)

    # parts layout: [pre_text, header1, body1, header2, body2, ...]
    result = [parts[0]]
    for i in range(1, len(parts), 2):
        header = parts[i]
        body   = parts[i + 1] if i + 1 < len(parts) else ''
        result.append(header)
        result.append(process_sme_body(body))

    return ''.join(result)


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
    _startup_ready.wait()  # block until background startup finishes
    if _startup_error:
        print(f"Startup error prevented chat: {_startup_error}")
        return jsonify({"error": "Service is not ready. Please try again shortly."}), 503

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
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
                print(f"Agent run failed: {run.last_error}")
                return jsonify({"error": "Agent run failed"}), 500

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
            assistant_reply = validate_sme_names(assistant_reply)
            service_line = classify_service_line(user_message)
            assistant_reply = append_contact_note(assistant_reply, service_line, SERVICE_LINE_CONTACTS)

            return jsonify({"reply": assistant_reply}), 200

        except ConnectionResetError as e:
            print(f"Connection reset on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 2s, 4s backoff
            else:
                return jsonify({"error": "Something went wrong. Please try again."}), 500

        except Exception as e:
            print(f"Chat error: {e}")
            return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run()