import os
import re
from flask import Flask, render_template, request, jsonify, session
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time
import json
import threading
import jwt
from jwt import PyJWKClient

# ---- Load .env file for local dev (ignored on Azure if no .env file exists) ----
load_dotenv()

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/chat", methods=["OPTIONS"])
@app.route("/history", methods=["OPTIONS"])
@app.route("/feedback", methods=["OPTIONS"])
@app.route("/reset", methods=["OPTIONS"])
@app.route("/me", methods=["OPTIONS"])
def handle_options():
    return "", 204

@app.get("/health")
def health():
    return {"status": "ok"}


DEFAULT_SERVICE_LINE = "Default"
# [LOGGING] Fallback username when Azure Easy Auth headers are absent (e.g. local dev)
_UNAUTHENTICATED_USER = "Unknown"

# Azure Blob Storage config (no network calls at startup)
_STORAGE_ACCOUNT_URL = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
_BLOB_CONTAINER = os.environ.get("BLOB_CONTAINER")

# ---- [STARTUP] Log key config at import time so Log Stream shows it immediately ----
print(f"[STARTUP] AZURE_STORAGE_ACCOUNT_URL={'set' if _STORAGE_ACCOUNT_URL else 'NOT SET'}")
print(f"[STARTUP] AZURE_STORAGE_CONNECTION_STRING={'set' if _STORAGE_CONNECTION_STRING else 'NOT SET'}")
print(f"[STARTUP] BLOB_CONTAINER={_BLOB_CONTAINER or 'NOT SET'}")
print(f"[STARTUP] LOGS_BLOB_CONTAINER={os.environ.get('LOGS_BLOB_CONTAINER') or 'NOT SET'}")
print(f"[STARTUP] AAD_TENANT_ID={'set' if os.environ.get('AAD_TENANT_ID') else 'NOT SET'}")
print(f"[STARTUP] AAD_CLIENT_ID={os.environ.get('AAD_CLIENT_ID', 'NOT SET')}")
print(f"[STARTUP] WEBAPP_FQDN={os.environ.get('WEBAPP_FQDN', 'NOT SET')}")
print(f"[STARTUP] PROJECT_ENDPOINT={'set' if os.environ.get('PROJECT_ENDPOINT') else 'NOT SET'}")
print(f"[STARTUP] AGENT_ID={os.environ.get('AGENT_ID', 'NOT SET')}")

if _STORAGE_CONNECTION_STRING:
    _blob_service = BlobServiceClient.from_connection_string(_STORAGE_CONNECTION_STRING)
    print("[STARTUP] BlobServiceClient created via connection string")
else:
    _blob_service = BlobServiceClient(account_url=_STORAGE_ACCOUNT_URL, credential=DefaultAzureCredential())
    print("[STARTUP] BlobServiceClient created via DefaultAzureCredential")

def _load_json_from_blob(blob_name: str) -> dict:
    blob_client = _blob_service.get_blob_client(container=_BLOB_CONTAINER, blob=blob_name)
    data = blob_client.download_blob().readall()
    return json.loads(data.decode("utf-8"))

# [LOGGING] Separate container for conversation logs (read from environment variable)
_LOGS_CONTAINER = os.environ.get("LOGS_BLOB_CONTAINER")


def _save_conversation_log(thread_id: str, user: str, query: str, response: str, message_timestamp: str) -> None:
    """
    [LOGGING] Appends one query/response exchange to the conversation JSON file for this thread.
    Blob path: logs/<MMDDYYYY>/<username>/<thread_id>.json
    - feedback field is absent initially; added later when user clicks like/dislike.
    - message_timestamp is passed in from the chat route so client and log always use the same value.
    Creates the file if it doesn't exist; appends to it if it does.
    Runs silently — logs errors but never raises, so chat flow is never interrupted.
    """
    try:
        # [LOGGING] Parse date folder from the passed-in timestamp (same value returned to client)
        msg_dt      = datetime.fromisoformat(message_timestamp.replace("Z", "+00:00"))
        date_folder = msg_dt.strftime("%m%d%Y")                    # e.g. 03122026
        # [LOGGING] Strip domain from email to get clean username for folder name
        # When running locally, Easy Auth is not present so user defaults to "Unknown"
        username    = user.split("@")[0] if "@" in user else user  # e.g. chandan.mishra or Unknown
        blob_name   = f"{date_folder}/{username}/{thread_id}.json"

        blob_client = _blob_service.get_blob_client(container=_LOGS_CONTAINER, blob=blob_name)

        # Load existing conversation if the blob already exists
        try:
            existing = blob_client.download_blob().readall()
            conversation = json.loads(existing.decode("utf-8"))
            print(f"[LOGGING] Appending to existing conversation log: {blob_name}")
        except Exception:
            # Blob doesn't exist yet — start a new conversation record
            conversation = {
                "user":      username,
                "thread_id": thread_id,
                "date":      date_folder,
                "messages":  []
            }
            print(f"[LOGGING] Creating new conversation log: {blob_name}")

        # Append this exchange — no feedback field yet, added only if user clicks like/dislike
        conversation["messages"].append({
            "timestamp": message_timestamp,
            "query":     query,
            "response":  response
        })

        blob_client.upload_blob(
            json.dumps(conversation, indent=2),
            overwrite=True,
            content_settings=None
        )
        print(f"[LOGGING] Conversation log saved: {blob_name} (total messages: {len(conversation['messages'])})")
    except Exception as e:
        print(f"[LOGGING] Failed to save conversation log: {e}")


def _save_error_log(error_msg: str, user: str, context: str = "") -> None:
    """
    [LOGGING] Saves error details to a dedicated errorlog subfolder in blob storage.
    Blob path: logs/<MMDDYYYY>/errorlog/errorlog_{ddMMyyyyHHmmssffff}.json
    - ffff = first 4 digits of microseconds (sub-second uniqueness)
    - Runs silently — never raises, never interrupts any flow.
    """
    try:
        if not _blob_service or not _LOGS_CONTAINER:
            print(f"[ERROR_LOG] Skipped (no blob service or logs container) — context={context} error={error_msg}")
            return
        now = datetime.now(timezone.utc)
        date_folder = now.strftime("%m%d%Y")
        ffff = str(now.microsecond).zfill(6)[:4]
        filename = f"errorlog_{now.strftime('%d%m%Y%H%M%S')}{ffff}.json"
        blob_name = f"{date_folder}/errorlog/{filename}"
        username = user.split("@")[0] if "@" in user else user
        error_data = {
            "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S.") + ffff + "Z",
            "user": username,
            "context": context,
            "error": str(error_msg)
        }
        blob_client = _blob_service.get_blob_client(container=_LOGS_CONTAINER, blob=blob_name)
        blob_client.upload_blob(json.dumps(error_data, indent=2), overwrite=True)
        print(f"[ERROR_LOG] Saved: {blob_name} context={context}")
    except Exception as e:
        print(f"[ERROR_LOG] Failed to save error log: {e}")  # Never let error logging interrupt anything


# Load from environment variables
SERVICE_LINE_CONTACTS = json.loads(os.environ.get("SERVICE_LINE_CONTACTS", "{}"))
SERVICE_LINE_KEYWORDS = json.loads(os.environ.get("SERVICE_LINE_KEYWORDS", "{}"))
print(f"[STARTUP] SERVICE_LINE_CONTACTS loaded: {list(SERVICE_LINE_CONTACTS.keys())}")
print(f"[STARTUP] SERVICE_LINE_KEYWORDS loaded: {list(SERVICE_LINE_KEYWORDS.keys())}")

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
    client = _detect_client()
    user, err = _require_user()
    if err:
        return err
    data = request.json
    feedback_type = data.get("feedback", "unknown")
    thread_id = data.get("thread_id", "unknown")
    print(f"[FEEDBACK] [{client}] user={user} type={feedback_type} thread_id={thread_id}")

    # [LOGGING] Fire-and-forget background thread — never blocks response
    threading.Thread(target=_embed_feedback_in_conversation, args=(user, data), daemon=True).start()

    return {"status": "ok"}


def _embed_feedback_in_conversation(user: str, data: dict) -> None:
    """
    [LOGGING] Finds the matching message in the conversation log by message_timestamp
    and embeds feedback + feedback_timestamp directly into it.
    Blob path: logs/<MMDDYYYY>/<username>/<thread_id>.json
    - message_timestamp identifies exactly which response was liked/disliked.
    - All responses without feedback have no feedback field — easy to differentiate on display.
    Runs silently — logs errors but never raises.
    """
    try:
        username          = user.split("@")[0] if "@" in user else user
        thread_id         = data.get("thread_id", "unknown")
        message_timestamp = data.get("message_timestamp", "")
        feedback_type     = data.get("feedback")           # "like" or "dislike"
        feedback_timestamp = data.get("feedback_timestamp")

        # [LOGGING] Derive conversation date from message_timestamp to locate the correct blob
        msg_dt      = datetime.fromisoformat(message_timestamp.replace("Z", "+00:00"))
        date_folder = msg_dt.strftime("%m%d%Y")
        blob_name   = f"{date_folder}/{username}/{thread_id}.json"

        blob_client  = _blob_service.get_blob_client(container=_LOGS_CONTAINER, blob=blob_name)
        existing     = blob_client.download_blob().readall()
        conversation = json.loads(existing.decode("utf-8"))

        # [LOGGING] Find the matching message and embed feedback into it.
        # Normalize to seconds precision before comparing — backend stores without milliseconds
        # (e.g. "2026-03-12T16:35:12Z") but frontend sends with milliseconds ("2026-03-12T16:35:12.727Z")
        def _to_seconds(ts: str) -> str:
            return ts[:19] if ts else ""

        updated = False
        for message in conversation["messages"]:
            if _to_seconds(message.get("timestamp", "")) == _to_seconds(message_timestamp):
                message["feedback"]           = feedback_type
                message["feedback_timestamp"] = feedback_timestamp
                updated = True
                break

        if updated:
            blob_client.upload_blob(json.dumps(conversation, indent=2), overwrite=True)
            print(f"[LOGGING] Feedback '{feedback_type}' embedded in {blob_name}")
        else:
            print(f"[LOGGING] No matching message found for timestamp {message_timestamp} in {blob_name}")

    except Exception as e:
        print(f"[LOGGING] Failed to embed feedback: {e}")
        _save_error_log(e, user, "feedback")

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# ---- Azure AD / Teams SSO config ----
AAD_TENANT_ID = os.environ.get("AAD_TENANT_ID", "")
AAD_CLIENT_ID = os.environ.get("AAD_CLIENT_ID", "")
WEBAPP_FQDN   = os.environ.get("WEBAPP_FQDN", "")

_jwks_client = None
_jwks_client_lock = threading.Lock()

def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        with _jwks_client_lock:
            if _jwks_client is None:
                jwks_url = f"https://login.microsoftonline.com/{AAD_TENANT_ID}/discovery/v2.0/keys"
                print(f"[AUTH] Initialising JWKS client: {jwks_url}")
                _jwks_client = PyJWKClient(jwks_url, cache_keys=True)
    return _jwks_client

def _validate_teams_token(token: str) -> bool:
    """Validate an AAD Bearer token issued by Teams SSO. Returns True if valid.
    Checks: signature, algorithm, audience, issuer, expiry, and access_as_user scope.
    """
    if not AAD_TENANT_ID or not AAD_CLIENT_ID:
        print("[AUTH] _validate_teams_token: AAD_TENANT_ID or AAD_CLIENT_ID not configured")
        return False
    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        expected_audiences = [
            AAD_CLIENT_ID,
            f"api://{WEBAPP_FQDN}/{AAD_CLIENT_ID}"
        ]
        decoded = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=expected_audiences,
            issuer=[
                f"https://sts.windows.net/{AAD_TENANT_ID}/",
                f"https://login.microsoftonline.com/{AAD_TENANT_ID}/v2.0"
            ],
            options={"require": ["exp", "aud", "iss"]}
        )
        # Log key claims for verification (no sensitive data — upn is email)
        scp  = decoded.get("scp", "")
        upn  = decoded.get("upn") or decoded.get("preferred_username", "unknown")
        aud  = decoded.get("aud", "")
        iss  = decoded.get("iss", "")
        exp  = decoded.get("exp", 0)
        exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if exp else "none"
        print(f"[AUTH] Token decoded — upn={upn} aud={aud} iss={iss} scp='{scp}' exp={exp_dt}")

        # Enforce access_as_user scope — scp is a space-separated string
        if "access_as_user" not in scp.split():
            msg = f"Teams token rejected: 'access_as_user' scope missing (scp='{scp}')"
            print(f"[AUTH] {msg}")
            threading.Thread(target=_save_error_log, args=(msg, _UNAUTHENTICATED_USER, "teams_token_scope"), daemon=True).start()
            return False
        print(f"[AUTH] Token valid for user={upn}")
        return True
    except Exception as e:
        print(f"[AUTH] Teams token validation failed: {e}")
        threading.Thread(target=_save_error_log, args=(e, _UNAUTHENTICATED_USER, "teams_token_validation"), daemon=True).start()
        return False


def _detect_client() -> str:
    """
    Detects whether the request is coming from Browser, Teams Web, or Teams Desktop.

    Detection logic:
      - No Bearer token + Easy Auth headers present  → "Browser"
      - Bearer token + "Electron" in User-Agent      → "Teams Desktop"
        (Teams desktop app is built on Electron; its UA always contains "Electron")
      - Bearer token + no "Electron" in User-Agent   → "Teams Web"
      - No token, no Easy Auth (local dev)           → "Local Dev"
    """
    auth_header    = request.headers.get("Authorization", "")
    easy_auth_user = request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME") or \
                     request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
    user_agent     = request.headers.get("User-Agent", "")

    if not auth_header and easy_auth_user:
        return "Browser"
    if auth_header:
        return "Teams Desktop" if "Electron" in user_agent else "Teams Web"
    return "Local Dev"


def _require_user() -> tuple:
    """
    Returns (user, error_response) for all API routes.
    Accepts either:
      - Easy Auth headers (browser path) — X-MS-CLIENT-PRINCIPAL-NAME / ID
      - Valid Teams Bearer token (Teams path)
    Rejects requests that have neither — unauthenticated browser access blocked at API level.
    """
    auth_header = request.headers.get("Authorization", "")
    easy_auth_user = (
        request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME") or
        request.headers.get("X-MS-CLIENT-PRINCIPAL-ID")
    )
    client = _detect_client()
    user_agent = request.headers.get("User-Agent", "")
    print(f"[AUTH] {request.method} {request.path} — "
          f"client='{client}' "
          f"auth_header={'present' if auth_header else 'absent'} "
          f"easy_auth_user={easy_auth_user or 'absent'} "
          f"user_agent={user_agent[:80]!r}")

    if auth_header:
        # Teams path — validate Bearer token
        token = auth_header.removeprefix("Bearer ").strip()
        print(f"[AUTH] [{client}] Teams token received (length={len(token)}) — validating...")
        if not _validate_teams_token(token):
            print(f"[AUTH] [{client}] Token validation FAILED for {request.path} — returning 401")
            return None, (jsonify({"error": "Unauthorized"}), 401)
        # Token is valid — user identity comes from Easy Auth header if present,
        # otherwise fall back to Unknown (Teams SSO doesn't inject Easy Auth headers)
        resolved_user = easy_auth_user or _UNAUTHENTICATED_USER
        print(f"[AUTH] [{client}] Token valid — resolved user={resolved_user}")
        return resolved_user, None

    if easy_auth_user:
        # Browser path — Easy Auth already authenticated the user
        print(f"[AUTH] [{client}] Easy Auth — user={easy_auth_user}")
        return easy_auth_user, None

    # Neither header nor token — only possible in local dev (Easy Auth always injects headers in production)
    # Allow through as Unknown so local testing works without Azure authentication
    print(f"[AUTH] [{client}] No auth headers — falling back to Unknown")
    return _UNAUTHENTICATED_USER, None

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
print(f"[STARTUP] AIProjectClient created — endpoint={PROJECT_ENDPOINT}")

# Globals for agent (populated by background startup thread)
agent = None
_startup_ready = threading.Event()
_startup_error = None

def _background_startup():
    global PARTNERS_DIRECTORY, SKILLS_DATABASE, ALL_NAMES, agent, _startup_error
    try:
        partners_blob = os.environ.get("PARTNERS_DIRECTORY_BLOB")
        skills_blob   = os.environ.get("SKILLS_DATABASE_BLOB")
        print(f"[STARTUP] Loading blob: {partners_blob}")
        PARTNERS_DIRECTORY = _load_json_from_blob(partners_blob)
        partner_count = len(PARTNERS_DIRECTORY.get("partners", []))
        print(f"[STARTUP] Partners directory loaded — {partner_count} entries")

        print(f"[STARTUP] Loading blob: {skills_blob}")
        SKILLS_DATABASE = _load_json_from_blob(skills_blob)
        skills_count = len(SKILLS_DATABASE.get("Report_Entry", []))
        print(f"[STARTUP] Skills database loaded — {skills_count} entries")

        ALL_NAMES.update(
            [p["Name"] for p in PARTNERS_DIRECTORY["partners"] if "Name" in p] +
            [s["Legal_Full_Name"] for s in SKILLS_DATABASE["Report_Entry"] if "Legal_Full_Name" in s]
        )
        print(f"[STARTUP] ALL_NAMES populated — {len(ALL_NAMES)} unique names")

        print(f"[STARTUP] Fetching agent: {AGENT_ID}")
        agent = project.agents.get_agent(AGENT_ID)
        print(f"[STARTUP] Agent loaded — id={agent.id} name={getattr(agent, 'name', 'unknown')}")
        print("[STARTUP] Background startup complete.")
    except Exception as e:
        _startup_error = e
        print(f"[STARTUP] Background startup FAILED: {e}")
    finally:
        _startup_ready.set()

threading.Thread(target=_background_startup, daemon=True).start()


def get_or_create_thread_id() -> str:
    """Create one thread per user (browser session)."""
    if "thread_id" not in session:
        thread = project.agents.threads.create()
        session["thread_id"] = thread.id
        print(f"[CHAT] New agent thread created: {thread.id}")
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


@app.route("/auth-start")
def auth_start():
    """Teams SSO consent start page — opens Azure AD consent in a popup."""
    tenant_id = AAD_TENANT_ID or "common"
    client_id = AAD_CLIENT_ID
    redirect_uri = request.host_url.rstrip("/") + "/auth-end"
    scope = f"api://{WEBAPP_FQDN}/{client_id}/access_as_user"
    print(f"[AUTH] /auth-start — tenant={tenant_id} client={client_id} redirect={redirect_uri} scope={scope}")
    auth_url = (
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
        f"?client_id={client_id}"
        f"&response_type=token"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope}"
        f"&prompt=consent"
    )
    return f'<script>window.location.href = "{auth_url}";</script>'


@app.route("/auth-end")
def auth_end():
    """Teams SSO auth completion page — called after consent redirect."""
    print("[AUTH] /auth-end called — notifying Teams consent success")
    return """<!DOCTYPE html><html><head>
    <script src="https://res.cdn.office.net/teams-js/2.22.0/js/MicrosoftTeams.min.js"></script>
    <script>
      microsoftTeams.app.initialize().then(() => {
        microsoftTeams.authentication.notifySuccess();
      });
    </script>
    </head><body></body></html>"""


@app.route("/me", methods=["GET"])
def me():
    logged_in_user = (
        request.headers.get("X-MS-CLIENT-PRINCIPAL-NAME") or
        request.headers.get("X-MS-CLIENT-PRINCIPAL-ID") or
        _UNAUTHENTICATED_USER
    )
    local_part = logged_in_user.split("@")[0] if "@" in logged_in_user else logged_in_user
    display_name = " ".join(p.capitalize() for p in local_part.split("."))
    print(f"[ME] user={logged_in_user} display_name={display_name}")
    return jsonify({"display_name": display_name})


@app.route("/history", methods=["GET"])
def history():
    """[HISTORY] Returns all conversation logs for the currently logged-in user."""
    client = _detect_client()
    user, err = _require_user()
    if err:
        return err
    username = user.split("@")[0] if "@" in user else user
    print(f"[HISTORY] [{client}] Loading history for user={username}")
    try:
        container_client = _blob_service.get_container_client(_LOGS_CONTAINER)

        # Build set of valid date folders for the last 30 days
        valid_dates = {
            (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%m%d%Y")
            for i in range(31)
        }

        conversations = []
        blobs_scanned = 0
        blobs_matched = 0
        for blob in container_client.list_blobs():
            blobs_scanned += 1
            # Blob path: MMDDYYYY/username/thread_id.json — only load blobs for this user within last 30 days
            parts = blob.name.split("/")
            if len(parts) == 3 and parts[1] == username and parts[0] in valid_dates:
                blobs_matched += 1
                try:
                    blob_client = _blob_service.get_blob_client(container=_LOGS_CONTAINER, blob=blob.name)
                    data = blob_client.download_blob().readall()
                    conversations.append(json.loads(data.decode("utf-8")))
                except Exception as e:
                    print(f"[HISTORY] Failed to read blob {blob.name}: {e}")
        print(f"[HISTORY] [{client}] Scan complete — blobs_scanned={blobs_scanned} blobs_matched={blobs_matched} conversations_loaded={len(conversations)}")
        # Sort newest date first
        conversations.sort(key=lambda c: c.get("date", ""), reverse=True)
        return jsonify({"conversations": conversations})
    except Exception as e:
        print(f"[HISTORY] [{client}] Failed to list conversations: {e}")
        return jsonify({"conversations": []})


@app.route("/reset", methods=["POST"])
def reset():
    _, err = _require_user()
    if err:
        return err
    old_thread = session.pop("thread_id", None)
    print(f"[RESET] Session thread cleared — old_thread_id={old_thread or 'none'}")
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    client = _detect_client()
    logged_in_user, err = _require_user()
    if err:
        return err
    _startup_ready.wait()  # block until background startup finishes
    if _startup_error:
        print(f"[CHAT] [{client}] Startup error prevented chat: {_startup_error}")
        return jsonify({"error": "Service is not ready. Please try again shortly."}), 503
    print(f"[CHAT] [{client}] Request from user={logged_in_user}")

    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    print(f"[CHAT] [{client}] Message length={len(user_message)} chars")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            thread_id = get_or_create_thread_id()
            print(f"[CHAT] [{client}] Using thread_id={thread_id} (attempt {attempt}/{max_retries})")

            # Add user message
            project.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=user_message
            )
            print(f"[CHAT] [{client}] User message added to thread {thread_id}")

            # Run agent
            print(f"[CHAT] [{client}] Starting agent run on thread {thread_id}")
            run = project.agents.runs.create_and_process(
                thread_id=thread_id,
                agent_id=agent.id
            )
            print(f"[CHAT] [{client}] Agent run completed — status={run.status} run_id={run.id}")

            if run.status == "failed":
                print(f"[CHAT] [{client}] Agent run FAILED: {run.last_error}")
                return jsonify({"error": "Agent run failed"}), 500

            # Get latest assistant reply
            messages = list(project.agents.messages.list(
                thread_id=thread_id,
                order=ListSortOrder.ASCENDING
            ))
            print(f"[CHAT] [{client}] Retrieved {len(messages)} messages from thread")

            raw_reply = ""
            for msg in reversed(messages):
                if msg.role == "assistant" and msg.text_messages:
                    raw_reply = msg.text_messages[-1].text.value
                    break

            print(f"[CHAT] [{client}] Raw reply length={len(raw_reply)} chars")
            assistant_reply = clean_agent_response(raw_reply)
            assistant_reply = validate_sme_names(assistant_reply)
            service_line = classify_service_line(user_message)
            print(f"[CHAT] [{client}] Service line classified as: {service_line}")
            assistant_reply = append_contact_note(assistant_reply, service_line, SERVICE_LINE_CONTACTS)

            # [LOGGING] Compute timestamp here so both the log and the client use the exact same value
            message_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            threading.Thread(target=_save_conversation_log, args=(thread_id, logged_in_user, user_message, assistant_reply, message_timestamp), daemon=True).start()
            print(f"[CHAT] [{client}] Response ready — thread_id={thread_id} timestamp={message_timestamp} reply_length={len(assistant_reply)} chars")

            return jsonify({"reply": assistant_reply, "thread_id": thread_id, "message_timestamp": message_timestamp}), 200

        except ConnectionResetError as e:
            print(f"[CHAT] [{client}] ConnectionResetError on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 2s, 4s backoff
            else:
                threading.Thread(target=_save_error_log, args=(e, logged_in_user, "chat_connection_reset"), daemon=True).start()
                return jsonify({"error": "Something went wrong. Please try again."}), 500

        except Exception as e:
            print(f"[CHAT] [{client}] Error: {e}")
            threading.Thread(target=_save_error_log, args=(e, logged_in_user, "chat"), daemon=True).start()
            return jsonify({"error": "Something went wrong. Please try again."}), 500


if __name__ == "__main__":
    app.run()
