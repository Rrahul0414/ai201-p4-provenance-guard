import argparse
import json
import math
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

try:
    from groq import Groq
except ImportError:  # pragma: no cover
    Groq = None

load_dotenv()

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

DB_PATH = Path(__file__).resolve().parent / "data" / "provenance_guard.db"
DB_PATH.parent.mkdir(exist_ok=True)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

client = None
if Groq is not None and os.getenv("GROQ_API_KEY"):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            content_id TEXT PRIMARY KEY,
            creator_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL,
            attribution TEXT NOT NULL,
            confidence REAL NOT NULL,
            label TEXT NOT NULL,
            llm_score REAL,
            heuristic_score REAL,
            appeal_reasoning TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            creator_id TEXT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            attribution TEXT,
            confidence REAL,
            llm_score REAL,
            heuristic_score REAL,
            status TEXT,
            label TEXT,
            appeal_reasoning TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def semantic_signal(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.5

    if client is not None:
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You assess whether a short creative text reads more like a human-authored piece "
                            "or machine-generated prose. Return JSON with keys 'score' (0.0-1.0, where 1.0 means AI-like) "
                            "and 'reason'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Assess this text:\n\n{normalized}",
                    },
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            payload = response.choices[0].message.content
            parsed = json.loads(payload)
            score = float(parsed.get("score", 0.5))
            return clamp(score)
        except Exception:
            pass

    tokens = re.findall(r"\b[\w']+\b", normalized.lower())
    sentence_split = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [part for part in sentence_split if part.strip()]
    avg_sentence_length = sum(len(re.findall(r"\b[\w']+\b", s)) for s in sentences) / max(1, len(sentences))
    punctuation_density = len(re.findall(r"[.!?,;:]", normalized)) / max(1, len(tokens))
    contractions = len(re.findall(r"\b(?:can't|won't|don't|isn't|aren't|didn't|i'm|you're|we're|they're|it's|that's|there's|let's)\b", normalized.lower()))
    first_person = len(re.findall(r"\b(i|me|my|mine|we|our|us)\b", normalized.lower()))
    casual_markers = len(re.findall(r"\b(lol|kinda|sorta|gonna|wanna|honestly|super|pretty|really|totally|stuff)\b", normalized.lower()))

    score = 0.18
    if avg_sentence_length >= 12:
        score += 0.18
    if punctuation_density > 0.05:
        score += 0.15
    if contractions == 0 and first_person == 0:
        score += 0.12
    if casual_markers > 0:
        score -= 0.12
    if len(sentences) >= 3:
        score += 0.08
    if len(tokens) < 25:
        score -= 0.05
    return clamp(score)


def stylometric_signal(text: str) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.5

    tokens = re.findall(r"\b[\w']+\b", normalized.lower())
    sentence_split = re.split(r"(?<=[.!?])\s+", normalized)
    sentences = [part for part in sentence_split if part.strip()]
    lengths = [len(re.findall(r"\b[\w']+\b", s)) for s in sentences]
    if not lengths:
        return 0.5

    mean_length = sum(lengths) / len(lengths)
    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    std_dev = math.sqrt(variance)
    coefficient_of_variation = std_dev / mean_length if mean_length else 0.0
    type_token_ratio = len(set(tokens)) / max(1, len(tokens))
    punctuation_density = len(re.findall(r"[.!?,;:]", normalized)) / max(1, len(tokens))

    score = 0.2
    if coefficient_of_variation < 0.35:
        score += 0.25
    if type_token_ratio > 0.55:
        score += 0.15
    if punctuation_density > 0.06:
        score += 0.2
    if len(sentences) >= 3:
        score += 0.1
    if re.search(r"\b(i|me|my|mine|we|our|us)\b", normalized.lower()):
        score -= 0.12
    if re.search(r"\b(lol|kinda|sorta|gonna|wanna|honestly|super|pretty|really|totally|stuff)\b", normalized.lower()):
        score -= 0.1
    return clamp(score)


def combine_scores(llm_score: float, stylometry_score: float) -> float:
    combined = (0.6 * llm_score) + (0.4 * stylometry_score)
    return round(clamp(combined), 3)


def classify_score(score: float) -> Tuple[str, str]:
    if score >= 0.65:
        return "likely_ai", "High confidence this content appears to have been generated by AI."
    if score <= 0.35:
        return "likely_human", "High confidence this content appears to have been created by a human."
    return "uncertain", "Uncertain origin: this content could be AI-generated or human-authored."


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def log_event(content_id: str, creator_id: str | None, event_type: str, attribution: str | None, confidence: float | None, llm_score: float | None, heuristic_score: float | None, status: str | None, label: str | None, appeal_reasoning: str | None) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO audit_log (
            content_id, creator_id, timestamp, event_type, attribution, confidence,
            llm_score, heuristic_score, status, label, appeal_reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content_id,
            creator_id,
            now_iso(),
            event_type,
            attribution,
            confidence,
            llm_score,
            heuristic_score,
            status,
            label,
            appeal_reasoning,
        ),
    )
    conn.commit()
    conn.close()


@app.get("/")
def index() -> Any:
    return jsonify(
        {
            "service": "Provenance Guard",
            "status": "running",
            "endpoints": ["POST /submit", "POST /appeal", "GET /log", "GET /analytics"],
        }
    )


@app.post("/submit")
@limiter.limit("10 per minute; 100 per day")
def submit() -> Any:
    payload = request.get_json(silent=True) or {}
    text = normalize_text(payload.get("text", ""))
    creator_id = payload.get("creator_id") or "anonymous"

    if not text:
        return jsonify({"error": "A non-empty text field is required."}), 400

    llm_score = semantic_signal(text)
    heuristic_score = stylometric_signal(text)
    confidence = combine_scores(llm_score, heuristic_score)
    attribution, label = classify_score(confidence)
    content_id = uuid.uuid4().hex
    timestamp = now_iso()

    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO submissions (
            content_id, creator_id, created_at, text, status, attribution, confidence,
            label, llm_score, heuristic_score, appeal_reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            content_id,
            creator_id,
            timestamp,
            text,
            "classified",
            attribution,
            confidence,
            label,
            llm_score,
            heuristic_score,
            None,
        ),
    )
    conn.commit()
    conn.close()

    log_event(
        content_id=content_id,
        creator_id=creator_id,
        event_type="submission",
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        heuristic_score=heuristic_score,
        status="classified",
        label=label,
        appeal_reasoning=None,
    )

    return jsonify(
        {
            "content_id": content_id,
            "attribution": attribution,
            "confidence": confidence,
            "label": label,
            "transparency_label": label,
            "signals": {
                "llm_score": llm_score,
                "heuristic_score": heuristic_score,
            },
        }
    )


@app.post("/appeal")
def appeal() -> Any:
    payload = request.get_json(silent=True) or {}
    content_id = payload.get("content_id", "").strip()
    reasoning = normalize_text(payload.get("creator_reasoning", ""))

    if not content_id or not reasoning:
        return jsonify({"error": "content_id and creator_reasoning are required."}), 400

    conn = get_db_connection()
    submission = conn.execute(
        "SELECT * FROM submissions WHERE content_id = ?",
        (content_id,),
    ).fetchone()

    if submission is None:
        conn.close()
        return jsonify({"error": "content_id not found."}), 404

    if submission["status"] == "under_review":
        conn.close()
        return jsonify({"message": "Appeal already under review.", "content_id": content_id})

    conn.execute(
        "UPDATE submissions SET status = ?, appeal_reasoning = ? WHERE content_id = ?",
        ("under_review", reasoning, content_id),
    )
    conn.commit()
    conn.close()

    log_event(
        content_id=content_id,
        creator_id=submission["creator_id"],
        event_type="appeal",
        attribution=submission["attribution"],
        confidence=submission["confidence"],
        llm_score=submission["llm_score"],
        heuristic_score=submission["heuristic_score"],
        status="under_review",
        label=submission["label"],
        appeal_reasoning=reasoning,
    )

    return jsonify(
        {
            "message": "Appeal received and the content is now under review.",
            "content_id": content_id,
            "status": "under_review",
        }
    )


@app.get("/log")
def view_log() -> Any:
    conn = get_db_connection()
    entries = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()

    payload = []
    for entry in entries:
        payload.append(
            {
                "id": entry["id"],
                "content_id": entry["content_id"],
                "creator_id": entry["creator_id"],
                "timestamp": entry["timestamp"],
                "event_type": entry["event_type"],
                "attribution": entry["attribution"],
                "confidence": entry["confidence"],
                "llm_score": entry["llm_score"],
                "heuristic_score": entry["heuristic_score"],
                "status": entry["status"],
                "label": entry["label"],
                "appeal_reasoning": entry["appeal_reasoning"],
            }
        )

    return jsonify({"entries": payload})


@app.get("/analytics")
def analytics() -> Any:
    conn = get_db_connection()
    submissions = conn.execute(
        "SELECT attribution, confidence FROM submissions"
    ).fetchall()
    appeal_count = conn.execute(
        "SELECT COUNT(*) AS count FROM audit_log WHERE event_type = 'appeal'"
    ).fetchone()["count"]
    conn.close()

    total_submissions = len(submissions)
    ai_count = sum(1 for row in submissions if row["attribution"] == "likely_ai")
    human_count = sum(1 for row in submissions if row["attribution"] == "likely_human")
    uncertain_count = sum(1 for row in submissions if row["attribution"] == "uncertain")
    average_confidence = round(
        sum(row["confidence"] for row in submissions) / total_submissions,
        3,
    ) if total_submissions else 0.0
    appeal_rate = round(appeal_count / total_submissions, 3) if total_submissions else 0.0

    return jsonify(
        {
            "total_submissions": total_submissions,
            "detection_pattern": {
                "likely_ai": ai_count,
                "likely_human": human_count,
                "uncertain": uncertain_count,
            },
            "appeal_rate": appeal_rate,
            "average_confidence": average_confidence,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Provenance Guard.")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    args = parser.parse_args()
    port = args.port or int(os.getenv("PORT", "5000"))
    app.run(debug=True, host="0.0.0.0", port=port)
