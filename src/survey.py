"""
survey.py -- Eye-strain survey storage for PredEye

Data is written to data/survey/survey_log.jsonl (append-only).
Admin access requires a password verified against a SHA-256 hash.

To change the admin password, run:
    import hashlib; print(hashlib.sha256(b'your-password').hexdigest())
and paste the result into ADMIN_HASH below.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

SURVEY_DIR  = Path("data") / "survey"
SURVEY_FILE = SURVEY_DIR / "survey_log.jsonl"

# Replace the string below with the SHA-256 hash of your chosen admin password.
ADMIN_HASH = "REPLACE_WITH_SHA256_OF_YOUR_PASSWORD"


def save_response(user_id, typed_text, session_sec, mode, responses):
    SURVEY_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "user_id":     user_id,
        "chars_typed": len(typed_text.strip()),
        "session_sec": round(session_sec, 1),
        "mode":        mode,
        "responses":   responses,
    }
    with SURVEY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def verify_admin(password):
    return hashlib.sha256(password.encode()).hexdigest() == ADMIN_HASH


def load_all(password):
    if not verify_admin(password):
        raise PermissionError("Invalid admin password.")
    if not SURVEY_FILE.exists():
        return []
    records = []
    with SURVEY_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_summary(records):
    if not records:
        print("No survey responses found.")
        return
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  PredEye Eye-Strain Survey  --  {len(records)} response(s)")
    print(sep)
    print("\n  Recent responses (latest 20):\n")
    for rec in records[-20:]:
        ts    = rec.get("timestamp", "?")
        uid   = rec.get("user_id",   "?")
        mode  = rec.get("mode",      "?")
        chars = rec.get("chars_typed", 0)
        secs  = rec.get("session_sec", 0)
        print(f"  [{ts}]  user={uid}  mode={mode}  chars={chars}  sec={secs}")
        for q, a in rec.get("responses", {}).items():
            print(f"      {q:<30} -> {a}")
        print()
    print(f"  {'-'*64}")
    print("  Aggregate statistics:\n")
    strain_vals = [r.get("responses", {}).get("eye_strain") for r in records]
    strain_vals = [v for v in strain_vals if isinstance(v, (int, float))]
    if strain_vals:
        avg = sum(strain_vals) / len(strain_vals)
        high = sum(1 for v in strain_vals if v >= 3)
        print(f"    Eye strain (1-4): avg={avg:.2f}  high(>=3): {high}/{len(strain_vals)}")
    typed_ok = [r.get("responses", {}).get("typed_intended") for r in records]
    print(f"    Typed intended:   Yes={typed_ok.count('Yes')}  Partial={typed_ok.count('Partially')}  No={typed_ok.count('No')}")
    overall = [r.get("responses", {}).get("overall_experience") for r in records]
    print(f"    Overall:          Great={overall.count('Great')}  OK={overall.count('OK')}  Poor={overall.count('Poor')}")
    print(f"\n{sep}\n")
