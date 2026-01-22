import re
import json
from datasets import load_dataset

# GDPR Art8
AGE_UNDER_13 = re.compile(
    r"\b(?:"
    r"(?:[0-9]|1[0-2])\s*[- ]?(?:year|yr|yo|y/o|years)\s*old"
    r"|age\s*(?:[0-9]|1[0-2])"
    r"|(?:[0-9]|1[0-2])\s*(?:yo|y/o)\b"
    r")\b",
    re.IGNORECASE
)

CHILD_CONTEXT = re.compile(
    r"\b(?:my|our)\s*(?:kid|child|son|daughter|niece|nephew|little\s+brother|little\s+sister)\b"
    r"|\b(?:kindergarten|elementary\s+school|middle\s+school|grade\s*(?:k|[1-8]))\b",
    re.IGNORECASE
)

PARENTAL_CONSENT = re.compile(
    r"\b(?:parental\s+consent|consent\s+form|guardian|legal\s+guardian|permission\s+slip)\b",
    re.IGNORECASE
)

def match_gdpr8(text: str) -> bool:
    if not text:
        return False
    return bool(AGE_UNDER_13.search(text) or (CHILD_CONTEXT.search(text) and re.search(r"\b(?:sign\s*up|account|register|app|website)\b", text, re.I)) or PARENTAL_CONSENT.search(text))

# GDPR Art9
ART9_TERMS = re.compile(
    r"\b(?:"
    r"diagnos(?:is|ed)|symptom|treatment|medication|prescription|doctor|hospital|clinic|mental\s+health|depression|anxiety|adhd|autism|cancer|hiv|pregnan(?:t|cy)"
    r"|religion|christian|muslim|jewish|hindu|buddhis(?:t|m)|atheis(?:t|m)|church|mosque|synagogue|temple"
    r"|politic(?:s|al)|democrat|republican|conservative|liberal|leftist|right[- ]?wing|election|vote\s+for|party\s+member"
    r"|race|ethnic(?:ity)?|asian|black|white|latino|hispanic|indigenous"
    r"|union\s+member|trade\s+union"
    r"|biometric|fingerprint|facial\s+recognition|iris\s+scan"
    r"|genetic|dna|genome|ancestry\s+test"
    r"|sexual\s+orientation|gay|lesbian|bisexual|transgender|sex\s+life"
    r")\b",
    re.IGNORECASE
)

def match_gdpr9(text: str) -> bool:
    if not text:
        return False
    return bool(ART9_TERMS.search(text))


DATASET_NAME = "allenai/WildChat-1M"
OUT_PATH = "outputs/wildchat_candidates_gdpr8_gdpr9.jsonl"

ds = load_dataset(DATASET_NAME, split="train", streaming=True)

seen_text_hashes = set()
kept = 0

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for record in ds:
        conv = record.get("conversation", [])
        for utt in conv:
            if utt.get("role") != "user":
                continue

            text = (utt.get("content") or "").strip()
            if not text:
                continue

            lang = utt.get("language")
            if lang and lang.lower() != "english":
                continue

            is_8 = match_gdpr8(text)
            is_9 = match_gdpr9(text)
            if not (is_8 or is_9):
                continue

            h = hash(text.lower())
            if h in seen_text_hashes:
                continue
            seen_text_hashes.add(h)

            out = {
                "policy_hits": {
                    "gdpr_art8": bool(is_8),
                    "gdpr_art9": bool(is_9),
                },
                "user_text": text,
                "turn_identifier": utt.get("turn_identifier"),
                "utterance_language": lang,
                "conversation_hash": record.get("conversation_hash"),
                "model": record.get("model"),
                "timestamp_utc": (
                    record.get("timestamp").isoformat()
                    if record.get("timestamp") is not None
                    else None
                ),
                "redacted": record.get("redacted"),
                "toxic": record.get("toxic"),
            }
            f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
            kept += 1

print(f"Wrote {kept} candidate user turns to {OUT_PATH}")