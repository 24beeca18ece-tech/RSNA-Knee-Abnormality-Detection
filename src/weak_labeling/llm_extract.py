"""LLM-based weak-labeling of knee MRI report text.

Why an LLM instead of regex/keyword matching: exploration of the 4349
report-only rows (2026-08-07) found reports spanning at least 9 languages
(English ~41%, Spanish ~16%, Turkish ~10%, Croatian ~9%, Greek ~8%, German
~6%, Bulgarian ~4%, Dutch ~4%, French ~1%), inconsistent structure (some
use explicit section headers, others free prose), and pervasive negation
("no tear", "sin signos de rotura", "normal", "intakt", "b.o.") that a
keyword match would misclassify as positive findings. This is exactly the
class of clinical-NLP problem where naive term matching is known to fail
(see docs/baseline_plan.md "Step 2" for the full writeup) - an LLM that
reads the whole sentence handles negation and cross-language terminology
directly.

Two providers:
- claude: shells out to the already-authenticated Claude Code CLI
  (`claude -p ... --output-format json --json-schema ...`) rather than a
  separate API key, since this machine has no ANTHROPIC_API_KEY configured
  but Claude Code itself is already logged in. Used for the initial 58-row
  validation and the first 300 rows of the full run.
- groq: calls Groq's OpenAI-compatible chat completions endpoint directly
  with an API key (from GROQ_API_KEY env var or a local .env file), to
  avoid burning Claude Code session credits on the bulk of the 4349-row
  run. Also emits a per-report "confidence" field (high/low; see
  SYSTEM_PROMPT's confidence paragraph) so low-confidence rows can be
  bucketed separately rather than trusted outright - see
  scripts/weak_label_reports.py.
- gemini: calls the plain Gemini Developer API (generativelanguage.
  googleapis.com, free tier, GEMINI_API_KEY) directly - NOT the Vertex AI /
  Agent Platform route, which was abandoned after hitting an unresolved
  BILLING_DISABLED block on the linked GCP project. Model is gemini-3.6-flash
  (Pro tier 429s immediately on this free-tier key; the older 2.x line
  404s as "no longer available to new users"). Like gpt-oss on Groq, this
  model can't fully disable its internal "thinking" (thinkingBudget=0 is
  rejected for the 3.x Flash family) - thinkingBudget is capped low instead
  to keep it from eating the batch's completion budget the way Groq's
  reasoning did before that got capped.

This is strictly an offline, one-time, local data-prep step (see
docs/baseline_plan.md) - it never runs in the scored/internet-off Kaggle
notebook, so calling external LLM APIs here is fine.

JSON Schema property names must match ^[a-zA-Z0-9_.-]{1,64}$ (no spaces or
apostrophes), so target names are sanitized for the schema/response and
mapped back to the real column names (with spaces / the apostrophe in
"Baker's") immediately after parsing - see SAFE_TO_REAL.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import requests

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

REAL_TO_SAFE = {
    "ACL": "ACL", "MCL": "MCL",
    "Medial Meniscus": "Medial_Meniscus", "Lateral Meniscus": "Lateral_Meniscus",
    "Medial OA": "Medial_OA", "Lateral OA": "Lateral_OA", "PF OA": "PF_OA",
    "Effusion": "Effusion", "Synovitis": "Synovitis", "Baker's": "Bakers",
    "Contusion": "Contusion", "Fracture": "Fracture",
}
SAFE_TO_REAL = {v: k for k, v in REAL_TO_SAFE.items()}

CLAUDE_EXE = r"C:\Users\Brij Nandan Dogra\AppData\Roaming\npm\node_modules\@anthropic-ai\claude-code\bin\claude.exe"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Largest currently-available Groq model with strict JSON-schema structured
# output support (checked live against /v1/models on 2026-08-11) - favored
# over smaller/faster options given this is nuanced multilingual clinical text.
GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"

GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Newest GA flash-tier model available on the free-tier Gemini Developer API
# key as of 2026-08-12 (checked live against /v1beta/models): Pro-tier
# (gemini-3.1-pro-preview, gemini-pro-latest) 429s immediately on this key's
# quota, and the whole 2.5 line 404s as "no longer available to new users".
# Pinned to an explicit version rather than the "-latest" alias for a
# reproducible one-time extraction run.
GEMINI_MODEL_DEFAULT = "gemini-3.6-flash"


def _load_env_key(name: str) -> str:
    """Env var `name`, or from a .env file at the repo root (gitignored -
    see .gitignore) if not already set."""
    key = os.environ.get(name)
    if key:
        return key
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"{name} not set and not found in {env_path}")


def _load_groq_api_key() -> str:
    return _load_env_key("GROQ_API_KEY")


def _load_gemini_api_key() -> str:
    return _load_env_key("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist assistant extracting structured findings from knee MRI reports. Reports may be in any language (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, or others) - read and interpret each in its own language, do not assume English.

For EACH report, determine whether each of these 12 findings is clearly PRESENT (1), clearly ABSENT/NORMAL (0), or NOT ASSESSABLE (null):

- ACL: anterior cruciate ligament tear, sprain, or injury
- MCL: medial collateral ligament tear, sprain, or injury
- Medial Meniscus: medial meniscus tear
- Lateral Meniscus: lateral meniscus tear
- Medial OA: osteoarthritis / degenerative changes / chondropathy / cartilage loss specifically in the MEDIAL femorotibial compartment
- Lateral OA: osteoarthritis / degenerative changes / chondropathy / cartilage loss specifically in the LATERAL femorotibial compartment
- PF OA: osteoarthritis / degenerative changes / chondropathy / cartilage loss in the PATELLOFEMORAL compartment
- Effusion: joint effusion / fluid in the joint
- Synovitis: synovitis / synovial thickening / synovial inflammation
- Baker's: Baker's cyst / popliteal cyst
- Contusion: bone contusion / bone marrow edema of traumatic origin ("bone bruise")
- Fracture: any fracture

CRITICAL - negation: pay very careful attention to negation in whatever language the report is in (e.g. "no tear", "sin signos de rotura", "normal", "intact", "yoktur", "b.o.", "ohne", "sans", "zonder") - a negated finding is ABSENT (0), not present (1). Do not guess based on anatomy terms appearing in the text alone; read what is actually stated.

CRITICAL - severity threshold: for ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, and Contusion, only score 1 if the finding is described as at least MODERATE severity (e.g. "grade 2" or higher, "moderate", "high-grade", or an unqualified/significant finding - an unqualified mention with no size/grade word at all should count as significant). If explicitly qualified as mild, minimal, trace, low-grade, small, "some", or grade 1, score 0 even though the finding is textually present - this dataset's labeling convention treats mild/low-grade/small findings in these categories as negative, this includes small/mild/minimal effusions. This does NOT apply to Synovitis, Baker's, or Fracture - score those 1 whenever present regardless of described size/severity.

CRITICAL - default when not mentioned: if a structure/finding is not explicitly discussed at all, but the report otherwise appears to be a normal/complete assessment of the knee (i.e. other structures were reviewed and reported), score that finding 0 (absent) rather than null - the convention in this dataset is that unmentioned findings are normal. Reserve null for cases where the report is genuinely incomplete/cut off, illegible, or explicitly states a structure could not be evaluated.

CONFIDENCE (when the schema includes a "confidence" field): after scoring all 12 findings for a report, also output "confidence": "low" if ANY of the following apply, otherwise "high":
- the report's findings are ambiguous, self-contradictory, or use hedging language throughout ("possible", "cannot exclude", "equivocal") for multiple findings
- the report is unusually short, garbled, poorly translated, or missing whole sections you'd expect in a knee MRI report
- you had to rely on the "default when not mentioned = 0" rule (rather than explicit textual evidence) for 2 or more of the 12 findings, because large portions of the anatomy were never discussed
- you are genuinely unsure between two plausible readings for 2 or more findings
A report can be "high" confidence even with some null/absent findings, as long as what IS stated is clear and unambiguous.

Output strictly via the provided JSON schema: one object per report, in the same order and using the same "id" values given in the input, with one field per finding using the exact field names in the schema."""


def build_batch_schema_dict(safe_keys: list[str], include_confidence: bool = False) -> dict:
    finding_props = {k: {"type": ["integer", "null"], "enum": [0, 1, None]} for k in safe_keys}
    item_props = {"id": {"type": "string"}, **finding_props}
    required = ["id"] + safe_keys
    if include_confidence:
        item_props["confidence"] = {"type": "string", "enum": ["high", "low"]}
        required.append("confidence")
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def build_batch_schema(safe_keys: list[str]) -> str:
    return json.dumps(build_batch_schema_dict(safe_keys))


def build_batch_schema_gemini(safe_keys: list[str], include_confidence: bool = False) -> dict:
    """Gemini's responseSchema is an OpenAPI-3.0 subset, not plain JSON
    Schema: types are UPPERCASE strings, nullability is a separate
    "nullable" boolean rather than a `type` union, and there's no
    `additionalProperties` keyword."""
    finding_props = {k: {"type": "INTEGER", "nullable": True} for k in safe_keys}
    item_props = {"id": {"type": "STRING"}, **finding_props}
    required = ["id"] + safe_keys
    if include_confidence:
        item_props["confidence"] = {"type": "STRING", "enum": ["high", "low"]}
        required.append("confidence")
    return {
        "type": "OBJECT",
        "properties": {
            "results": {
                "type": "ARRAY",
                "items": {"type": "OBJECT", "properties": item_props, "required": required},
            }
        },
        "required": ["results"],
    }


def build_batch_prompt(rows: list[tuple[str, str]]) -> str:
    parts = [f"Label the following {len(rows)} knee MRI reports.\n"]
    for report_id, text in rows:
        parts.append(f"--- report id={report_id} ---\n{text}\n")
    return "\n".join(parts)


def call_claude_cli(prompt: str, schema_json: str, model: str = "claude-sonnet-5",
                     max_retries: int = 3, timeout: int = 420) -> dict:
    """Shell out to the Claude Code CLI for one structured-output completion.
    Returns the parsed `structured_output` dict. Raises on repeated failure.

    The prompt is passed via stdin (no positional `-p <prompt>` argument),
    not as a command-line argument - large batches (many reports
    concatenated) blow past Windows' ~32K command-line length limit
    (WinError 206) when passed as argv; stdin has no such limit.
    """
    cmd = [
        CLAUDE_EXE, "-p",
        "--system-prompt", SYSTEM_PROMPT,
        "--output-format", "json",
        "--json-schema", schema_json,
        "--model", model,
        "--allowedTools", "",
    ]
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
            if proc.returncode != 0:
                last_err = f"exit {proc.returncode}: {proc.stderr[:500] or proc.stdout[:500]}"
                time.sleep(2 * attempt)
                continue
            resp = json.loads(proc.stdout)
            if resp.get("is_error"):
                last_err = f"is_error: {resp.get('result')}"
                time.sleep(2 * attempt)
                continue
            structured = resp.get("structured_output")
            if structured is None:
                # fall back to parsing `result` as JSON text
                structured = json.loads(resp["result"])
            return structured
        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            last_err = str(e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"claude CLI call failed after {max_retries} attempts: {last_err}")


def call_groq_api(prompt: str, schema_dict: dict, model: str = GROQ_MODEL_DEFAULT,
                   max_retries: int = 4, timeout: int = 35) -> dict:
    """Call Groq's OpenAI-compatible chat completions endpoint for one
    structured-output completion. Returns the parsed JSON content dict.

    timeout/max_retries deliberately tight (35s x4 rather than the earlier
    120s x5): a 2026-08-13 run showed batches occasionally hanging for
    10-20+ min per attempt (not a clean 429 - looked like Groq-side
    congestion a plain HTTP timeout doesn't reliably catch fast). A failed
    batch here just gets skipped and retried later (see
    scripts/weak_label_reports.py's per-batch try/except) - better to fail
    a batch fast and move on than block the whole run on one stuck request.

    Rate-limit handling: reads the actual x-ratelimit-* response headers
    (rather than assuming a fixed budget) and sleeps until the token/request
    window resets on 429, or proactively sleeps a beat if the *previous*
    response showed the window nearly exhausted - "respect rate limits with
    backoff rather than hammering and failing" per the task that added this.
    """
    api_key = _load_groq_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "extraction", "schema": schema_dict}},
    }
    if "gpt-oss" in model:
        # gpt-oss is a reasoning model - its internal chain-of-thought can
        # consume the whole completion token budget before it ever emits
        # the final JSON on multi-report batches (seen empirically: batches
        # of 3+ reports failed with "max completion tokens reached before
        # generating a valid document"). Capping reasoning effort low keeps
        # the visible answer itself, which is all this task needs.
        payload["reasoning_effort"] = "low"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(3 * attempt)
            continue

        if resp.status_code == 413:
            # Request itself exceeds the per-minute token budget (e.g. batch
            # too large) - retrying the identical payload will never
            # succeed, no matter how long we wait. Fail fast so the caller
            # (scripts/weak_label_reports.py) can be re-run with a smaller
            # --batch-size instead of burning all max_retries pointlessly.
            raise RuntimeError(f"413 request too large (reduce --batch-size): {resp.text[:400]}")
        if resp.status_code == 429:
            retry_after = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens")
            wait = _parse_wait_seconds(retry_after) if retry_after else 10 * attempt
            last_err = f"429 rate limited, waiting {wait:.1f}s"
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            last_err = f"server error {resp.status_code}: {resp.text[:300]}"
            time.sleep(5 * attempt)
            continue
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
            time.sleep(3 * attempt)
            continue

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            structured = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = f"parse error: {e}: {resp.text[:300]}"
            time.sleep(3 * attempt)
            continue

        # Proactively back off if we're close to the per-minute token budget,
        # so the *next* call doesn't get a hard 429.
        remaining_tokens = resp.headers.get("x-ratelimit-remaining-tokens")
        reset_tokens = resp.headers.get("x-ratelimit-reset-tokens")
        if remaining_tokens is not None and int(remaining_tokens) < 500 and reset_tokens:
            time.sleep(_parse_wait_seconds(reset_tokens))
        return structured

    raise RuntimeError(f"groq API call failed after {max_retries} attempts: {last_err}")


class GeminiDailyQuotaExhausted(Exception):
    """Raised instead of retrying when a 429 is clearly the free tier's
    per-day-per-model quota (confirmed 2026-08-12: only 20 requests/day for
    gemini-3.6-flash), not a short-window rate limit. Retrying that with
    backoff is pointless - it won't reset for hours - so the caller should
    stop the whole run rather than burn through every remaining batch
    retrying and failing one by one."""


def call_gemini_api(prompt: str, schema_dict: dict, model: str = GEMINI_MODEL_DEFAULT,
                     max_retries: int = 5, timeout: int = 120, thinking_budget: int = 128) -> dict:
    """Call the plain Gemini Developer API for one structured-output
    completion. Returns the parsed JSON content dict.

    thinking_budget caps (not disables - the 3.x Flash family rejects
    thinkingBudget=0) the model's internal reasoning tokens, which
    otherwise scale with batch size and can crowd out the actual JSON
    answer - the same failure mode hit with Groq's gpt-oss reasoning,
    fixed the same way here pre-emptively rather than rediscovering it.

    No x-ratelimit-* response headers are exposed by this API (unlike
    Groq), so backoff on 429 uses Retry-After if present, else a plain
    escalating wait. Daily-quota 429s raise GeminiDailyQuotaExhausted
    immediately instead (see that class's docstring).
    """
    api_key = _load_gemini_api_key()
    url = GEMINI_API_URL_TEMPLATE.format(model=model)
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema_dict,
            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            time.sleep(3 * attempt)
            continue

        if resp.status_code == 429:
            if "PerDay" in resp.text:
                raise GeminiDailyQuotaExhausted(resp.text[:500])
            retry_after = resp.headers.get("retry-after")
            wait = _parse_wait_seconds(retry_after) if retry_after else 15 * attempt
            last_err = f"429 rate limited, waiting {wait:.1f}s"
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            last_err = f"server error {resp.status_code}: {resp.text[:300]}"
            time.sleep(5 * attempt)
            continue
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {resp.text[:500]}"
            time.sleep(3 * attempt)
            continue

        try:
            data = resp.json()
            parts = data["candidates"][0]["content"]["parts"]
            text = next(p["text"] for p in parts if "text" in p)
            structured = json.loads(text)
        except (KeyError, IndexError, StopIteration, json.JSONDecodeError) as e:
            last_err = f"parse error: {e}: {resp.text[:300]}"
            time.sleep(3 * attempt)
            continue
        return structured

    raise RuntimeError(f"gemini API call failed after {max_retries} attempts: {last_err}")


def _parse_wait_seconds(value: str) -> float:
    """Groq's reset headers look like '1.665s' or '1m26.4s'; retry-after may
    just be a plain number of seconds."""
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    total = 0.0
    num = ""
    for ch in value:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch == "m":
            total += float(num or 0) * 60
            num = ""
        elif ch == "s":
            total += float(num or 0)
            num = ""
    return total + 1.0  # small safety margin


def extract_batch(rows: list[tuple[str, str]], model: str = "claude-sonnet-5",
                   provider: str = "claude") -> dict[str, dict[str, int | str | None]]:
    """rows: list of (report_id, report_text). Returns {report_id: {target_name: 0/1/None, ["confidence": "high"/"low"]}}."""
    safe_keys = list(REAL_TO_SAFE.values())
    prompt = build_batch_prompt(rows)

    if provider == "claude":
        schema_json = build_batch_schema(safe_keys)
        structured = call_claude_cli(prompt, schema_json, model=model)
        extra_keys = []
    elif provider == "groq":
        schema_dict = build_batch_schema_dict(safe_keys, include_confidence=True)
        structured = call_groq_api(prompt, schema_dict, model=model)
        extra_keys = ["confidence"]
    elif provider == "gemini":
        schema_dict = build_batch_schema_gemini(safe_keys, include_confidence=True)
        structured = call_gemini_api(prompt, schema_dict, model=model)
        extra_keys = ["confidence"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    out: dict[str, dict[str, int | str | None]] = {}
    for item in structured["results"]:
        report_id = item["id"]
        row = {SAFE_TO_REAL[k]: item.get(k) for k in safe_keys}
        for k in extra_keys:
            row[k] = item.get(k)
        out[report_id] = row
    return out
