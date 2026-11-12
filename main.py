
#!/usr/bin/env python3
"""
Local-LLM Reasoning Agent (Ollama)
Single-file submission.

Usage:
    python main.py --input input.json --output output.json

Sections:
    ANSWERS | MODEL | PROMPTS | SOLVER | CLI

Uses only Python standard library.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ======================================================================
# ANSWERS
# ======================================================================

_TRANS = {}

for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _TRANS[ord(c)] = str(i)

for i, c in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _TRANS[ord(c)] = str(i)

_TRANS.update({
    ord("٫"): ".",
    ord("٬"): ",",
    ord("،"): ",",
    ord("؛"): ";",
    ord("؟"): "?",
    ord("−"): "-",
    ord("–"): "-",
    ord("—"): "-",
    ord("ي"): "ی",
    ord("ك"): "ک",
})

_INVISIBLE = re.compile(
    "[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]"
)

_PERSIAN = re.compile(r"[\u0600-\u06FF]")


def normalize_text(s):
    s = str(s).translate(_TRANS)
    s = _INVISIBLE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[ \t\u00a0]+", " ", ln).strip()
        for ln in s.split("\n")
    ]
    return "\n".join(lines).strip()


def natural_key(s):
    out = []

    for tok in re.split(r"(\d+)", s.lower()):
        if tok.isdigit():
            out.append((0, int(tok), ""))
        elif tok:
            out.append((1, 0, tok))

    return out


def same_answer(a, b):
    return re.sub(r"\s+", "", str(a).lower()) == \
           re.sub(r"\s+", "", str(b).lower())


@dataclass
class FormatSpec:
    kind: str = "text"
    choices: list = field(default_factory=list)
    template: str = ""
    digits: int = 0
    source: str = "default"

    def describe(self):
        if self.kind == "integer":
            return "a single integer (digits only)"
        if self.kind == "number":
            return "a single number (digits only)"
        if self.kind == "letter":
            return "a single capital Latin letter"
        if self.kind == "choice":
            return "exactly one of: " + " | ".join(self.choices)
        if self.kind == "template":
            return "exactly this template: " + self.template
        if self.kind == "digits":
            return f"a single {self.digits}-digit code"
        return "only the final answer without explanation"

    @staticmethod
    def from_dict(d):
        return FormatSpec(
            kind=str(d.get("kind", "text")),
            choices=[
                normalize_text(c)
                for c in d.get("choices", [])
            ],
            template=normalize_text(d.get("template", "")),
            digits=int(d.get("digits", 0) or 0),
            source=d.get("source", "dataset")
        )


_TPL = re.compile(
    r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*<\s*([A-Za-z]+)\s*>"
)

_WORDNUM = {
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7
}


def _detect_template(text):
    matches = list(_TPL.finditer(text))

    if not matches:
        return None

    out = matches[0].group(0)

    for prev, m in zip(matches, matches[1:]):
        sep = text[prev.end():m.start()]

        if "\n" in sep or len(sep) > 6:
            break

        out += sep + m.group(0)

    return out


def _detect_inline_choice(text):
    m = re.search(
        r"یکی از (?:اعداد|حروف|گزینه‌های|عبارت‌های|عبارات)?"
        r"\s*([^\n]{1,60}?)\s*را\s*برگردان",
        text
    )

    if not m:
        return None

    tokens = [
        t for t in re.split(r"[,\s]+", m.group(1))
        if t and t not in ("یا", "و")
    ]

    if len(tokens) >= 2 and all(
        re.fullmatch(r"[A-Za-z0-9]{1,3}", t)
        for t in tokens
    ):
        return tokens

    return None


def _detect_choice_lines(text):
    lines = text.split("\n")

    def candidate(line):
        t = re.sub(r"^[-•*]\s*", "", line).strip()

        if t and not _PERSIAN.search(t) and len(t) <= 80:
            return t

        return None

    i = len(lines)
    block = []

    while i > 0 and candidate(lines[i - 1]) is not None:
        block.insert(0, candidate(lines[i - 1]))
        i -= 1

    if len(block) >= 2 and i > 0:
        context = "\n".join(lines[max(0, i - 2):i])

        if re.search(r"برگردان|یکی|گزینه|عبارت|بنویسید", context):
            return block

    return None


def detect_format(problem):
    text = normalize_text(problem)

    template = _detect_template(text)

    if template:
        return FormatSpec(
            kind="template",
            template=template,
            source="heuristic"
        )

    choices = _detect_inline_choice(text)

    if choices:
        return FormatSpec(
            kind="choice",
            choices=choices,
            source="heuristic"
        )

    choices = _detect_choice_lines(text)

    if choices:
        return FormatSpec(
            kind="choice",
            choices=choices,
            source="heuristic"
        )

    sentences = re.split(r"(?<=[.?!])\s+|\n", text)

    for sentence in sentences:

        if not re.search(r"برگردان|بنویسید|مشخص کنید", sentence):
            continue

        if "عدد صحیح" in sentence:
            return FormatSpec(kind="integer", source="heuristic")

        m = re.search(r"کد\s+(\S+)\s+رقمی", sentence)

        if m:
            word = m.group(1)
            n = _WORDNUM.get(word)

            if n is None and word.isdigit():
                n = int(word)

            if n:
                return FormatSpec(
                    kind="digits",
                    digits=n,
                    source="heuristic"
                )

        if "حرف" in sentence and "عدد" not in sentence:
            return FormatSpec(kind="letter", source="heuristic")

        if "عدد" in sentence:
            return FormatSpec(kind="number", source="heuristic")

    return None


def spec_from_model_json(obj):
    kind = str(obj.get("type", "text")).strip().lower()

    allowed = (
        "integer", "number", "letter",
        "choice", "template", "digits", "text"
    )

    if kind not in allowed:
        kind = "text"

    spec = FormatSpec(kind=kind, source="model")

    spec.choices = [
        normalize_text(c)
        for c in (obj.get("choices") or [])
        if str(c).strip()
    ]

    spec.template = normalize_text(obj.get("template") or "")

    try:
        spec.digits = int(obj.get("digits") or 0)
    except (TypeError, ValueError):
        spec.digits = 0

    if kind == "choice" and len(spec.choices) < 2:
        spec.kind = "text"

    if kind == "template" and not _TPL.search(spec.template):
        spec.kind = "text"

    if kind == "digits" and spec.digits <= 0:
        spec.kind = "text"

    return spec


# ----------------------------------------------------------------------
# ANSWER EXTRACTION
# ----------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_think(text):
    text = text or ""

    if "</think>" in text.lower() and "<think>" not in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]

    text = _THINK.sub("", text)

    if "<think>" in text.lower():
        text = re.split(r"<think>", text, flags=re.I)[0]

    return text.strip()


def _clean(text):
    text = normalize_text(text)

    text = re.sub(r"\\boxed\{(.*?)\}", r"\1", text)

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = text.replace("$", "")

    text = text.strip().strip("\"'“”«»").strip()

    text = re.sub(
        r"^(final answer|answer|پاسخ|جواب)\s*[:：]\s*",
        "",
        text,
        flags=re.I
    )

    return text.strip().rstrip(".").strip()


def _candidates(raw, final_only=False):
    text = strip_think(raw)
    out = []

    matches = re.findall(
        r"final\s*answer\s*[:：]\s*(.+)",
        text,
        flags=re.I
    )

    if matches:
        out.append(matches[-1])

    lines = [line for line in text.split("\n") if line.strip()]

    if lines and not final_only:
        out.append(lines[-1])

    return out


_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _numbers(text):
    return [
        m.group(0).replace(",", "")
        for m in _NUM.finditer(text)
    ]


def _after_equals(text):
    if "=" in text:
        tail = text.rsplit("=", 1)[1]

        if len(_numbers(tail)) == 1:
            return tail

    return text


def _parse_integer(text, relaxed):
    if len(_numbers(text)) > 1:
        text = _after_equals(text)

    nums = _numbers(text)

    if not nums or (len(nums) > 1 and not relaxed):
        return None

    value = nums[-1]

    if "." in value:
        number = float(value)

        if number != int(number):
            return None

        value = str(int(number))

    return str(int(value))


def _parse_number(text, relaxed):
    if len(_numbers(text)) > 1:
        text = _after_equals(text)

    nums = _numbers(text)

    if not nums or (len(nums) > 1 and not relaxed):
        return None

    value = nums[-1]

    if "." in value:
        value = value.rstrip("0").rstrip(".")

    return value


def _parse_letter(text, relaxed):
    text = text.strip()

    if re.fullmatch(r"[A-Za-z]", text):
        return text.upper()

    tokens = re.findall(
        r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])",
        text
    )

    if not tokens:
        return None

    if len({x.upper() for x in tokens}) == 1:
        return tokens[0].upper()

    return tokens[-1].upper() if relaxed else None


def _norm_key(text):
    return re.sub(r"\s+", "", text.lower()).rstrip(".")


def _parse_choice(text, spec, relaxed):
    key = _norm_key(text)

    for choice in spec.choices:
        if key == _norm_key(choice):
            return choice

    low = re.sub(r"\s+", " ", text.lower())

    found = []

    for choice in spec.choices:
        words = re.sub(r"\s+", " ", choice.lower()).split(" ")

        pattern = (
            r"(?<![A-Za-z0-9])"
            + r"\s*".join(re.escape(w) for w in words)
            + r"(?![A-Za-z0-9])"
        )

        matches = list(re.finditer(pattern, low))

        if matches:
            found.append((choice, matches[-1].start()))

    if len(found) == 1:
        return found[0][0]

    if found and relaxed:
        return max(found, key=lambda x: x[1])[0]

    return None


def _parse_digits(text, n, relaxed):
    runs = re.findall(r"\d+", text)

    if len(runs) == 1 and len(runs[0]) == n:
        return runs[0]

    joined = "".join(runs)

    if re.fullmatch(r"[\d\s,\-]+", text) and len(joined) == n:
        return joined

    if relaxed:
        good = [r for r in runs if len(r) == n]

        if good:
            return good[-1]

    return None


def _template_regex(template):
    parts = re.split(r"<\s*([A-Za-z]+)\s*>", template)

    rx = ""

    for i, part in enumerate(parts):

        if i % 2 == 0:
            for char in part:
                if char.isspace():
                    rx += r"\s*"
                elif char in "=,:;":
                    rx += r"\s*" + re.escape(char) + r"\s*"
                else:
                    rx += re.escape(char)

        else:
            kind = part.lower()

            if kind in ("integer", "int"):
                rx += r"(-?\d+)(?:\.0+)?"

            elif kind in ("number", "float", "decimal", "real"):
                rx += r"(-?\d+(?:\.\d+)?)"

            elif kind in ("letter", "char"):
                rx += r"([A-Za-z])"

            else:
                rx += r"([^\s,;]+)"

    return re.compile(rx, re.I), parts


def _parse_template(text, spec):
    try:
        rx, parts = _template_regex(spec.template)
    except re.error:
        return None

    matches = list(rx.finditer(text))

    if not matches:
        return None

    match = matches[-1]

    out = ""
    group = 1

    for i, part in enumerate(parts):

        if i % 2 == 0:
            out += part

        else:
            out += match.group(group)
            group += 1

    return out


def _parse(text, spec, relaxed):
    if not text:
        return None

    kind = spec.kind

    if kind == "integer":
        return _parse_integer(text, relaxed)

    if kind == "number":
        return _parse_number(text, relaxed)

    if kind == "letter":
        return _parse_letter(text, relaxed)

    if kind == "choice":
        return _parse_choice(text, spec, relaxed)

    if kind == "digits":
        return _parse_digits(text, spec.digits, relaxed)

    if kind == "template":
        return _parse_template(text, spec)

    return text.split("\n")[0].strip() or None


def extract_answer(raw, spec, relaxed=False, final_only=False):
    if not raw:
        return None

    for candidate in _candidates(raw, final_only):
        answer = _parse(_clean(candidate), spec, False)

        if answer:
            return answer

    if relaxed and not final_only:
        tail = strip_think(raw)[-400:]
        return _parse(_clean(tail), spec, True)

    return None


def last_line_cleaned(raw):
    candidates = _candidates(raw)

    return _clean(candidates[-1]) if candidates else ""


def fallback_answer(spec):
    if spec.kind == "choice" and spec.choices:
        return spec.choices[0]

    if spec.kind == "letter":
        return "A"

    if spec.kind == "digits" and spec.digits:
        return "1" * spec.digits

    if spec.kind == "template" and spec.template:

        def substitute(match):
            kind = match.group(1).lower()

            if kind in (
                "integer", "int", "number",
                "float", "decimal", "real"
            ):
                return "0"

            return "A"

        return re.sub(
            r"<\s*([A-Za-z]+)\s*>",
            substitute,
            spec.template
        )

    return "0"


# ======================================================================
# MODEL
# ======================================================================

log = logging.getLogger("agent")

PREFERRED_MODEL = "qwen3.8:27b"
SECOND_MODEL = "qwen3:8b"

FINAL_LINE = re.compile(
    r"FINAL ANSWER:[ \t]*\S[^\n]*\n",
    re.I
)

RETRY_DELAYS = [2.0, 4.0]


def _is_looping(text):
    if len(text) < 600:
        return False

    return text.count(text[-150:]) >= 3


def _answer_line_done(text):
    low = text.lower()

    if "</think>" in low:
        text = text[low.rfind("</think>") + len("</think>"):]

    elif "<think>" in low:
        return False

    return bool(FINAL_LINE.search(text))


def list_installed(host):
    url = host.rstrip("/") + "/api/tags"

    request = urllib.request.Request(url)

    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    return [
        m.get("name", "")
        for m in data.get("models", [])
        if m.get("name")
    ]


def choose_model(cli_model, installed):
    explicit = cli_model or os.environ.get("AGENT_MODEL")

    if explicit:
        return explicit

    low = {name.lower(): name for name in installed}

    if PREFERRED_MODEL in low:
        return low[PREFERRED_MODEL]

    if SECOND_MODEL in low:
        return low[SECOND_MODEL]

    qwen_models = sorted(
        name for name in installed
        if "qwen" in name.lower()
    )

    if qwen_models:
        return qwen_models[0]

    return PREFERRED_MODEL


def model_size_b(name):
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())

    return float(match.group(1)) if match else 0.0


@dataclass
class Settings:
    model: str
    big: bool
    think: str = "off"
    num_ctx: int = 4096
    num_predict: int = 1024
    max_calls: int = 5
    call_timeout: float = 120.0
    call_wall: float = 180.0
    time_limit: float = 240.0
    stream: bool = True
    host: str = "http://127.0.0.1:11434"
    hints: bool = False


def build_settings(
    model,
    think=None,
    max_calls=None,
    time_limit=None,
    host=None,
    hints=None
):
    big = model_size_b(model) >= 20

    st = Settings(
        model=model,
        big=big,
        max_calls=5 if big else 5,
        call_timeout=180 if big else 120,
        call_wall=240 if big else 180,
        time_limit=300 if big else 240,
        host=(
            host
            or os.environ.get(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434"
            )
        ).rstrip("/")
    )

    st.hints = bool(hints) or (
        os.environ.get("AGENT_HINTS", "").lower()
        in ("1", "on", "true")
    )

    st.think = (
        think or os.environ.get("AGENT_THINK") or "off"
    ).lower()

    if st.think not in ("off", "on", "auto"):
        st.think = "off"

    if st.think == "on":
        st.num_predict = 2048

    if max_calls is not None:
        st.max_calls = max(1, max_calls)

    if time_limit is not None:
        st.time_limit = max(10, time_limit)

    return st


# ----------------------------------------------------------------------
# OLLAMA HTTP CLIENT
# ----------------------------------------------------------------------

class LLM:

    def __init__(self, st, client=None):
        self.st = st
        self.calls = 0
        self.requests = 0
        self.last_error = ""
        self.last_truncated = False
        self._send_think = True
        self._soft_switch = False

    def _consume(self, response, start):
        parts = []
        error = None
        checked = 0

        try:
            for line in response:

                if time.time() - start > self.st.call_wall:
                    error = TimeoutError("Call wall limit exceeded")
                    break

                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line.decode("utf-8"))

                    message = chunk.get("message", {})
                    piece = message.get("content", "") or ""

                    if piece:
                        parts.append(piece)

                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])

                    if chunk.get("done"):
                        break

                except json.JSONDecodeError:
                    continue

                text_all = "".join(parts)

                if "\n" in piece and _answer_line_done(text_all):
                    break

                if len(text_all) - checked >= 200:
                    checked = len(text_all)

                    if _is_looping(text_all):
                        error = RuntimeError("Repetition loop")
                        break

        except Exception as e:
            error = e

        finally:
            response.close()

        return "".join(parts), error

    def chat(
        self,
        prompt,
        system=None,
        temperature=0.6,
        num_predict=None,
        json_mode=False
    ):
        self.calls += 1
        self.last_truncated = False

        options = {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": self.st.num_ctx,
            "num_predict": num_predict or self.st.num_predict
        }

        errors = 0

        for attempt in range(6):

            messages = []

            if system:
                messages.append({
                    "role": "system",
                    "content": system
                })

            user_prompt = prompt

            if self._soft_switch:
                user_prompt += " /no_think"

            messages.append({
                "role": "user",
                "content": user_prompt
            })

            payload = {
                "model": self.st.model,
                "messages": messages,
                "stream": self.st.stream,
                "options": options
            }

            if json_mode:
                payload["format"] = "json"

            if self._send_think and self.st.think in ("on", "off"):
                payload["think"] = self.st.think == "on"

            data = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                self.st.host + "/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            self.requests += 1
            start = time.time()

            try:
                response = urllib.request.urlopen(
                    request,
                    timeout=self.st.call_timeout
                )

                if self.st.stream:
                    text, error = self._consume(response, start)

                    if error:
                        self.last_error = str(error)
                        self.last_truncated = True

                        if not text.strip():
                            raise error

                    return strip_think(text)

                result = json.loads(response.read().decode("utf-8"))

                return strip_think(
                    result.get("message", {}).get("content", "")
                )

            except urllib.error.HTTPError as e:

                error_body = e.read().decode(
                    "utf-8",
                    errors="replace"
                )

                self.last_error = f"HTTP {e.code}: {error_body[:200]}"

                if e.code == 400 and "think" in error_body.lower():
                    self._send_think = False
                    self._soft_switch = True
                    continue

                if e.code >= 500 and errors < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[errors])
                    errors += 1
                    continue

                log.warning(self.last_error)
                return None

            except Exception as e:
                self.last_error = (
                    f"{type(e).__name__}: {str(e)[:200]}"
                )

                if isinstance(
                    e,
                    (urllib.error.URLError, ConnectionError)
                ) and errors < len(RETRY_DELAYS):
                    time.sleep(RETRY_DELAYS[errors])
                    errors += 1
                    continue

                log.warning(self.last_error)
                return None

        return None


# ======================================================================
# PROMPTS
# ======================================================================

P_SYSTEM = (
    "You are a careful mathematical and logical problem solver. "
    "Read Persian problems carefully. "
    "Reason step by step and verify every constraint."
)

P_RULES = """Rules:
- The problem is in Persian; digits are normalized.
- Use only relevant facts.
- List the facts and constraints logically.
- Check the answer against every constraint.
- If several answers satisfy all constraints, choose the smallest.
- Test each case in truth-teller and liar puzzles.
- Never leave the answer empty.
- Finish with exactly one last line:
FINAL ANSWER: <answer>
Write nothing after that line.
Required format: {fmt}
"""

P_EX_STEP = """Example:
Problem: A person walks 3 km/h for 2 hours and 4 km/h for 3 hours.
How many kilometres does he walk?
Reasoning:
1. First distance = 3 * 2 = 6.
2. Second distance = 4 * 3 = 12.
3. Total = 18.
FINAL ANSWER: 18
"""

P_EX_LTM = """Example:
Problem: A shop buys 12 crates with 15 apples each.
20 percent are rotten. How many good apples remain?
Sub-questions:
1. Total apples = 12 * 15 = 180.
2. Rotten apples = 36.
3. Good apples = 180 - 36 = 144.
FINAL ANSWER: 144
"""

P_STYLE_HEAD = {
    "stepwise": "Solve the problem step by step.",
    "ltm": (
        "Solve using least-to-most reasoning. "
        "First create sub-questions, then answer them."
    ),
    "analogical": (
        "Recall a simpler related problem and its solution method. "
        "Then solve the actual problem."
    )
}

P_STYLE_TAIL = {
    "stepwise": "Keep reasoning compact and finish with FINAL ANSWER.",
    "ltm": "Keep the sub-questions compact and finish with FINAL ANSWER.",
    "analogical": "Finish with FINAL ANSWER."
}

P_HINTS = """
Useful techniques:
- Grid paths: expand legal neighbours level by level.
- Ordering puzzles: list candidates and eliminate invalid ones.
- Truth/liar puzzles: test each assumption.
- Scheduling: calculate earliest finish times.
- Percentages: calculate each operation separately.
"""

P_STYLES = ["stepwise", "ltm", "analogical"]

TEMPS = [0.3, 0.5, 0.4, 0.6, 0.5, 0.4]


def P_solver_prompt(
    problem,
    fmt_desc,
    style,
    rewritten="",
    hints=False
):
    parts = [
        P_STYLE_HEAD[style],
        P_RULES.replace("{fmt}", fmt_desc)
    ]

    if hints:
        parts.append(P_HINTS)

    if style == "stepwise":
        parts.append(P_EX_STEP)

    elif style == "ltm":
        parts.append(P_EX_LTM)

    parts.append("PROBLEM:\n" + problem)

    if rewritten:
        parts.append("SUMMARY:\n" + rewritten)

    parts.append(P_STYLE_TAIL[style])

    return "\n\n".join(parts)


P_REWRITE = """Rewrite the Persian problem below as a compact English summary.

GIVEN: relevant facts and numbers.
CONSTRAINTS: all conditions.
QUESTION: what must be found.

Do not solve the problem.
Keep all names and numbers.

PROBLEM:
{problem}
"""

P_FORMAT_SPEC = """Read the problem and identify its required answer format.

Return JSON only:
{
"type": "integer|number|letter|choice|template|digits|text",
"choices": [],
"template": "",
"digits": 0
}

PROBLEM:
{problem}
"""

P_BASELINE = (
    "{problem}\n\n"
    "Return only the final answer in the exact required format."
)


# ======================================================================
# SOLVER
# ======================================================================

@dataclass
class Result:
    answer: str
    calls: int
    elapsed: float
    valid: bool
    source: str = ""
    votes: dict = field(default_factory=dict)
    spec_kind: str = ""
    trace: list = field(default_factory=list)


def _model_format(text, llm):
    raw = llm.chat(
        P_FORMAT_SPEC.replace("{problem}", text),
        temperature=0.0,
        num_predict=300,
        json_mode=True
    )

    if not raw:
        return None

    try:
        match = re.search(r"\{.*\}", raw, re.S)

        if match:
            return spec_from_model_json(
                json.loads(match.group(0))
            )

    except (ValueError, TypeError):
        pass

    return None


def _decided(votes):
    top = votes.most_common()

    if not top:
        return False

    first = top[0][1]
    second = top[1][1] if len(top) > 1 else 0

    return first >= 2 and first > second


def solve_agent(problem, llm, st, rewrite=False):
    t0 = time.time()
    c0 = llm.calls
    deadline = t0 + st.time_limit

    text = normalize_text(problem)

    spec = detect_format(text)

    if spec is None:
        spec = _model_format(text, llm)

    if spec is None:
        spec = FormatSpec(kind="text", source="default")

    rewritten = ""

    if rewrite and time.time() < deadline - 30:
        result = llm.chat(
            P_REWRITE.replace("{problem}", text),
            system=P_SYSTEM,
            temperature=0.2,
            num_predict=500
        )

        rewritten = (result or "")[:1200]

    votes = Counter()
    tails = []
    trace = []
    failures = 0
    n = 0

    while (
        llm.calls - c0 < st.max_calls
        and time.time() < deadline - 5
    ):

        style = P_STYLES[n % len(P_STYLES)]

        prompt = P_solver_prompt(
            text,
            spec.describe(),
            style,
            rewritten,
            st.hints
        )

        raw = llm.chat(
            prompt,
            system=P_SYSTEM,
            temperature=TEMPS[n % len(TEMPS)]
        )

        n += 1

        if raw is None:
            failures += 1

            trace.append({
                "style": style,
                "extracted": None,
                "error": llm.last_error
            })

            if failures >= 3:
                break

            continue

        failures = 0

        cut = llm.last_truncated

        if not cut:
            tails.append(raw[-500:])

        body = (
            raw.rsplit("\n", 1)[0]
            if cut and "\n" in raw
            else raw
        )

        answer = extract_answer(
            body,
            spec,
            final_only=cut
        )

        trace.append({
            "style": style,
            "extracted": answer,
            "chars": len(raw)
        })

        if answer:
            votes[answer] += 1

            if _decided(votes):
                break

    if votes:
        best = max(votes.values())

        winners = sorted(
            (
                answer for answer, count in votes.items()
                if count == best
            ),
            key=natural_key
        )

        answer = winners[0]
        source = "vote"

    else:
        answer = None
        source = ""

        for tail in reversed(tails):
            answer = extract_answer(
                tail,
                spec,
                relaxed=True
            )

            if answer:
                source = "salvage"
                break

        if not answer:
            answer = fallback_answer(spec)
            source = "fallback"

    return Result(
        answer=str(answer),
        calls=llm.calls - c0,
        elapsed=time.time() - t0,
        valid=True,
        source=source,
        votes=dict(votes),
        spec_kind=spec.kind,
        trace=trace
    )


def solve_baseline(problem, llm, st):
    t0 = time.time()
    c0 = llm.calls

    text = normalize_text(problem)

    spec = detect_format(text) or FormatSpec()

    raw = llm.chat(
        P_BASELINE.replace("{problem}", text),
        temperature=0.0
    )

    answer = extract_answer(raw or "", spec)

    valid = answer is not None

    if answer is None:
        answer = last_line_cleaned(raw or "")

    if not answer:
        answer = fallback_answer(spec)

    return Result(
        answer=answer,
        calls=llm.calls - c0,
        elapsed=time.time() - t0,
        valid=valid,
        source="baseline",
        spec_kind=spec.kind
    )


# ======================================================================
# CLI
# ======================================================================

def write_output(path, obj):
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2
        )

    os.replace(tmp, path)


def _decode(raw):
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")

    for encoding in ("utf-8-sig", "cp1256"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def load_input(path):
    with open(path, "rb") as f:
        text = _decode(f.read()).strip()

    try:
        data = json.loads(text)

    except json.JSONDecodeError:
        lines = [
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ]

        data = json.loads("\n".join(lines))

    if isinstance(data, list):
        return data, True

    return [data], False


def heuristic_fallback(problem):
    spec = detect_format(
        normalize_text(problem)
    ) or FormatSpec(kind="text")

    return fallback_answer(spec)


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Local LLM reasoning agent"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--think",
        default=None,
        choices=["off", "on", "auto"]
    )

    parser.add_argument("--max-calls", type=int, default=None)

    parser.add_argument(
        "--time-limit",
        type=float,
        default=None
    )

    parser.add_argument("--host", default=None)

    parser.add_argument(
        "--hints",
        action="store_true"
    )

    parser.add_argument(
        "--rewrite",
        action="store_true"
    )

    parser.add_argument(
        "--baseline",
        action="store_true"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=(
            logging.INFO
            if args.verbose
            else logging.WARNING
        ),
        stream=sys.stderr,
        format="%(message)s"
    )

    try:
        items, is_list = load_input(args.input)

    except Exception as e:
        log.error("Cannot read input: %s", e)

        write_output(
            args.output,
            {"id": "", "answer": "0"}
        )

        return 1

    results = []

    for item in items:

        if not isinstance(item, dict):
            item = {}

        results.append({
            "id": item.get("id", ""),
            "answer": heuristic_fallback(
                item.get("problem", "")
            )
        })

    # Write fallback output before connecting to Ollama.
    write_output(
        args.output,
        results if is_list else results[0]
    )

    try:
        host = (
            args.host
            or os.environ.get(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434"
            )
        ).rstrip("/")

        try:
            installed = list_installed(host)

        except Exception as e:
            log.warning(
                "Could not list Ollama models: %s",
                e
            )

            installed = []

        model = choose_model(
            args.model,
            installed
        )

        st = build_settings(
            model,
            think=args.think,
            max_calls=args.max_calls,
            time_limit=args.time_limit,
            host=host,
            hints=args.hints
        )

        llm = LLM(st)

        log.info("Model: %s", model)

    except Exception as e:
        log.error("Model setup failed: %s", e)
        return 0

    for i, item in enumerate(items):

        try:
            problem = str(
                (item or {}).get("problem", "")
            )

            if args.baseline:
                result = solve_baseline(
                    problem,
                    llm,
                    st
                )

            else:
                result = solve_agent(
                    problem,
                    llm,
                    st,
                    rewrite=args.rewrite
                )

            results[i]["answer"] = str(result.answer)

            log.info(
                "Problem %s | answer=%s | calls=%d | source=%s",
                results[i]["id"],
                result.answer,
                result.calls,
                result.source
            )

        except Exception as e:
            log.error(
                "Problem %s failed: %s",
                results[i]["id"],
                e
            )

        write_output(
            args.output,
            results if is_list else results[0]
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
