#!/usr/bin/env python3
"""
humanizer_server.py
===================

A Model Context Protocol (MCP) server that rewrites AI-sounding text so it reads
as naturally human, in English or Arabic.

Features
--------
* Language auto-detection (langdetect + Arabic-script heuristic) and dedicated
  system prompts for Arabic and English.
* Guardrail loop: word-count and keyword validation with up to 3 retries.
* Semantic cache in Redis (all-MiniLM-L6-v2 embeddings, cosine >= 0.92).
  Falls back gracefully if Redis or the embedder is unavailable.
* Analytics: Burstiness Index and heuristic AI Detection Risk Score
  (before and after), plus keyword preservation confirmation.

Install
-------
    pip install "mcp[cli]" langchain-groq langchain-core sentence-transformers \
                redis langdetect numpy

Environment variables
---------------------
    GROQ_API_KEY            (required) Groq API key
    GROQ_MODEL              default: openai/gpt-oss-120b
    REDIS_URL               default: redis://localhost:6379/0
    HUMANIZER_EMBED_MODEL   default: all-MiniLM-L6-v2
    HUMANIZER_CACHE_TTL     default: 604800 (7 days, in seconds)
    MCP_TRANSPORT           stdio (default) | sse | streamable-http

Run
---
    python humanizer_server.py
"""

from __future__ import annotations
from pydantic import BaseModel, Field

import asyncio
import hashlib
import logging
import math
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langdetect import DetectorFactory, LangDetectException, detect_langs
# from mcp.server.fastmcp import FastMCP
from mcp.server.mcpserver import MCPServer as FastMCP
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------- #
# Configuration & logging (stdout is reserved for the MCP stdio protocol)
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("humanizer")

DetectorFactory.seed = 0  # deterministic language detection

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBED_MODEL_NAME = os.getenv("HUMANIZER_EMBED_MODEL", "all-MiniLM-L6-v2")
CACHE_TTL_SECONDS = int(os.getenv("HUMANIZER_CACHE_TTL", str(7 * 24 * 3600)))
CACHE_PREFIX = "humanizer:cache:"
SIMILARITY_THRESHOLD = 0.92
MAX_RETRIES = 3  # retries after the first attempt (4 attempts total)
SCAN_BATCH = 200

LANG_NAMES = {"en": "English", "ar": "Arabic"}

# --------------------------------------------------------------------------- #
# AI-cliché lexicons
# --------------------------------------------------------------------------- #
EN_CLICHES: list[str] = [
    "delve", "delving", "tapestry", "testament", "landscape", "realm",
    "moreover", "furthermore", "additionally", "in conclusion",
    "it is important to note", "it is worth noting", "crucial", "pivotal",
    "leverage", "leveraging", "utilize", "utilizing", "seamless", "seamlessly",
    "robust", "holistic", "game-changer", "game changer", "navigate",
    "navigating", "foster", "fostering", "underscore", "multifaceted",
    "intricate", "ever-evolving", "fast-paced world", "in today's",
    "unlock", "elevate", "embark", "journey", "cutting-edge", "groundbreaking",
    "paradigm", "synergy", "comprehensive", "meticulous", "vibrant", "beacon",
    "plays a vital role", "plays a crucial role", "a myriad of", "rich tapestry",
    "when it comes to", "in the realm of", "stands as a testament",
]

AR_CLICHES: list[str] = [
    "في عالمنا اليوم", "في عصرنا الحالي", "في عصرنا الحديث", "من الجدير بالذكر",
    "تجدر الإشارة", "لا شك أن", "مما لا شك فيه", "لا يخفى على أحد",
    "في الختام", "علاوة على ذلك", "بالإضافة إلى ذلك", "في هذا السياق",
    "يلعب دورا محوريا", "دورا بارزا", "دورا حيويا", "في ظل", "نسيج",
    "ثورة حقيقية", "عصر التحول الرقمي", "يعد من أهم", "على نحو شامل",
    "في غضون ذلك", "لا يمكن إنكار", "يمثل حجر الزاوية", "رحلة",
    "آفاق جديدة", "نقلة نوعية",
]

_AR_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_WORD_RE = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[.!?؟…。])\s+|\n+")


def normalize_text(text: str) -> str:
    """Lowercase and normalize Arabic orthography so lexicon and keyword matching is robust."""
    text = _AR_DIACRITICS.sub("", text.lower())
    text = re.sub("[إأآٱ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


_EN_PATTERNS = [
    re.compile(r"\b" + re.escape(normalize_text(w)) + r"(?:s|es|ed|d)?\b")
    for w in EN_CLICHES
]
_AR_PHRASES = [normalize_text(p) for p in AR_CLICHES]


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> str:
    """
    Detect whether ``text`` is Arabic ('ar') or English ('en').

    Uses a script-ratio check first (langdetect often labels Arabic-script text
    as Persian/Urdu), then langdetect.

    Raises:
        ValueError: if the text is empty or neither Arabic nor English.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        raise ValueError("Input text contains no alphabetic characters.")

    arabic = sum(1 for c in letters if "\u0600" <= c <=
                 "\u06FF" or "\u0750" <= c <= "\u077F")
    latin = sum(1 for c in letters if c.isascii())
    arabic_ratio, latin_ratio = arabic / len(letters), latin / len(letters)

    if arabic_ratio >= 0.6:
        return "ar"

    try:
        top = detect_langs(text)[0].lang
    except (LangDetectException, IndexError):
        top = None

    if top == "ar" or (top in {"fa", "ur"} and arabic_ratio >= 0.3):
        return "ar"
    if top == "en" or latin_ratio >= 0.8:  # short English snippets are often misdetected
        return "en"
    raise ValueError(
        f"Unsupported language (detected: {top}). Only Arabic and English are supported.")


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
# @dataclass
# class TextMetrics:


class TextMetrics(BaseModel):
    """Quality metrics for a piece of text."""
    words: int
    sentences: int
    burstiness: float
    trigger_hits: int
    trigger_density: float  # AI trigger terms per 100 words
    risk_score: float       # 0-100
    triggers_found: list[str] = Field(default_factory=list)
# class TextMetrics(BaseModel):
#     """Quality metrics for a piece of text."""
#     words: int
#     sentences: int
#     burstiness: float
#     trigger_hits: int
#     trigger_density: float  # AI trigger terms per 100 words
#     risk_score: float       # 0-100
#     triggers_found: list[str] = field(default_factory=list)


def count_words(text: str) -> int:
    """Count words in a Unicode-aware way."""
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (supports Latin and Arabic punctuation)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and count_words(s) > 0]


def burstiness_index(text: str) -> float:
    """Burstiness Index = population std-dev of sentence lengths / mean length (0 if < 2 sentences)."""
    lengths = [count_words(s) for s in split_sentences(text)]
    if len(lengths) < 2:
        return 0.0
    mean = statistics.fmean(lengths)
    return statistics.pstdev(lengths) / mean if mean > 0 else 0.0


def find_ai_triggers(text: str, lang: str) -> list[str]:
    """Return every AI-cliché occurrence (with repeats) found in ``text``."""
    norm = normalize_text(text)
    found: list[str] = []
    if lang == "ar":
        for original, phrase in zip(AR_CLICHES, _AR_PHRASES):
            found.extend([original] * norm.count(phrase))
    else:
        for original, pattern in zip(EN_CLICHES, _EN_PATTERNS):
            found.extend([original] * len(pattern.findall(norm)))
    return found


def ai_risk_score(density_per_100_words: float) -> float:
    """
    Heuristic AI Detection Risk (0-100) from trigger density, using a saturating
    curve: ~4 triggers/100 words is about 63, ~8 is about 86. Not a real detector.
    """
    return round(100.0 * (1.0 - math.exp(-density_per_100_words / 4.0)), 1)


def risk_label(score: float) -> str:
    """Human-readable label for a risk score."""
    if score < 20:
        return "Low"
    if score < 45:
        return "Moderate"
    if score < 70:
        return "High"
    return "Very High"


def analyze(text: str, lang: str) -> TextMetrics:
    """Compute all analytics for ``text``."""
    words = count_words(text)
    found = find_ai_triggers(text, lang)
    density = (len(found) / words * 100.0) if words else 0.0
    return TextMetrics(
        words=words,
        sentences=len(split_sentences(text)),
        burstiness=round(burstiness_index(text), 3),
        trigger_hits=len(found),
        trigger_density=round(density, 2),
        risk_score=ai_risk_score(density),
        triggers_found=sorted(set(found)),
    )


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def get_system_prompt(lang: str) -> str:
    """Return the language-specific system prompt (Arabic or English)."""
    if lang == "ar":
        cliches = "، ".join(AR_CLICHES[:18])
        return (
            "أنت محرر عربي محترف وكاتب محتوى بارع، مهمتك إعادة صياغة النص بحيث يبدو "
            "كأن كاتبًا بشريًا كتبه بنفسه، لا نموذج ذكاء اصطناعي.\n\n"
            "القواعد:\n"
            "1. حافظ على المعنى والحقائق والأرقام والأسماء وترتيب الأفكار، ولا تضف "
            "معلومات أو أمثلة أو آراء جديدة.\n"
            "2. زد التباين في أطوال الجمل: امزج بين جمل قصيرة جدًا (٣–٧ كلمات) وجمل "
            "متوسطة (١٠–١٨ كلمة) وجمل طويلة مركبة (٢٥–٤٠ كلمة)، ولا تكرر أكثر من "
            "جملتين متقاربتين في الطول على التوالي.\n"
            f"3. احذف العبارات المستهلكة والقوالب الجاهزة التي تكثر في النصوص المولّدة "
            f"آليًا، مثل: {cliches}. استبدلها بتعبير مباشر أو احذفها.\n"
            "4. اكتب بنبرة طبيعية: عربية فصيحة سلسة قريبة من لغة الكاتب المعاصر، بلا "
            "تكلف ولا سجع متصنع ولا ترجمة حرفية عن الإنجليزية. نوّع بدايات الجمل "
            "(لا تبدأ كل جملة بالفعل نفسه أو بحرف العطف نفسه)، وفضّل الأفعال الملموسة، "
            "وتجنب القوائم الثلاثية المتكررة والخاتمة التلخيصية المصطنعة.\n"
            "5. حافظ على مستوى الرسمية في النص الأصلي.\n"
            "6. يجب أن تظهر الكلمات المفتاحية المطلوبة حرفيًا كما هي، دون تغيير أو ترجمة.\n"
            "7. اكتب بالعربية، ولا تستخدم الإنجليزية إلا للمصطلحات الواردة في الأصل.\n"
            "8. أخرج النص المعاد صياغته فقط، دون مقدمة أو شرح أو علامات اقتباس أو "
            "تنسيق ماركداون، مع الحفاظ على الفقرات."
        )

    cliches = ", ".join(EN_CLICHES[:30])
    return (
        "You are an expert human editor and ghostwriter. Rewrite the user's text so it "
        "reads as if a thoughtful person wrote it, not an AI model.\n\n"
        "Rules:\n"
        "1. Preserve meaning, facts, figures, names and the order of ideas. Never add "
        "new facts, opinions or invented examples.\n"
        "2. Increase burstiness: mix very short sentences (3-7 words) with medium ones "
        "(10-18 words) and a few long, flowing ones (25-40 words). Never use more than "
        "two similarly sized sentences in a row.\n"
        f"3. Remove AI clichés and filler such as: {cliches}. Replace them with plain, "
        "specific wording or cut them.\n"
        "4. Sound natural: use contractions where the register allows, prefer active "
        "voice and concrete verbs, vary sentence openers, and allow an occasional "
        "fragment or rhetorical question. Avoid reflexive triplets, 'not only... but "
        "also' constructions, em-dash overuse and tidy summarizing conclusions.\n"
        "5. Keep the register of the original (formal stays formal).\n"
        "6. Required keywords must appear verbatim, exactly as given.\n"
        "7. Write in English only.\n"
        "8. Output ONLY the rewritten text: no preamble, no quotation marks, no "
        "markdown, no commentary. Keep paragraph breaks."
    )


def build_user_prompt(lang: str, text: str, min_w: int, max_w: int, target: int, kws: list[str]) -> str:
    """Build the first-attempt user message with constraints."""
    if lang == "ar":
        kw = "، ".join(kws) if kws else "لا يوجد"
        return (
            f"أعد صياغة النص التالي.\n"
            f"- عدد الكلمات المطلوب: بين {min_w} و{max_w} كلمة (والأفضل قرابة {target}).\n"
            f"- الكلمات المفتاحية الواجب إبقاؤها حرفيًا: {kw}\n\n"
            f"النص:\n{text}"
        )
    kw = ", ".join(kws) if kws else "none"
    return (
        f"Rewrite the following text.\n"
        f"- Required length: between {min_w} and {max_w} words (aim for about {target}).\n"
        f"- Keywords to keep verbatim: {kw}\n\n"
        f"Text:\n{text}"
    )


def build_feedback_prompt(lang: str, wc: int, min_w: int, max_w: int, missing: list[str]) -> str:
    """Build the corrective message used on retries."""
    issues_en, issues_ar = [], []
    if wc < min_w:
        issues_en.append(
            f"it has {wc} words, which is too short (minimum {min_w})")
        issues_ar.append(f"عدد كلماته {wc} وهو أقل من الحد الأدنى ({min_w})")
    elif wc > max_w:
        issues_en.append(
            f"it has {wc} words, which is too long (maximum {max_w})")
        issues_ar.append(f"عدد كلماته {wc} وهو أكثر من الحد الأقصى ({max_w})")
    if missing:
        issues_en.append("it is missing these keywords: " + ", ".join(missing))
        issues_ar.append("تنقصه هذه الكلمات المفتاحية: " + "، ".join(missing))

    if lang == "ar":
        return (
            "النص السابق مرفوض لأن " + " و".join(issues_ar) + ". "
            f"أعد كتابته بحيث يتراوح بين {min_w} و{max_w} كلمة مع إبقاء الكلمات "
            "المفتاحية حرفيًا، والتزم بكل القواعد السابقة. أخرج النص فقط."
        )
    return (
        "Your previous version was rejected because " +
        " and ".join(issues_en) + ". "
        f"Rewrite it so it is between {min_w} and {max_w} words, keeps every required "
        "keyword verbatim, and still follows all earlier rules. Output the text only."
    )


def clean_llm_output(raw: str) -> str:
    """Strip code fences, chatty preambles and wrapping quotes from model output."""
    text = raw.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    lines = text.split("\n")
    if len(lines) > 1 and lines[0].rstrip().endswith(":") and len(lines[0]) < 120:
        text = "\n".join(lines[1:]).strip()
    if len(text) > 1 and text[0] in "\"'“«" and text[-1] in "\"'”»":
        text = text[1:-1].strip()
    return text


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def missing_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return keywords not present in ``text`` (case/diacritic-insensitive)."""
    norm = normalize_text(text)
    return [k for k in keywords if normalize_text(k) not in norm]


def validate_candidate(text: str, min_w: int, max_w: int, kws: list[str]) -> tuple[bool, int, list[str], int]:
    """
    Check a candidate against the guardrails.

    Returns:
        (is_valid, word_count, missing_keywords, penalty) where a lower penalty
        means a closer match (used to pick the best fallback candidate).
    """
    wc = count_words(text)
    missing = missing_keywords(text, kws)
    distance = max(min_w - wc, wc - max_w, 0)
    return (distance == 0 and not missing), wc, missing, distance + 1000 * len(missing)


# --------------------------------------------------------------------------- #
# Lazy singletons: LLM, embedder, Redis
# --------------------------------------------------------------------------- #
class _Services:
    """Thread-safe lazy holder for the embedder, Redis client and LLM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedder: Optional[SentenceTransformer] = None
        self._embedder_failed = False
        self._redis: Optional[redis.Redis] = None
        self._redis_retry_at = 0.0
        self._llm: Optional[ChatGroq] = None

    def embedder(self) -> Optional[SentenceTransformer]:
        """Return the embedding model, or None if it cannot be loaded."""
        with self._lock:
            if self._embedder is None and not self._embedder_failed:
                try:
                    self._embedder = SentenceTransformer(EMBED_MODEL_NAME)
                    logger.info("Loaded embedding model %s", EMBED_MODEL_NAME)
                except Exception as exc:  # noqa: BLE001
                    self._embedder_failed = True
                    logger.warning(
                        "Embedder unavailable, semantic cache disabled: %s", exc)
            return self._embedder

    def redis(self) -> Optional[redis.Redis]:
        """Return a connected Redis client, or None (retries every 30 s)."""
        with self._lock:
            if self._redis is not None:
                return self._redis
            if time.monotonic() < self._redis_retry_at:
                return None
            try:
                client = redis.Redis.from_url(
                    REDIS_URL, socket_connect_timeout=2, socket_timeout=3, decode_responses=False
                )
                client.ping()
                self._redis = client
                logger.info("Connected to Redis at %s", REDIS_URL)
            except Exception as exc:  # noqa: BLE001
                self._redis_retry_at = time.monotonic() + 30
                logger.warning("Redis offline, caching disabled: %s", exc)
            return self._redis

    def drop_redis(self) -> None:
        """Mark the Redis connection as broken so it is re-established later."""
        with self._lock:
            self._redis = None
            self._redis_retry_at = time.monotonic() + 30

    def llm(self) -> ChatGroq:
        """Return the Groq chat model. Raises RuntimeError if GROQ_API_KEY is missing."""
        with self._lock:
            if self._llm is None:
                key = os.getenv("GROQ_API_KEY")
                if not key:
                    raise RuntimeError(
                        "GROQ_API_KEY environment variable is not set.")
                self._llm = ChatGroq(
                    model=GROQ_MODEL, temperature=0.8, max_retries=2, api_key=key)
            return self._llm


SERVICES = _Services()


# --------------------------------------------------------------------------- #
# Semantic cache (plain Redis hashes + NumPy cosine search; no RediSearch needed)
# --------------------------------------------------------------------------- #
# @dataclass

class CacheHit(BaseModel):
    """A semantic-cache hit."""
    output: str
    similarity: float


def embed_text(text: str) -> Optional[np.ndarray]:
    """Return a unit-normalized float32 embedding, or None if the embedder is unavailable."""
    model = SERVICES.embedder()
    if model is None:
        return None
    try:
        vec = model.encode(text, normalize_embeddings=True,
                           convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding failed: %s", exc)
        return None


def _cache_key(text: str, lang: str, min_w: int, max_w: int, kws: list[str]) -> str:
    raw = f"{lang}\0{min_w}\0{max_w}\0{'|'.join(sorted(kws))}\0{text}"
    return CACHE_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_lookup(emb: np.ndarray, lang: str, min_w: int, max_w: int, kws: list[str]) -> Optional[CacheHit]:
    """
    Find the most similar cached entry with similarity >= SIMILARITY_THRESHOLD whose
    stored output also satisfies the *current* word-range and keyword constraints.
    """
    client = SERVICES.redis()
    if client is None:
        return None
    try:
        keys = list(client.scan_iter(match=CACHE_PREFIX + "*", count=500))
        scored: list[tuple[float, bytes]] = []
        for i in range(0, len(keys), SCAN_BATCH):
            batch = keys[i:i + SCAN_BATCH]
            pipe = client.pipeline(transaction=False)
            for k in batch:
                pipe.hmget(k, "emb", "lang")
            for k, (raw_emb, raw_lang) in zip(batch, pipe.execute()):
                if raw_emb is None or raw_lang is None or raw_lang.decode() != lang:
                    continue
                cached = np.frombuffer(raw_emb, dtype=np.float32)
                if cached.shape != emb.shape:
                    continue
                # both unit-normalized -> cosine similarity
                sim = float(np.dot(emb, cached))
                if sim >= SIMILARITY_THRESHOLD:
                    scored.append((sim, k))

        for sim, key in sorted(scored, key=lambda t: t[0], reverse=True):
            raw_out = client.hget(key, "output")
            if raw_out is None:
                continue
            output = raw_out.decode("utf-8")
            ok, *_ = validate_candidate(output, min_w, max_w, kws)
            if ok:
                return CacheHit(output=output, similarity=sim)
        return None
    except redis.RedisError as exc:
        logger.warning(
            "Cache lookup failed, continuing without cache: %s", exc)
        SERVICES.drop_redis()
        return None


def cache_store(emb: np.ndarray, text: str, output: str, lang: str, min_w: int, max_w: int, kws: list[str]) -> None:
    """Persist a validated result in Redis (best effort)."""
    client = SERVICES.redis()
    if client is None:
        return
    try:
        key = _cache_key(text, lang, min_w, max_w, kws)
        pipe = client.pipeline(transaction=False)
        pipe.hset(key, mapping={"emb": emb.tobytes(),
                  "lang": lang, "input": text, "output": output})
        pipe.expire(key, CACHE_TTL_SECONDS)
        pipe.execute()
    except redis.RedisError as exc:
        logger.warning("Cache store failed: %s", exc)
        SERVICES.drop_redis()


# --------------------------------------------------------------------------- #
# Generation with guardrail loop
# --------------------------------------------------------------------------- #
# @dataclass
class GenerationResult(BaseModel):
    """Outcome of the guardrail loop."""
    text: str
    attempts: int
    valid: bool
    word_count: int
    missing: list[str]


async def generate_with_guardrails(
    text: str, lang: str, min_w: int, max_w: int, kws: list[str]
) -> GenerationResult:
    """
    Rewrite ``text`` and validate it. Retries up to MAX_RETRIES times when the word
    count is outside [min_w, max_w] or required keywords are missing, feeding the
    failure reason back to the model. If every attempt fails, the closest candidate
    is returned with ``valid=False``.

    Raises:
        RuntimeError: if the LLM produced no usable output at all.
    """
    llm = SERVICES.llm()
    input_words = count_words(text)
    target = min(max(input_words, min_w), max_w)

    messages: list[BaseMessage] = [
        SystemMessage(content=get_system_prompt(lang)),
        HumanMessage(content=build_user_prompt(
            lang, text, min_w, max_w, target, kws)),
    ]

    best: Optional[GenerationResult] = None
    best_penalty = math.inf
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 2):  # first try + MAX_RETRIES retries
        try:
            response = await llm.ainvoke(messages)
            content = response.content
            raw = content if isinstance(content, str) else "".join(
                p if isinstance(p, str) else p.get("text", "") for p in content
            )
            candidate = clean_llm_output(raw)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("LLM call failed on attempt %d: %s", attempt, exc)
            continue

        if not candidate:
            logger.warning("Empty LLM output on attempt %d", attempt)
            continue

        ok, wc, missing, penalty = validate_candidate(
            candidate, min_w, max_w, kws)
        logger.info("Attempt %d: %d words, missing=%s, valid=%s",
                    attempt, wc, missing, ok)

        if penalty < best_penalty:
            best_penalty = penalty
            best = GenerationResult(
                text=candidate,
                attempts=attempt,
                valid=ok,
                word_count=wc,
                missing=missing,
            )
        if ok:
            return GenerationResult(
                text=candidate,
                attempts=attempt,
                valid=True,
                word_count=wc,
                missing=[],
            )

        messages.append(AIMessage(content=candidate))
        messages.append(HumanMessage(
            content=build_feedback_prompt(lang, wc, min_w, max_w, missing)))

    if best is None:
        raise RuntimeError(
            f"The language model returned no usable output ({last_error}).")
    best.attempts = MAX_RETRIES + 1
    return best


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def render_markdown(
    output: str,
    lang: str,
    before: TextMetrics,
    after: TextMetrics,
    keywords: list[str],
    min_w: int,
    max_w: int,
    cache_status: str,
    attempts: int,
    valid: bool,
) -> str:
    """Produce the final markdown report."""
    def fmt(m: TextMetrics) -> str:
        return f"{m.risk_score}/100 ({risk_label(m.risk_score)})"

    lines = ["## Humanized Text", "", output, "", "---", "", "## Analytics", "",
             "| Metric | Before | After |", "|---|---|---|",
             f"| Word count | {before.words} | {after.words} |",
             f"| Sentences | {before.sentences} | {after.sentences} |",
             f"| Burstiness Index | {before.burstiness} | {after.burstiness} |",
             f"| AI trigger terms | {before.trigger_hits} | {after.trigger_hits} |",
             f"| Trigger density (per 100 words) | {before.trigger_density} | {after.trigger_density} |",
             f"| AI Detection Risk Score | {fmt(before)} | {fmt(after)} |",
             "",
             f"- **Language:** {LANG_NAMES[lang]}",
             f"- **Target range:** {min_w}-{max_w} words "
             f"({'within range' if min_w <= after.words <= max_w else 'OUT OF RANGE'})",
             f"- **Source:** {cache_status}",
             f"- **Generation attempts:** {attempts}"]

    if before.triggers_found:
        lines.append(
            f"- **AI clichés in original:** {', '.join(before.triggers_found)}")
    if after.triggers_found:
        lines.append(
            f"- **AI clichés remaining:** {', '.join(after.triggers_found)}")

    lines += ["", "## Keyword Preservation", ""]
    if keywords:
        missing = set(missing_keywords(output, keywords))
        lines += [f"- {'❌ MISSING' if k in missing else '✅ preserved'}: `{k}`" for k in keywords]
        if not missing:
            lines += ["", "All required keywords were preserved."]
    else:
        lines.append("No keywords were specified.")

    if not valid:
        lines += ["", "> ⚠️ **Warning:** guardrails could not be fully satisfied after "
                      f"{attempts} attempts. This is the closest result; it was not cached."]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("humanizer")


@mcp.tool()
async def humanize_text(
    text: str,
    min_words: int = 50,
    max_words: int = 500,
    keywords_to_preserve: list[str] = Field(default_factory=list),
) -> str:
    """
    Rewrite AI-sounding Arabic or English text so it reads as naturally human.

    The language is auto-detected. The rewrite increases sentence-length variety
    (burstiness), removes AI cliché phrases, and keeps the original meaning. Output
    is validated against a word range and required keywords, with up to 3 retries.
    Near-duplicate inputs (cosine similarity >= 0.92) are served from a Redis cache.

    Args:
        text: The text to humanize (Arabic or English).
        min_words: Minimum word count of the result (default 50).
        max_words: Maximum word count of the result (default 500).
        keywords_to_preserve: Words/phrases that must appear verbatim in the result.

    Returns:
        A markdown report with the humanized text, before/after analytics
        (Burstiness Index, AI Detection Risk Score) and keyword confirmation.
    """
    keywords = [k.strip()
                for k in list(keywords_to_preserve) if k and k.strip()]

    # ---- Input validation ------------------------------------------------- #
    if not text or not text.strip():
        return "**Error:** `text` must not be empty."
    if min_words < 1 or max_words < 1:
        return "**Error:** `min_words` and `max_words` must be positive integers."
    if min_words > max_words:
        return f"**Error:** `min_words` ({min_words}) cannot exceed `max_words` ({max_words})."
    text = text.strip()

    try:
        lang = detect_language(text)
    except ValueError as exc:
        return f"**Error:** {exc}"

    before = analyze(text, lang)

    # ---- Semantic cache lookup ------------------------------------------- #
    embedding = await asyncio.to_thread(embed_text, text)
    if embedding is not None:
        hit = await asyncio.to_thread(cache_lookup, embedding, lang, min_words, max_words, keywords)
        if hit is not None:
            logger.info("Semantic cache hit (similarity %.3f)", hit.similarity)
            return render_markdown(
                hit.output, lang, before, analyze(hit.output, lang), keywords,
                min_words, max_words, f"Semantic cache hit (similarity {hit.similarity:.3f})",
                attempts=0, valid=True,
            )

    # ---- Generation with guardrails -------------------------------------- #
    try:
        result = await generate_with_guardrails(text, lang, min_words, max_words, keywords)
    except RuntimeError as exc:
        logger.error("Generation failed: %s", exc)
        return f"**Error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected generation failure")
        return f"**Error:** unexpected failure during generation: {exc}"

    # ---- Cache only fully validated results ------------------------------ #
    if result.valid and embedding is not None:
        await asyncio.to_thread(cache_store, embedding, text, result.text, lang, min_words, max_words, keywords)

    return render_markdown(
        result.text, lang, before, analyze(result.text, lang), keywords,
        min_words, max_words, "Freshly generated", result.attempts, result.valid,
    )


def main() -> None:
    # Warm up heavy resources so the first tool call is fast; failures are non-fatal.
    SERVICES.embedder()
    SERVICES.redis()
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    logger.info(
        "Starting Humanizer MCP server (transport=%s, model=%s)", transport, GROQ_MODEL)
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
