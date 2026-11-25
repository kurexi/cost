"""The one fuzzy matcher every query tool uses.

Text is folded the way the app folds OCR text (typographic quotes and dashes,
width, accents), then compared on letters and digits only. A candidate scores
by the first rule that holds, strongest first: the same name, one name a
prefix of the other (the printed "TESCO STORES 3297" and the recorded
"Tesco"), shared words, a near-miss spelling. A short name ("M&S", "BP")
matches only as a whole word, never as part of one."""

import re
import unicodedata

SHORT_NAME_LENGTH = 3
MINIMUM_WORD_LENGTH = 3
MATCH_THRESHOLD = 0.6

EXACT = 1.0
SAME_WORDS = 0.95
WHOLE_SHORT_NAME = 0.9
PREFIX = 0.85
SHARED_WORDS_FLOOR = 0.5
SHARED_WORDS_SPAN = 0.3
NEAR_SPELLING = 0.6

_TYPOGRAPHY = [("‘", "'"), ("’", "'"), ("‛", "'"), ("′", "'"), ("´", "'"),
               ("“", '"'), ("”", '"'), ("„", '"'),
               ("–", "-"), ("—", "-"), ("−", "-"), (" ", " ")]


def folded(text):
    text = text or ""
    for typographic, plain in _TYPOGRAPHY:
        text = text.replace(typographic, plain)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return text.lower()


def key(text):
    return "".join(c for c in folded(text) if c.isalnum())


def words(text):
    return [word for word in re.split(r"[^\w]+", folded(text)) if word]


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def singular(word):
    """A light English singular, enough that "grocery" finds "Groceries"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _word_keys(text):
    return [singular(k) for k in (key(word) for word in words(text)) if k]


def _spaced_word_keys(text):
    return [k for k in (key(word) for word in folded(text).split()) if k]


def _shared_words_score(pattern, candidate):
    pattern_words = [w for w in _word_keys(pattern) if len(w) >= MINIMUM_WORD_LENGTH]
    candidate_words = [w for w in _word_keys(candidate) if len(w) >= MINIMUM_WORD_LENGTH]
    if not pattern_words or not candidate_words:
        return 0.0
    shared = sum(1 for p in pattern_words if any(c.startswith(p) or p.startswith(c) for c in candidate_words))
    if not shared:
        return 0.0
    return SHARED_WORDS_FLOOR + SHARED_WORDS_SPAN * shared / len(pattern_words)


def score(pattern, candidate):
    p, c = key(pattern), key(candidate)
    if not p or not c:
        return 0.0
    if p == c:
        return EXACT
    shorter, longer = (pattern, candidate) if len(p) <= len(c) else (candidate, pattern)
    if len(key(shorter)) <= SHORT_NAME_LENGTH:
        return WHOLE_SHORT_NAME if key(shorter) in _spaced_word_keys(longer) else 0.0
    if "".join(_word_keys(pattern)) == "".join(_word_keys(candidate)):
        return SAME_WORDS
    if c.startswith(p) or p.startswith(c):
        return PREFIX
    best = _shared_words_score(pattern, candidate)
    if edit_distance(p, c) <= max(1, len(p) // 5):
        best = max(best, NEAR_SPELLING)
    return best


def matches(pattern, candidate, threshold=MATCH_THRESHOLD):
    return score(pattern, candidate) >= threshold


def rank(pattern, candidates, limit=None, threshold=MATCH_THRESHOLD):
    """Candidates scoring at least `threshold`, best first, ties by name."""
    scored = [(candidate, score(pattern, candidate)) for candidate in candidates]
    kept = sorted(((c, s) for c, s in scored if s >= threshold), key=lambda pair: (-pair[1], key(pair[0])))
    return kept[:limit] if limit else kept
