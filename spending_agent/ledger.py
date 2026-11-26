import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

RECORD_FIELDS = ["id", "date", "time", "merchant", "payment_instrument", "amount_minor", "currency",
                 "category", "tags", "note", "source", "content", "created_at"]
CATEGORY_FIELDS = ["title"]
TAG_FIELDS = ["name", "note"]
REVIEW_FIELDS = ["session", "created_at", "status", "reason", "is_spending", "merchant", "payment_instrument",
                 "amount_minor", "currency", "date", "time", "category", "proposed_category", "tags",
                 "proposed_tag", "note", "confidence", "reasoning", "content"]
MEMORY_HISTORY_FIELDS = ["written_at", "session", "text"]

SOURCES = ("manual", "agent", "agent_reviewed")
USER_CONFIRMED_SOURCES = ("manual", "agent_reviewed")
TAG_SEPARATOR = ";"


def now_stamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Record:
    id: str
    date: str
    amount_minor: int
    currency: str
    merchant: str = ""
    time: str = ""
    payment_instrument: str = ""
    category: str = ""
    tags: list = field(default_factory=list)
    note: str = ""
    source: str = "manual"
    content: str = ""
    created_at: str = ""

    @property
    def is_user_confirmed(self):
        return self.source in USER_CONFIRMED_SOURCES

    def to_row(self):
        row = asdict(self)
        row["tags"] = TAG_SEPARATOR.join(self.tags)
        return row

    @classmethod
    def from_row(cls, row):
        return cls(
            id=row.get("id", ""),
            date=row.get("date", ""),
            time=row.get("time", ""),
            merchant=row.get("merchant", ""),
            payment_instrument=row.get("payment_instrument", ""),
            amount_minor=int(row.get("amount_minor") or 0),
            currency=(row.get("currency") or "").upper(),
            category=row.get("category", ""),
            tags=[t.strip() for t in (row.get("tags") or "").split(TAG_SEPARATOR) if t.strip()],
            note=row.get("note", ""),
            source=row.get("source") or "manual",
            content=row.get("content", ""),
            created_at=row.get("created_at", ""),
        )


@dataclass
class Snapshot:
    records: list
    categories: list
    tags: list


def _read(path, fields):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{name: row.get(name) or "" for name in fields} for row in csv.DictReader(f)]


def _write_header(path, fields):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _append(path, fields, row):
    _write_header(path, fields)
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields, extrasaction="ignore").writerow(row)


def _unique(names):
    seen, out = set(), []
    for name in names:
        folded = name.strip().casefold()
        if name.strip() and folded not in seen:
            seen.add(folded)
            out.append(name.strip())
    return out


class Ledger:
    def __init__(self, folder):
        self.folder = folder

    def path(self, name):
        return os.path.join(self.folder, name)

    @classmethod
    def create(cls, folder, categories=(), tags=()):
        os.makedirs(os.path.join(folder, "sessions"), exist_ok=True)
        ledger = cls(folder)
        for name, fields in (("records.csv", RECORD_FIELDS), ("categories.csv", CATEGORY_FIELDS),
                             ("tags.csv", TAG_FIELDS), ("review.csv", REVIEW_FIELDS),
                             ("memory_history.csv", MEMORY_HISTORY_FIELDS)):
            _write_header(ledger.path(name), fields)
        for title in _unique(categories):
            if title.casefold() not in {c.casefold() for c in ledger.category_titles()}:
                _append(ledger.path("categories.csv"), CATEGORY_FIELDS, {"title": title})
        for name in _unique(tags):
            if name.casefold() not in {t.casefold() for t in ledger.tag_names()}:
                _append(ledger.path("tags.csv"), TAG_FIELDS, {"name": name, "note": ""})
        if not os.path.exists(ledger.path("memory.txt")):
            with open(ledger.path("memory.txt"), "w", encoding="utf-8") as f:
                f.write("")
        return ledger

    def records(self):
        return [Record.from_row(row) for row in _read(self.path("records.csv"), RECORD_FIELDS)]

    def category_titles(self):
        listed = [row["title"] for row in _read(self.path("categories.csv"), CATEGORY_FIELDS)]
        return _unique(listed)

    def tag_names(self):
        listed = [row["name"] for row in _read(self.path("tags.csv"), TAG_FIELDS)]
        used = [tag for record in self.records() for tag in record.tags]
        return _unique(listed + used)

    def snapshot(self):
        records = self.records()
        categories = _unique(self.category_titles() + [r.category for r in records if r.category])
        return Snapshot(records=records, categories=categories, tags=self.tag_names())

    def append_record(self, record):
        if not record.created_at:
            record.created_at = now_stamp()
        _append(self.path("records.csv"), RECORD_FIELDS, record.to_row())

    def add_to_review(self, row):
        _append(self.path("review.csv"), REVIEW_FIELDS, {"created_at": now_stamp(), **row})

    def review(self):
        return _read(self.path("review.csv"), REVIEW_FIELDS)

    def memory(self):
        path = self.path("memory.txt")
        if not os.path.exists(path):
            return ""
        with open(path, encoding="utf-8") as f:
            return f.read()

    def write_memory(self, text, session):
        with open(self.path("memory.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        _append(self.path("memory_history.csv"), MEMORY_HISTORY_FIELDS,
                {"written_at": now_stamp(), "session": session, "text": text})

    def memory_history(self):
        return _read(self.path("memory_history.csv"), MEMORY_HISTORY_FIELDS)

    def save_session(self, session_id, transcript):
        os.makedirs(self.path("sessions"), exist_ok=True)
        with open(self.path(os.path.join("sessions", f"{session_id}.json")), "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=1)
