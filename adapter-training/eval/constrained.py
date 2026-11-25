"""Schema-constrained decoding for the transformers backend.

The app never lets a local model choose the shape of its answer: the final
request carries `LocalModelAgent.extractionSchema` as a `json_schema`
response format, which servers compile into a grammar. Without that, a small
model invents its own keys ("amount": 1111, "spending": {...}) and a grading
run would measure the missing grammar, not the model.

`ExtractionGrammar` is the same constraint as a transformers
`LogitsProcessor`, for this one flat schema: the object's keys, punctuation
and order are forced (`{"reasoning": "`, `", "isSpending": `, ...), and the
model only ever chooses the *values* — free text inside strings (a closing
quote is allowed only where it can't run into the next key), `true`/`false`
for the boolean, digits for the number. Greedy forcing of the literal parts
picks the longest vocabulary token that is a prefix of what must come next.
"""

import json

import torch
from transformers import LogitsProcessor


class ExtractionGrammar(LogitsProcessor):
    def __init__(self, tokenizer, schema, prompt_length, eos_ids):
        self.tok = tokenizer
        self.prompt_length = prompt_length
        self.eos_ids = [e for e in eos_ids if e is not None]
        props = schema["properties"]
        self.fields = [(name, props[name]["type"]) for name in schema["required"]]
        self.literals = []
        for i, (name, _) in enumerate(self.fields):
            lead = "{" if i == 0 else ", "
            self.literals.append(f'{lead}{json.dumps(name)}: ')
        self.closing = "}"
        vocab = tokenizer.get_vocab()
        self.strings = {}
        for _, idx in vocab.items():
            self.strings[idx] = tokenizer.decode([idx])
        self.by_first = {}
        for idx, s in self.strings.items():
            if s:
                self.by_first.setdefault(s[0], []).append(idx)
        self.number_tokens = [i for i, s in self.strings.items() if s and all(c in "0123456789." for c in s)]
        self.quote_tokens = [i for i, s in self.strings.items() if '"' in s]
        self.bool_tokens = [i for i, s in self.strings.items()
                            if s and s.strip() == s and ("true".find(s) >= 0 or "false".find(s) >= 0)]
        self.newline_tokens = [i for i, s in self.strings.items() if "\n" in s or "\r" in s]
        self.vocab_size = len(tokenizer)

    def _state(self, text):
        """Walk `text` (the generated part) against the template. Returns
        ("literal", remaining_literal) | ("string", None) | ("bool", buf) |
        ("number", buf) | ("done", None)."""
        pos = 0
        for i, (_, kind) in enumerate(self.fields):
            self._index = i
            lit = self.literals[i] + ('"' if kind == "string" else "")
            chunk = text[pos:pos + len(lit)]
            if len(chunk) < len(lit):
                return ("literal", lit[len(chunk):]) if lit.startswith(chunk) else ("broken", None)
            if chunk != lit:
                return ("broken", None)
            pos += len(lit)
            if kind == "string":
                j, esc = pos, False
                while j < len(text):
                    c = text[j]
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        break
                    j += 1
                if j >= len(text):
                    return ("string", text[pos:])
                pos = j + 1
            elif kind == "boolean":
                rest = text[pos:]
                for word in ("true", "false"):
                    if rest.startswith(word):
                        pos += len(word)
                        break
                else:
                    return ("bool", rest)
            else:
                j = pos
                while j < len(text) and text[j] in "0123456789.":
                    j += 1
                if j >= len(text):
                    return ("number", text[pos:])
                pos = j
        self._index = len(self.fields) - 1
        rest = text[pos:]
        if rest == "":
            return ("literal", self.closing)
        return ("done", None) if rest.startswith(self.closing) else ("broken", None)

    def _force_literal(self, remaining):
        best, best_len = None, 0
        for idx in self.by_first.get(remaining[0], []):
            s = self.strings[idx]
            if len(s) > best_len and remaining.startswith(s):
                best, best_len = idx, len(s)
        return best

    def _next_literal_after_value(self, field_index):
        if field_index + 1 < len(self.fields):
            nxt = self.literals[field_index + 1] + ('"' if self.fields[field_index + 1][1] == "string" else "")
        else:
            nxt = self.closing
        return nxt

    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[row, self.prompt_length:], skip_special_tokens=False)
            for e in ("<|im_end|>", "<|endoftext|>"):
                if e in text:
                    text = text.split(e)[0]
            self._index = 0
            state, data = self._state(text)
            field_index = self._index
            mask = torch.full_like(scores[row], float("-inf"))
            if state == "literal":
                idx = self._force_literal(data)
                if idx is not None:
                    mask[idx] = 0
                    scores[row] = scores[row] + mask
                continue
            if state in ("done", "broken"):
                for e in self.eos_ids:
                    mask[e] = 0
                scores[row] = scores[row] + mask
                continue
            if state == "string":
                after = self._next_literal_after_value(field_index)
                banned = []
                for idx in self.quote_tokens:
                    s = self.strings[idx]
                    q = _first_unescaped_quote(s, data)
                    if q is None:
                        continue
                    tail = s[q + 1:]
                    if not after.startswith(tail):
                        banned.append(idx)
                if banned:
                    scores[row, banned] = float("-inf")
                if self.newline_tokens and len(data) > 0 and self.fields[field_index][0] != "reasoning":
                    scores[row, self.newline_tokens] = float("-inf")
                for e in self.eos_ids:
                    scores[row, e] = float("-inf")
                continue
            if state == "bool":
                allowed = [i for i in self.bool_tokens if w_prefix(data + self.strings[i])]
                for i in allowed:
                    mask[i] = 0
                scores[row] = scores[row] + mask
                continue
            if state == "number":
                allowed = list(self.number_tokens)
                if data:
                    after = self._next_literal_after_value(field_index)
                    allowed += [i for i in self.by_first.get(after[0], []) if after.startswith(self.strings[i])]
                if len(data) >= 4:
                    allowed = [i for i in allowed if i not in set(self.number_tokens)] or allowed
                for i in allowed:
                    mask[i] = 0
                scores[row] = scores[row] + mask
        return scores


def w_prefix(s):
    return "true".startswith(s) or "false".startswith(s)


def _first_unescaped_quote(token_text, buffer):
    esc = buffer.endswith("\\") and not buffer.endswith("\\\\")
    for k, c in enumerate(token_text):
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            return k
    return None
