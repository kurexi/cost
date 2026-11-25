"""An OpenAI-compatible chat-completions client (llama-server, Ollama, LM
Studio, vLLM), called the way the app's local path calls one: one request at
a time, and the answer asked for as `json_schema`, falling back to
`json_object` and then to no format when a server refuses one."""

import contextlib
import io
import json
import re
import threading
import urllib.error
import urllib.request

_THINK = re.compile(r"<think>.*?</think>", re.S)
_XML_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*(?:</tool_call>|$)", re.S)
_XML_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

_ONE_REQUEST_AT_A_TIME = threading.Lock()


def describe_error(error):
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8", "replace")[:300]
        except Exception:
            body = ""
        return f"HTTP {error.code} {body}".strip()
    return f"{type(error).__name__}: {error}"


def without_thinking(text):
    text = _THINK.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def tool_calls_of(message):
    """The calls a reply asks for, as [{"id", "name", "arguments"}]. Native
    `tool_calls` first; else calls a template left as text in the content."""
    calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function") or {}
        calls.append({"id": call.get("id") or f"call_{index}", "name": function.get("name", ""),
                      "arguments": function.get("arguments", "")})
    if calls:
        return calls
    content = message.get("content") or ""
    for index, (name, body) in enumerate(_XML_CALL.findall(content)):
        arguments = {k.strip(): v for k, v in _XML_PARAMETER.findall(body)}
        calls.append({"id": f"call_{index}", "name": name.strip(), "arguments": arguments})
    if not calls:
        for index, blob in enumerate(_JSON_CALL.findall(content)):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            calls.append({"id": f"call_{index}", "name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
    return calls


def arguments_of(call):
    arguments = call.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


SCHEMA_BOUNDS = ("pattern", "maxLength", "minLength", "maxItems", "minItems", "minimum", "maximum")


def without_bounds(schema):
    """The schema with every length, pattern and range bound removed: the same
    fields in the same order, which is what the grammar most needs to fix."""
    if isinstance(schema, dict):
        return {k: without_bounds(v) for k, v in schema.items() if k not in SCHEMA_BOUNDS}
    if isinstance(schema, list):
        return [without_bounds(v) for v in schema]
    return schema


def field_guide(schema):
    lines = ["Answer with one JSON object with exactly these keys, in this order:"]
    for name, node in schema.get("properties", {}).items():
        kind = node.get("type") or " or ".join(option.get("type", "") for option in node.get("anyOf", []))
        lines.append(f"- {name} ({kind or 'string'}): {node.get('description', '')}".rstrip(": "))
    return "\n".join(lines)


def with_field_guide(messages, schema):
    """The messages with the field list appended to the last user turn, for a
    format whose grammar doesn't name the fields."""
    messages = [dict(m) for m in messages]
    for message in reversed(messages):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content') or ''}\n\n{field_guide(schema)}"
            break
    return messages


def answer_ladder(messages, schema):
    guided = with_field_guide(messages, schema)
    return [("json_schema", {"type": "json_schema",
                             "json_schema": {"name": "record_draft", "strict": True, "schema": schema}}, messages),
            ("json_schema_plain", {"type": "json_schema",
                                   "json_schema": {"name": "record_draft", "strict": True,
                                                   "schema": without_bounds(schema)}}, messages),
            ("json_object", {"type": "json_object"}, guided),
            ("none", None, guided)]


class ChatBackend:
    def __init__(self, base_url, model, api_key=None, timeout=900, temperature=0.0, reasoning_effort=None,
                 one_request_at_a_time=True):
        """`one_request_at_a_time` is the app's rule (on unified memory a second
        generation opens a second KV cache). An evaluation on a GPU server with
        parallel slots turns it off to batch several sessions."""
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.timeout, self.temperature, self.reasoning_effort = timeout, temperature, reasoning_effort
        self.one_request_at_a_time = one_request_at_a_time
        self.sends_reasoning_effort = True
        self.last_format = None
        self.last_refusals = []

    def _post(self, body):
        """A server that refuses the `reasoning_effort` field (some take only
        `chat_template_kwargs`) is asked again without it, and never sent it
        again."""
        try:
            return self._send(body)
        except urllib.error.HTTPError as error:
            text = error.read()
            if error.code == 400 and b"reasoning_effort" in text and "reasoning_effort" in body:
                self.sends_reasoning_effort = False
                return self._send({k: v for k, v in body.items() if k != "reasoning_effort"})
            raise urllib.error.HTTPError(error.url, error.code, error.msg, error.hdrs, io.BytesIO(text))

    def _send(self, body):
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        gate = _ONE_REQUEST_AT_A_TIME if self.one_request_at_a_time else contextlib.nullcontext()
        with gate, urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    @staticmethod
    def _usage(reply):
        usage = dict(reply.get("usage") or {})
        if isinstance(reply.get("timings"), dict):
            usage["timings"] = reply["timings"]
        return usage

    def _body(self, messages, max_tokens):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": max_tokens, "stream": False}
        if self.reasoning_effort:
            if self.sends_reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
            body["chat_template_kwargs"] = {"enable_thinking": self.reasoning_effort != "none"}
        return body

    def complete(self, messages, tools=None, max_tokens=800):
        """One tool round: the reply message, which may ask for tool calls."""
        body = self._body(messages, max_tokens)
        if tools:
            body.update(tools=tools, tool_choice="auto")
        reply = self._post(body)
        return reply["choices"][0]["message"], self._usage(reply)

    def answer(self, messages, schema, max_tokens=1500):
        """The structured answer, down the response-format ladder: the bounded
        schema; the same schema without its bounds (some grammar engines refuse
        patterns or length caps); then `json_object` and no format, where the
        grammar no longer names the fields, so the prompt spells them out."""
        self.last_refusals = []
        for label, response_format, prompt in answer_ladder(messages, schema):
            body = self._body(prompt, max_tokens)
            if response_format:
                body["response_format"] = response_format
            try:
                reply = self._post(body)
            except urllib.error.HTTPError as error:
                if error.code not in (400, 404, 415, 422, 501):
                    raise
                self.last_refusals.append(f"{label}: {describe_error(error)}")
                continue
            self.last_format = label
            return reply["choices"][0]["message"], self._usage(reply)
        raise RuntimeError("the server refused every response format — " + " | ".join(self.last_refusals))
