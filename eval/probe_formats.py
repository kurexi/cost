#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from spending_agent import draft as contract
from spending_agent.backend import SCHEMA_BOUNDS, describe_error, without_bounds

BOUND_GROUPS = {"pattern": ("pattern",), "lengths": ("maxLength", "minLength"), "items": ("maxItems", "minItems"),
                "ranges": ("minimum", "maximum")}


def without(schema, keys):
    if isinstance(schema, dict):
        return {k: without(v, keys) for k, v in schema.items() if k not in keys}
    if isinstance(schema, list):
        return [without(v, keys) for v in schema]
    return schema


def variants(schema):
    yield "full bounded schema", schema
    for name, keys in BOUND_GROUPS.items():
        yield f"without {name}", without(schema, keys)
    yield "no bounds (json_schema_plain)", without_bounds(schema)


def post(base_url, body):
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response.read()
        return "accepted"
    except urllib.error.HTTPError as error:
        return "refused  " + describe_error(error)


def request_body(model):
    return {"model": model, "max_tokens": 8, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": "Answer with an empty draft."}],
            "chat_template_kwargs": {"enable_thinking": False}}


def probe(base_url, model, schema):
    return post(base_url, {**request_body(model),
                           "response_format": {"type": "json_schema",
                                               "json_schema": {"name": "record_draft", "strict": True,
                                                               "schema": schema}}})


def probe_tool_round(base_url, model):
    return post(base_url, {**request_body(model), "tools": contract.TOOL_DEFINITIONS, "tool_choice": "auto"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    assert set(k for keys in BOUND_GROUPS.values() for k in keys) == set(SCHEMA_BOUNDS)
    print(f"{'tool round (tool_choice auto)':32} {probe_tool_round(args.base_url, args.model)}", flush=True)
    for label, schema in variants(contract.answer_schema()):
        print(f"{label:32} {probe(args.base_url, args.model, schema)}", flush=True)


if __name__ == "__main__":
    main()
