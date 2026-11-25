import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from spending_agent import draft as contract
from spending_agent.backend import ChatBackend, without_bounds


class RefusesReasoningEffort(BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *arguments):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        if "reasoning_effort" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "reasoning_effort: unexpected value none"}')
            return
        reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(reply)


class RefusesSchemas(BaseHTTPRequestHandler):
    bodies = []
    refuses_every_schema = False

    def log_message(self, *arguments):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).bodies.append(body)
        response_format = body.get("response_format") or {}
        schema = json.dumps(response_format.get("json_schema", {}))
        if response_format.get("type") == "json_schema" and (self.refuses_every_schema or "pattern" in schema):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "unsupported keyword: pattern"}')
            return
        reply = json.dumps({"choices": [{"message": {"role": "assistant", "content": "{}"}}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(reply)


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class BackendTests(unittest.TestCase):
    def test_a_refused_bounded_schema_is_retried_with_the_same_fields_unbounded(self):
        RefusesSchemas.bodies, RefusesSchemas.refuses_every_schema = [], False
        server = serve(RefusesSchemas)
        try:
            backend = ChatBackend(f"http://127.0.0.1:{server.server_port}/v1", "m")
            schema = contract.answer_schema()
            backend.answer([{"role": "user", "content": "the receipt"}], schema)
            self.assertEqual(backend.last_format, "json_schema_plain")
            self.assertEqual(len(backend.last_refusals), 1)
            self.assertIn("unsupported keyword", backend.last_refusals[0])
            sent = RefusesSchemas.bodies[-1]["response_format"]["json_schema"]["schema"]
            self.assertEqual(list(sent["properties"]), list(schema["properties"]))
            self.assertNotIn("pattern", json.dumps(sent))
            self.assertEqual(RefusesSchemas.bodies[-1]["messages"][-1]["content"], "the receipt")
        finally:
            server.shutdown()

    def test_without_a_schema_the_prompt_names_every_field_in_order(self):
        RefusesSchemas.bodies, RefusesSchemas.refuses_every_schema = [], True
        server = serve(RefusesSchemas)
        try:
            backend = ChatBackend(f"http://127.0.0.1:{server.server_port}/v1", "m")
            schema = contract.answer_schema()
            messages = [{"role": "user", "content": "the receipt"}]
            backend.answer(messages, schema)
            self.assertEqual(backend.last_format, "json_object")
            self.assertEqual(len(backend.last_refusals), 2)
            prompt = RefusesSchemas.bodies[-1]["messages"][-1]["content"]
            self.assertTrue(prompt.startswith("the receipt"))
            positions = [prompt.index(f"- {name} (") for name in schema["properties"]]
            self.assertIn("- totalAmount (number or string):", prompt)
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(messages[-1]["content"], "the receipt")
        finally:
            RefusesSchemas.refuses_every_schema = False
            server.shutdown()

    def test_unbounding_keeps_types_and_descriptions(self):
        schema = without_bounds(contract.answer_schema())
        self.assertEqual(schema["properties"]["tags"]["items"], {"type": "string"})
        self.assertEqual(schema["properties"]["currencyCode"]["type"], "string")
        self.assertEqual(schema["properties"]["totalAmount"]["anyOf"], [{"type": "number"}, {"type": "string"}])
        self.assertIn("description", schema["properties"]["totalAmount"])
        self.assertFalse(schema["additionalProperties"])

    def test_a_refused_reasoning_effort_is_dropped_and_thinking_stays_off(self):
        server = HTTPServer(("127.0.0.1", 0), RefusesReasoningEffort)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            backend = ChatBackend(f"http://127.0.0.1:{server.server_port}/v1", "m", reasoning_effort="none")
            message, _ = backend.complete([{"role": "user", "content": "hi"}])
            self.assertEqual(message["content"], "ok")
            backend.complete([{"role": "user", "content": "again"}])
            last = RefusesReasoningEffort.bodies[-1]
            self.assertNotIn("reasoning_effort", last)
            self.assertEqual(last["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(len(RefusesReasoningEffort.bodies), 3)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
