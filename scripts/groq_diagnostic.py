"""Temporary Groq diagnostic test — runs on GitHub Actions only.

Never prints, logs, or exposes GROQ_API_KEY. Reports only boolean
presence, HTTP status codes, and sanitized response bodies.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/workspaces/purffle-shorts-curiostick")

import requests
from jsonschema import Draft7Validator

from purffle_shorts.script import SCRIPT_SCHEMA, SCENE_SCHEMA

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"

results: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    results.append(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    key = os.getenv("GROQ_API_KEY")

    # TEST A — Secret availability (boolean only, never print value)
    check("GROQ_API_KEY set", bool(key), "")
    check("GROQ_API_KEY non-empty", bool(key and key.strip()), "")
    if not key or not key.strip():
        for line in results:
            print(line)
        return 1

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    # TEST B — API authentication: minimal completion
    print("=== TEST B: Groq API authentication ===")
    try:
        r = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Reply with exactly the word: OK. Keep your response short."},
                ],
                "temperature": 0.6,
                "max_tokens": 256,
            },
            timeout=60,
        )
        print(f"HTTP status: {r.status_code}")
        body = r.text[:2000]
        print(f"Response body (sanitized): {body!r}")
        check("Groq auth returns 200", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "")
            print(f"Completion content: {content!r}")
            print(f"Finish reason: {finish_reason}")
            check("Groq completion returns content", bool(content and content.strip()), "")
            check("Groq completion not truncated by length", finish_reason != "length", f"finish_reason={finish_reason}")
    except Exception as e:
        print(f"REQUEST ERROR: {type(e).__name__}: {e}")
        check("Groq auth request succeeded", False, f"{type(e).__name__}: {e}")

    # TEST C — Script structured JSON against SCRIPT_SCHEMA
    print("=== TEST C: SCRIPT_SCHEMA validation ===")
    validator = Draft7Validator(SCRIPT_SCHEMA)
    try:
        r = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that returns only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": "Generate a short video script about the Challenger Deep (the deepest point in the ocean). "
                        "Target 45 seconds. Return ONLY valid JSON matching this schema: "
                        + json.dumps(SCRIPT_SCHEMA),
                    },
                ],
                "temperature": 0.6,
                "max_tokens": 2000,
            },
            timeout=120,
        )
        print(f"HTTP status: {r.status_code}")
        body = r.text[:3000]
        print(f"Response body (sanitized, truncated): {body!r}")
        check("Groq script request returns 200", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"Script content length: {len(content)}")
            try:
                parsed = json.loads(content)
                errors = list(validator.iter_errors(parsed))
                check("Script JSON validates against SCRIPT_SCHEMA", len(errors) == 0,
                      f"{len(errors)} validation errors" if errors else "")
                if errors:
                    for e in errors[:5]:
                        print(f"  Validation error: {e.message} at {list(e.path)}")
            except json.JSONDecodeError as e:
                check("Script JSON is parseable", False, f"JSON decode error: {e}")
    except Exception as e:
        print(f"REQUEST ERROR: {type(e).__name__}: {e}")
        check("Groq script request succeeded", False, f"{type(e).__name__}: {e}")

    # TEST D — Scene structured JSON against SCENE_SCHEMA
    print("=== TEST D: SCENE_SCHEMA validation ===")
    scene_validator = Draft7Validator(SCENE_SCHEMA)
    try:
        r = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that returns only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": "Generate one scene for a short video about an octopus using tools. "
                        "Include speaker, narration, search_query, and image_prompt. "
                        "Return ONLY valid JSON matching this schema: "
                        + json.dumps(SCENE_SCHEMA),
                    },
                ],
                "temperature": 0.6,
                "max_tokens": 1000,
            },
            timeout=120,
        )
        print(f"HTTP status: {r.status_code}")
        body = r.text[:3000]
        print(f"Response body (sanitized, truncated): {body!r}")
        check("Groq scene request returns 200", r.status_code == 200, f"status={r.status_code}")
        if r.status_code == 200:
            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"Scene content length: {len(content)}")
            try:
                parsed = json.loads(content)
                errors = list(scene_validator.iter_errors(parsed))
                check("Scene JSON validates against SCENE_SCHEMA", len(errors) == 0,
                      f"{len(errors)} validation errors" if errors else "")
                if errors:
                    for e in errors[:5]:
                        print(f"  Validation error: {e.message} at {list(e.path)}")
            except json.JSONDecodeError as e:
                check("Scene JSON is parseable", False, f"JSON decode error: {e}")
    except Exception as e:
        print(f"REQUEST ERROR: {type(e).__name__}: {e}")
        check("Groq scene request succeeded", False, f"{type(e).__name__}: {e}")

    print("\n=== SUMMARY ===")
    for line in results:
        print(line)

    failed = [r for r in results if r.startswith("[FAIL]")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
