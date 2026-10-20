"""Generic OpenAI-compatible HTTP adapter; revise the settings for your company."""
import json
import os
from urllib.request import Request, urlopen

# Edit these defaults at work, or set environment variables with these names.
API_URL = os.getenv("LLM_API_URL", "https://your-company-server/v1/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "your-company-model")
API_KEY = os.getenv("LLM_API_KEY", "")

# Set False if your service does not support response_format/json_schema.
USE_STRUCTURED_OUTPUT = True


def base_model(*, system_prompt, user_prompt, json_schema):
    """Called automatically by pipeline.py. Return the final model JSON string."""
    mode = json.loads(user_prompt)["source_mode"]

    # Fallback: include the schema in the instructions. Pipeline still validates.
    if not USE_STRUCTURED_OUTPUT:
        system_prompt += "\nRequired JSON Schema:\n" + json.dumps(json_schema)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    if USE_STRUCTURED_OUTPUT:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "interview_analysis",
                "strict": True,
                "schema": json_schema,
            },
        }

    if mode == "model":
        # COMPANY-SPECIFIC: enable RAG here, or use a RAG-enabled API_URL above.
        # There is no universal RAG setting. Replace this pass with your service's
        # documented settings/tool loop if its endpoint requires them.
        pass

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    request = Request(API_URL, data=json.dumps(payload).encode("utf-8"),
                      headers=headers, method="POST")
    with urlopen(request, timeout=300) as response:
        result = json.load(response)

    # Revise this extraction if your company returns a different response shape.
    choice = result["choices"][0]
    message = choice["message"]
    if message.get("refusal"):
        raise RuntimeError("Model refused the request")
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("Model did not finish: check truncation or required tool calls")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Model returned no JSON text")
    return text
