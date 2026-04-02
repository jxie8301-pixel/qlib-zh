#!/usr/bin/env python3
"""Apply a direct BigModel forward patch into the installed rdagent backend.

This script updates the runtime backend used inside the rdagent container so
that requests are forwarded to the user's BigModel endpoint while preserving
rdagent's expected JSON-string return shape.
"""

from __future__ import annotations

from pathlib import Path


TARGET = Path("/usr/local/lib/python3.10/site-packages/rdagent/oai/backend/base.py")
FACTOR_TARGET = Path("/usr/local/lib/python3.10/site-packages/rdagent/scenarios/qlib/proposal/factor_proposal.py")
EVA_TARGET = Path("/usr/local/lib/python3.10/site-packages/rdagent/components/coder/factor_coder/eva_utils.py")
MARKER = "\n# >>> BIGMODEL DIRECT FORWARD PATCH >>>\n"


PATCH_BLOCK = r'''
# >>> BIGMODEL DIRECT FORWARD PATCH >>>
try:
    import json
    import os
    import threading
    import time

    import requests

    _bm_rate_lock = threading.Lock()
    _bm_last_request = {"t": 0.0}

    def __extract_text(obj):
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            if "choices" in obj:
                try:
                    choice0 = obj["choices"][0]
                    if isinstance(choice0, dict):
                        message = choice0.get("message", {})
                        if isinstance(message, dict) and "content" in message:
                            return __extract_text(message["content"])
                        if "text" in choice0:
                            return __extract_text(choice0["text"])
                except Exception:
                    pass
            for key in ("result", "data", "text", "answer"):
                if key in obj:
                    return __extract_text(obj[key])
            for value in obj.values():
                text = __extract_text(value)
                if text:
                    return text
        if isinstance(obj, list):
            for item in obj:
                text = __extract_text(item)
                if text:
                    return text
        return ""

    def __to_openai_json_text(rjson, fallback_text=""):
        rid = ""
        if isinstance(rjson, dict):
            rid = rjson.get("id", "")
        return json.dumps(
            {
                "id": rid,
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": fallback_text,
                        }
                    }
                ],
                "_raw": rjson,
            }
        )

    def __normalize_content_text(content_text):
        if not isinstance(content_text, str):
            return content_text
        stripped = content_text.strip()
        if not stripped:
            return content_text
        try:
            parsed = json.loads(stripped)
        except Exception:
            return content_text
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                return json.dumps(parsed[0], ensure_ascii=False)
            merged = {}
            synthetic_idx = 1
            for item in parsed:
                if not isinstance(item, dict):
                    merged = None
                    break
                if len(item) == 1:
                    only_key = next(iter(item))
                    only_val = item[only_key]
                    if isinstance(only_val, dict):
                        merged[str(only_key)] = only_val
                        continue
                key = item.get("factor_name") or item.get("name") or item.get("hypothesis") or item.get("title")
                if not key:
                    key = f"factor_{synthetic_idx}"
                    synthetic_idx += 1
                merged[str(key)] = item
            if merged:
                return json.dumps(merged, ensure_ascii=False)
        return content_text

    def __extract_json_payload_text(content_text):
        if not isinstance(content_text, str):
            return content_text
        text = content_text.strip()
        if "```" in text:
            parts = text.split("```")
            for idx in range(len(parts) - 1, 0, -1):
                block = parts[idx].strip()
                lower = block.lower()
                if lower.startswith("json"):
                    block = block[4:].strip()
                elif lower.startswith("python") or lower.startswith("py"):
                    continue
                if block.startswith("{") or block.startswith("["):
                    return block
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = text.find(open_char)
            end = text.rfind(close_char)
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1].strip()
                if candidate.startswith(open_char):
                    return candidate
        return content_text

    def __debug_dump(label, payload):
        try:
            with open("/tmp/bigmodel_debug.log", "a", encoding="utf-8") as fh:
                fh.write(f"--- {label} ---\n")
                fh.write(json.dumps(payload, ensure_ascii=False, default=str))
                fh.write("\n")
        except Exception:
            pass

    def __normalize_response_json(rjson):
        if isinstance(rjson, list):
            if not rjson:
                return {"choices": []}
            first = rjson[0]
            if isinstance(first, dict):
                return {"choices": rjson}
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": str(first)}}
                ]
            }
        return rjson

    def __sleep_for_rate_limit(resp, attempt, base_sleep, min_interval):
        retry_after = resp.headers.get("Retry-After") if getattr(resp, "headers", None) else None
        delay = base_sleep
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except Exception:
                pass
        delay = max(delay, min_interval)
        delay = min(delay * (attempt + 1), 120.0)
        time.sleep(delay)

    def __rdagent_bigmodel_build(self, *args, **kwargs):
        base = os.environ.get("OPENAI_API_BASE", "")
        key = os.environ.get("OPENAI_API_KEY", "")
        model = os.environ.get("CHAT_MODEL", "")
        verify_env = os.environ.get("BIGMODEL_VERIFY_SSL", "1")
        verify = False if verify_env in ("0", "false", "False") else True
        timeout = int(os.environ.get("BIGMODEL_TIMEOUT", "60"))
        retries = int(os.environ.get("BIGMODEL_RETRIES", "8"))
        min_interval = float(os.environ.get("BIGMODEL_MIN_INTERVAL", "10.0"))

        messages = None
        if "messages" in kwargs:
            messages = kwargs.get("messages")
        elif "user_prompt" in kwargs:
            user_prompt = kwargs.get("user_prompt")
            system_prompt = kwargs.get("system_prompt", "")
            if isinstance(user_prompt, str):
                msgs = []
                if isinstance(system_prompt, str) and system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.append({"role": "user", "content": user_prompt})
                messages = msgs
            elif isinstance(user_prompt, (list, dict)):
                messages = user_prompt
        elif args:
            first = args[0]
            if isinstance(first, (list, dict)):
                messages = first
            elif isinstance(first, str):
                system = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
                msgs = []
                if system:
                    msgs.append({"role": "system", "content": system})
                msgs.append({"role": "user", "content": first})
                messages = msgs
            else:
                messages = first

        if base and "open.bigmodel.cn" in base:
            url = base.rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = key

            payload = {
                "model": model or kwargs.get("model", ""),
                "messages": messages or kwargs.get("messages", []),
            }
            for name in ("max_tokens", "temperature", "top_p"):
                if name in kwargs:
                    payload[name] = kwargs[name]
            # Avoid provider-side schema errors from unsupported reasoning fields.

            backoff = 5.0
            for attempt in range(max(1, retries)):
                try:
                    with _bm_rate_lock:
                        now = time.time()
                        elapsed = now - _bm_last_request["t"]
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)
                        _bm_last_request["t"] = time.time()

                    resp = requests.post(
                        url,
                        json=payload,
                        headers=headers,
                        verify=verify,
                        timeout=timeout,
                    )

                    if resp.status_code == 429:
                        if attempt + 1 < retries:
                            __sleep_for_rate_limit(resp, attempt, backoff, min_interval)
                            backoff = min(backoff * 2.0, 120.0)
                            continue
                        raise requests.exceptions.HTTPError(
                            "429 Too Many Requests", response=resp
                        )

                    resp.raise_for_status()

                    try:
                        rjson = resp.json()
                    except Exception:
                        return __to_openai_json_text({}, resp.text or "")

                    rjson = __normalize_response_json(rjson)
                    if isinstance(rjson, dict) and "choices" in rjson:
                        cleaned = dict(rjson)
                        cleaned_choices = []
                        for choice in rjson.get("choices", []):
                            if isinstance(choice, dict):
                                cleaned_choice = dict(choice)
                                msg = cleaned_choice.get("message")
                                if isinstance(msg, dict) and "content" in msg:
                                    msg = dict(msg)
                                    msg["content"] = __normalize_content_text(msg["content"])
                                    cleaned_choice["message"] = msg
                                cleaned_choices.append(cleaned_choice)
                            else:
                                cleaned_choices.append(choice)
                        cleaned["choices"] = cleaned_choices
                        __debug_dump("choices_cleaned", cleaned)
                        return json.dumps(cleaned, ensure_ascii=False)

                    text = __extract_text(rjson)
                    normalized_text = __normalize_content_text(__extract_json_payload_text(text))
                    if normalized_text != text:
                        __debug_dump("normalized_content", {"before": text, "after": normalized_text})
                    else:
                        __debug_dump("raw_content", {"text": text})
                    return __to_openai_json_text(rjson, normalized_text)

                except requests.exceptions.HTTPError as exc:
                    status = None
                    try:
                        status = exc.response.status_code
                    except Exception:
                        pass
                    if status == 429 and attempt + 1 < retries:
                        __sleep_for_rate_limit(exc.response, attempt, backoff, min_interval)
                        backoff = min(backoff * 2.0, 120.0)
                        continue
                    raise
                except Exception:
                    if attempt + 1 < retries:
                        time.sleep(min(backoff, 120.0))
                        backoff = min(backoff * 2.0, 120.0)
                        continue
                    raise

        raise RuntimeError("BIGMODEL direct forward conditions not met")

    def __rdagent_bigmodel_try(self, *args, **kwargs):
        return __rdagent_bigmodel_build(None, *args, **kwargs)

    from rdagent.oai.backend.base import APIBackend

    APIBackend.build_messages_and_create_chat_completion = __rdagent_bigmodel_build
    APIBackend._try_create_chat_completion_or_embedding = __rdagent_bigmodel_try
    print("applied bigmodel direct forward patch (robust)")
except Exception as e:
    print("failed to apply bigmodel patch", e)
'''


def main() -> int:
    if not TARGET.exists():
        print(f"base.py not found: {TARGET}")
        return 1

    text = TARGET.read_text()
    if MARKER in text:
        start = text.index(MARKER)
        end = text.find("\n# <<< BIGMODEL DIRECT FORWARD PATCH <<<\n", start)
        if end == -1:
            end = len(text)
        else:
            end += len("\n# <<< BIGMODEL DIRECT FORWARD PATCH <<<\n")
        text = text[:start] + text[end:]

    TARGET.write_text(text + PATCH_BLOCK)
    print(f"wrote patch to {TARGET}")

    if FACTOR_TARGET.exists():
        factor_text = FACTOR_TARGET.read_text()
        old = """    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:\n        response_dict = json.loads(response)\n        tasks = []\n"""
        new = """    def convert_response(self, response: str, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:\n        response_dict = json.loads(response)\n        if isinstance(response_dict, list):\n            normalized = {}\n            for i, item in enumerate(response_dict, 1):\n                if isinstance(item, dict):\n                    if len(item) == 1:\n                        only_key = next(iter(item))\n                        only_val = item[only_key]\n                        if isinstance(only_val, dict):\n                            normalized[str(only_key)] = only_val\n                            continue\n                    key = item.get(\"factor_name\") or item.get(\"name\") or item.get(\"title\") or f\"factor_{i}\"\n                    normalized[str(key)] = item\n                else:\n                    normalized[f\"factor_{i}\"] = {\"description\": str(item), \"formulation\": \"\", \"variables\": {}}\n            response_dict = normalized\n        tasks = []\n"""
        if old in factor_text and new not in factor_text:
            factor_text = factor_text.replace(old, new)
        old_loop = """        for factor_name in response_dict:\n            description = response_dict[factor_name][\"description\"]\n            formulation = response_dict[factor_name][\"formulation\"]\n            variables = response_dict[factor_name][\"variables\"]\n            tasks.append(\n                FactorTask(\n                    factor_name=factor_name,\n                    factor_description=description,\n                    factor_formulation=formulation,\n                    variables=variables,\n                )\n            )\n"""
        new_loop = """        for factor_name in response_dict:\n            factor_value = response_dict[factor_name]\n            if isinstance(factor_value, list):\n                if len(factor_value) == 1 and isinstance(factor_value[0], dict):\n                    factor_value = factor_value[0]\n                else:\n                    merged = {}\n                    for item in factor_value:\n                        if isinstance(item, dict):\n                            merged.update(item)\n                    factor_value = merged or (factor_value[0] if factor_value else {})\n            if not isinstance(factor_value, dict):\n                factor_value = {}\n            description = factor_value.get(\"description\", \"\")\n            formulation = factor_value.get(\"formulation\", \"\")\n            variables = factor_value.get(\"variables\", {})\n            tasks.append(\n                FactorTask(\n                    factor_name=factor_name,\n                    factor_description=description,\n                    factor_formulation=formulation,\n                    variables=variables,\n                )\n            )\n"""
        if old_loop in factor_text and new_loop not in factor_text:
            factor_text = factor_text.replace(old_loop, new_loop)
            FACTOR_TARGET.write_text(factor_text)
            print(f"wrote list-normalization patch to {FACTOR_TARGET}")
        else:
            FACTOR_TARGET.write_text(factor_text)
            print(f"factor proposal patch already present or pattern missing in {FACTOR_TARGET}")

    if EVA_TARGET.exists():
        eva_text = EVA_TARGET.read_text()
        old_eva = """                if isinstance(final_evaluation_dict, list):\n                    if final_evaluation_dict and isinstance(final_evaluation_dict[0], dict):\n                        final_evaluation_dict = final_evaluation_dict[0]\n                    else:\n                        final_evaluation_dict = {}\n                if isinstance(final_evaluation_dict, dict):\n                    if 'final_decision' not in final_evaluation_dict and 'final_feedback' not in final_evaluation_dict:\n                        for key in ('decision', 'final decision', 'judge', 'result'):\n                            if key in final_evaluation_dict:\n                                final_evaluation_dict['final_decision'] = final_evaluation_dict[key]\n                                break\n                        for key in ('feedback', 'final feedback', 'comment', 'reason'):\n                            if key in final_evaluation_dict:\n                                final_evaluation_dict['final_feedback'] = final_evaluation_dict[key]\n                                break\n                final_decision = final_evaluation_dict[\"final_decision\"]\n                final_feedback = final_evaluation_dict[\"final_feedback\"]\n"""
        new_eva = """                if isinstance(final_evaluation_dict, list):\n                    if final_evaluation_dict and isinstance(final_evaluation_dict[0], dict):\n                        final_evaluation_dict = final_evaluation_dict[0]\n                    else:\n                        final_evaluation_dict = {}\n                if not isinstance(final_evaluation_dict, dict):\n                    final_evaluation_dict = {}\n                if 'final_decision' not in final_evaluation_dict:\n                    for key in ('decision', 'final decision', 'judge', 'result'):\n                        if key in final_evaluation_dict:\n                            final_evaluation_dict['final_decision'] = final_evaluation_dict[key]\n                            break\n                if 'final_feedback' not in final_evaluation_dict:\n                    for key in ('feedback', 'final feedback', 'comment', 'reason'):\n                        if key in final_evaluation_dict:\n                            final_evaluation_dict['final_feedback'] = final_evaluation_dict[key]\n                            break\n                final_decision = final_evaluation_dict.get(\"final_decision\", False)\n                final_feedback = final_evaluation_dict.get(\"final_feedback\", str(final_evaluation_dict))\n"""
        if old_eva in eva_text and new_eva not in eva_text:
            eva_text = eva_text.replace(old_eva, new_eva)
            EVA_TARGET.write_text(eva_text)
            print(f"wrote evaluation normalization patch to {EVA_TARGET}")
        else:
            EVA_TARGET.write_text(eva_text)
            print(f"evaluation patch already present or pattern missing in {EVA_TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())