from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agentguard.teacher.response_schema import (
    TeacherResponse,
    should_upgrade_to_plus,
)


class BailianClient:
    """Client for Bailian (Alibaba Cloud) multimodal LLM API.

    Config (from ``agentguard/configs/bailian.yaml``)::

        provider: bailian
        bulk_model: qwen3-vl-flash
        review_model: qwen3-vl-plus
        enable_thinking: false
        temperature: 0.0
        max_tokens: 900
        timeout_seconds: 120
        max_retries: 2
        max_concurrency: 2

    Parameters
    ----------
    config_path : str, optional
        Path to the Bailian YAML config file.  Defaults to
        ``agentguard/configs/bailian.yaml``.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config = self._load_config(config_path)
        self.model_name: str = self.config.get("bulk_model", "qwen3-vl-flash")
        self.review_model: str = self.config.get("review_model", "qwen3-vl-plus")
        self.temperature: float = float(self.config.get("temperature", 0.0))
        self.max_tokens: int = int(self.config.get("max_tokens", 900))
        self.timeout: int = int(self.config.get("timeout_seconds", 120))
        self.max_retries: int = int(self.config.get("max_retries", 2))
        self.max_concurrency: int = int(self.config.get("max_concurrency", 2))

        # API key loaded from environment
        self.api_key: str = os.environ.get(
            "BAILIAN_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")
        )
        if not self.api_key:
            raise RuntimeError(
                "Bailian API key not found. Set BAILIAN_API_KEY or "
                "DASHSCOPE_API_KEY environment variable."
            )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: Dict[str, Any],
        images: List[str],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a request to the Bailian multimodal LLM API.

        Parameters
        ----------
        prompt : dict
            Structured prompt with ``system`` and ``user`` fields.
        images : list of str
            Paths to images to include in the request.
        model : str, optional
            Model name override (defaults to ``self.model_name``).

        Returns
        -------
        dict
            Contains keys ``raw_response``, ``parsed_response`` (a
            ``TeacherResponse`` or ``None``), ``tokens`` (dict with
            ``prompt``, ``completion``, ``total``), ``latency`` (float
            seconds), ``success`` (bool), ``error`` (str or None).
        """
        model_name = model or self.model_name
        start = time.perf_counter()
        result: Dict[str, Any] = {
            "raw_response": "",
            "parsed_response": None,
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "latency": 0.0,
            "success": False,
            "error": None,
        }

        try:
            # Build the API payload
            api_messages = self._build_messages(prompt, images)
            api_response = self._post_request(model_name, api_messages)

            elapsed = time.perf_counter() - start
            result["latency"] = elapsed

            if api_response is None:
                result["error"] = "Empty API response"
                return result

            raw_text = self._extract_text(api_response)
            result["raw_response"] = raw_text

            # Token accounting
            usage = api_response.get("usage", {})
            if isinstance(usage, dict):
                result["tokens"]["prompt"] = int(usage.get("input_tokens", 0))
                result["tokens"]["completion"] = int(usage.get("output_tokens", 0))
                result["tokens"]["total"] = (
                    result["tokens"]["prompt"] + result["tokens"]["completion"]
                )

            # Parse JSON response
            parsed = self._parse_response(raw_text)
            if parsed is not None:
                result["parsed_response"] = parsed
                result["success"] = True
            else:
                result["error"] = "JSON parse failure"

        except Exception as exc:
            elapsed = time.perf_counter() - start
            result["latency"] = elapsed
            result["error"] = str(exc)

        return result

    def call_with_fallback(self, prompt: Dict[str, Any], images: List[str]) -> Dict[str, Any]:
        """Try Flash first, upgrade to Plus if needed.

        Max: 1 Flash + 1 Plus per event.  The upgrade decision is made
        by :func:`~agentguard.teacher.response_schema.should_upgrade_to_plus`.

        Parameters
        ----------
        prompt : dict
            Structured prompt.
        images : list of str
            Image paths.

        Returns
        -------
        dict
            API result dict with an additional key ``fallback_used``
            (bool).
        """
        # Attempt with Flash (bulk model)
        flash_result = self.call(prompt, images, model=self.model_name)

        if flash_result["success"]:
            parsed = flash_result["parsed_response"]
            if parsed is not None and hasattr(parsed, "confidence"):
                # Check if upgrade is needed using a default rollout gate of 0.5
                # when no gate context is available at this level.
                if not should_upgrade_to_plus(parsed, rollout_gate=0.5):
                    flash_result["fallback_used"] = False
                    return flash_result

        # Upgrade to Plus (review model)
        plus_result = self.call(prompt, images, model=self.review_model)
        plus_result["fallback_used"] = True
        return plus_result

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict[str, Any]:
        """Load the Bailian YAML configuration."""
        if config_path is None:
            config_path = "agentguard/configs/bailian.yaml"
        if not os.path.isfile(config_path):
            # Return defaults if config file does not exist
            return {
                "provider": "bailian",
                "bulk_model": "qwen3-vl-flash",
                "review_model": "qwen3-vl-plus",
                "enable_thinking": False,
                "temperature": 0.0,
                "max_tokens": 900,
                "timeout_seconds": 120,
                "max_retries": 2,
                "max_concurrency": 2,
                "prices": None,
            }
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_messages(
        self,
        prompt: Dict[str, Any],
        images: List[str],
    ) -> List[Dict[str, Any]]:
        """Build the message list for the Bailian API.

        Converts image paths to base64 data URIs.
        """
        import base64

        messages: List[Dict[str, Any]] = []

        # System message
        system_text = prompt.get("system", "")
        if system_text:
            messages.append({"role": "system", "content": system_text})

        # User message with text + images
        user_content: List[Dict[str, Any]] = []

        # Text part
        user_text = prompt.get("user", "")
        if user_text:
            user_content.append({"type": "text", "text": user_text})

        # Image parts
        for img_path in images:
            if os.path.isfile(img_path):
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                # Infer MIME type from extension
                ext = os.path.splitext(img_path)[1].lower()
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                    ext.lstrip("."), "image/jpeg"
                )
                user_content.append(
                    {
                        "type": "image",
                        "image": f"data:{mime};base64,{img_data}",
                    }
                )

        messages.append({"role": "user", "content": user_content})
        return messages

    def _post_request(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Post a request to the Bailian API endpoint.

        This is a stub for the actual HTTP call.  In production this
        would use ``requests.post`` against the Bailian (DashScope)
        endpoint.
        """
        # Stub: simulate a response for test/development.
        # In production, replace with:
        #
        # import requests
        # resp = requests.post(
        #     "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
        #     headers={
        #         "Authorization": f"Bearer {self.api_key}",
        #         "Content-Type": "application/json",
        #     },
        #     json={
        #         "model": model_name,
        #         "input": {"messages": messages},
        #         "parameters": {
        #             "temperature": self.temperature,
        #             "max_tokens": self.max_tokens,
        #             "enable_thinking": self.config.get("enable_thinking", False),
        #         },
        #     },
        #     timeout=self.timeout,
        # )
        # return resp.json()
        return None

    @staticmethod
    def _extract_text(api_response: Dict[str, Any]) -> str:
        """Extract the generated text from a Bailian API response.

        Expected structure::

            {
                "output": {
                    "choices": [
                        {"message": {"content": "..."}}
                    ]
                }
            }
        """
        try:
            choices = api_response.get("output", {}).get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        except Exception:
            pass
        return json.dumps(api_response, ensure_ascii=False)

    @staticmethod
    def _parse_response(raw_text: str) -> Optional[TeacherResponse]:
        """Attempt to parse the raw response text into a ``TeacherResponse``.

        Tries to extract a JSON block from the text (handling markdown
        code fences) and parse it with the pydantic model.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Find the first and last ```
            lines = text.split("\n")
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    start = i + 1
                    break
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end]).strip()

        try:
            data = json.loads(text)
            return TeacherResponse(**data)
        except (json.JSONDecodeError, Exception):
            return None
