"""Integration tests for MiniMax M2.7 API.

These tests require a valid MINIMAX_API_KEY environment variable.
They are skipped automatically if the key is not set.
"""

import json
import os
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


@pytest.fixture
def client():
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ["MINIMAX_API_KEY"],
        base_url="https://api.minimax.io/v1",
    )


class TestMiniMaxCompletion:
    """Integration tests for MiniMax chat completions."""

    def test_basic_completion(self, client):
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0.1,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_completion_with_system_prompt(self, client):
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {"role": "system", "content": "You always respond in exactly one word."},
                {"role": "user", "content": "What color is the sky?"},
            ],
            temperature=0.1,
        )
        assert response.choices[0].message.content is not None

    def test_json_mode(self, client):
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with key "answer" and value 42. Output only JSON, no explanation.',
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        # Strip thinking tags if present
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Extract JSON from markdown code block if present
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        result = json.loads(content)
        assert "answer" in result
        assert result["answer"] == 42

    def test_streaming(self, client):
        stream = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Count from 1 to 3."}],
            temperature=0.1,
            stream=True,
        )
        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_highspeed_model(self, client):
        response = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
            temperature=0.1,
        )
        assert response.choices[0].message.content is not None

    def test_usage_tracking(self, client):
        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.1,
        )
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0
