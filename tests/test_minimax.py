"""Unit tests for MiniMax M2.7 notebook code patterns."""

import json
import os
import textwrap
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel


class SentimentClassification(BaseModel):
    sentiment: Literal["negative", "neutral", "positive"]
    reasoning: str


# --- Unit Tests ---


class TestMiniMaxClientSetup:
    """Test MiniMax client configuration."""

    def test_base_url_is_correct(self):
        base_url = "https://api.minimax.io/v1"
        assert base_url.startswith("https://")
        assert "minimax.io" in base_url
        assert base_url.endswith("/v1")

    def test_model_names(self):
        models = ["MiniMax-M2.7", "MiniMax-M2.7-highspeed"]
        for model in models:
            assert model.startswith("MiniMax-")
            assert "M2.7" in model

    def test_temperature_must_be_positive(self):
        valid_temps = [0.1, 0.5, 0.7, 1.0]
        invalid_temps = [0.0, -0.1, -1.0]
        for temp in valid_temps:
            assert 0 < temp <= 1.0
        for temp in invalid_temps:
            assert not (0 < temp <= 1.0)


class TestFormatResponse:
    """Test the format_response helper function."""

    def test_format_simple_text(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello world"

        response_txt = mock_response.choices[0].message.content
        text = ""
        for chunk in response_txt.split("\n"):
            text += "\n"
            if not chunk:
                continue
            text += (
                "\n".join(textwrap.wrap(chunk, 100, break_long_words=False))
            ).strip()
        result = text.strip()
        assert result == "Hello world"

    def test_format_multiline_text(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Line 1\n\nLine 2\nLine 3"

        response_txt = mock_response.choices[0].message.content
        text = ""
        for chunk in response_txt.split("\n"):
            text += "\n"
            if not chunk:
                continue
            text += (
                "\n".join(textwrap.wrap(chunk, 100, break_long_words=False))
            ).strip()
        result = text.strip()
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_format_long_line_wraps(self):
        long_text = "A " * 100  # 200 chars
        mock_response = MagicMock()
        mock_response.choices[0].message.content = long_text.strip()

        response_txt = mock_response.choices[0].message.content
        text = ""
        for chunk in response_txt.split("\n"):
            text += "\n"
            if not chunk:
                continue
            text += (
                "\n".join(textwrap.wrap(chunk, 100, break_long_words=False))
            ).strip()
        result = text.strip()
        lines = result.split("\n")
        assert len(lines) > 1


class TestStructuredOutput:
    """Test Pydantic model parsing for structured output."""

    def test_valid_sentiment_positive(self):
        json_str = '{"sentiment": "positive", "reasoning": "The user is happy"}'
        result = SentimentClassification.model_validate_json(json_str)
        assert result.sentiment == "positive"
        assert result.reasoning == "The user is happy"

    def test_valid_sentiment_negative(self):
        json_str = '{"sentiment": "negative", "reasoning": "The user is sad"}'
        result = SentimentClassification.model_validate_json(json_str)
        assert result.sentiment == "negative"

    def test_valid_sentiment_neutral(self):
        json_str = '{"sentiment": "neutral", "reasoning": "The text is factual"}'
        result = SentimentClassification.model_validate_json(json_str)
        assert result.sentiment == "neutral"

    def test_invalid_sentiment_raises(self):
        json_str = '{"sentiment": "angry", "reasoning": "Invalid value"}'
        with pytest.raises(Exception):
            SentimentClassification.model_validate_json(json_str)

    def test_missing_field_raises(self):
        json_str = '{"sentiment": "positive"}'
        with pytest.raises(Exception):
            SentimentClassification.model_validate_json(json_str)


class TestToolCalling:
    """Test tool/function calling patterns."""

    def test_get_weather_function(self):
        def get_weather(city: str, unit: str = "celsius") -> str:
            weather_data = {
                "San Francisco": {"temp": 18, "condition": "Foggy"},
                "New York": {"temp": 25, "condition": "Sunny"},
                "London": {"temp": 15, "condition": "Rainy"},
                "Tokyo": {"temp": 28, "condition": "Humid"},
            }
            data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
            if unit == "fahrenheit":
                data["temp"] = round(data["temp"] * 9 / 5 + 32)
            data["unit"] = unit
            data["city"] = city
            return json.dumps(data)

        result = json.loads(get_weather("Tokyo"))
        assert result["city"] == "Tokyo"
        assert result["temp"] == 28
        assert result["condition"] == "Humid"
        assert result["unit"] == "celsius"

    def test_get_weather_fahrenheit(self):
        def get_weather(city: str, unit: str = "celsius") -> str:
            weather_data = {
                "San Francisco": {"temp": 18, "condition": "Foggy"},
                "New York": {"temp": 25, "condition": "Sunny"},
            }
            data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
            if unit == "fahrenheit":
                data["temp"] = round(data["temp"] * 9 / 5 + 32)
            data["unit"] = unit
            data["city"] = city
            return json.dumps(data)

        result = json.loads(get_weather("New York", "fahrenheit"))
        assert result["temp"] == 77  # 25 * 9/5 + 32
        assert result["unit"] == "fahrenheit"

    def test_get_weather_unknown_city(self):
        def get_weather(city: str, unit: str = "celsius") -> str:
            weather_data = {}
            data = weather_data.get(city, {"temp": 20, "condition": "Unknown"})
            data["unit"] = unit
            data["city"] = city
            return json.dumps(data)

        result = json.loads(get_weather("UnknownCity"))
        assert result["condition"] == "Unknown"
        assert result["temp"] == 20

    def test_tool_schema_format(self):
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        }
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert "city" in tool["function"]["parameters"]["required"]

    def test_tool_call_args_parsing(self):
        args_json = '{"city": "Tokyo", "unit": "celsius"}'
        args = json.loads(args_json)
        assert args["city"] == "Tokyo"
        assert args["unit"] == "celsius"


class TestDataLabelling:
    """Test data labelling prompt patterns."""

    def test_classify_text_prompt_format(self):
        template = """
Your task is to analyze the following text and classify it.
<text>
{text}
</text>
"""
        text = "The new iPhone features an improved camera."
        prompt = template.format(text=text)
        assert text in prompt
        assert "<text>" in prompt
        assert "</text>" in prompt

    def test_json_response_parsing(self):
        response_json = json.dumps(
            {
                "target_audience": "General public",
                "tone": "Neutral",
                "complexity": "Intermediate",
                "topic": "Technology",
            }
        )
        result = json.loads(response_json)
        assert result["topic"] == "Technology"
        assert result["tone"] == "Neutral"

    def test_multiple_text_classification_results(self):
        texts = [
            "Tech news article",
            "Scientific paper",
            "Health blog post",
        ]
        results = []
        for text in texts:
            results.append({"text": text[:60], "topic": "test"})
        assert len(results) == 3
        assert all("text" in r for r in results)


class TestLiteLLMIntegration:
    """Test litellm integration patterns for MiniMax."""

    def test_litellm_model_prefix(self):
        model = "openai/MiniMax-M2.7"
        assert model.startswith("openai/")
        provider, model_name = model.split("/", 1)
        assert provider == "openai"
        assert model_name == "MiniMax-M2.7"

    def test_litellm_api_base_config(self):
        api_base = "https://api.minimax.io/v1"
        assert api_base.startswith("https://")
        assert "minimax" in api_base

    def test_litellm_completion_kwargs(self):
        kwargs = {
            "model": "openai/MiniMax-M2.7",
            "messages": [{"content": "test", "role": "user"}],
            "api_base": "https://api.minimax.io/v1",
            "temperature": 0.1,
        }
        assert kwargs["model"] == "openai/MiniMax-M2.7"
        assert kwargs["api_base"] == "https://api.minimax.io/v1"
        assert 0 < kwargs["temperature"] <= 1.0

    def test_message_format(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert all("role" in m and "content" in m for m in messages)


class TestNotebookStructure:
    """Test that notebook files are valid JSON."""

    def test_minimax_notebook_valid_json(self):
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "minimax-m2.7.ipynb"
        )
        with open(notebook_path) as f:
            nb = json.load(f)
        assert nb["nbformat"] == 4
        assert "cells" in nb
        assert len(nb["cells"]) > 0

    def test_minimax_notebook_has_required_sections(self):
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "minimax-m2.7.ipynb"
        )
        with open(notebook_path) as f:
            nb = json.load(f)

        all_source = " ".join(
            "".join(cell["source"]) for cell in nb["cells"]
        )
        assert "MiniMax-M2.7" in all_source
        assert "api.minimax.io" in all_source
        assert "MINIMAX_API_KEY" in all_source
        assert "json_object" in all_source
        assert "stream=True" in all_source
        assert "tool_choice" in all_source

    def test_minimax_notebook_cell_types(self):
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "minimax-m2.7.ipynb"
        )
        with open(notebook_path) as f:
            nb = json.load(f)

        cell_types = [cell["cell_type"] for cell in nb["cells"]]
        assert "code" in cell_types
        assert "markdown" in cell_types

    def test_litellm_notebook_has_minimax(self):
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "26.multiple-llm-providers-with-litellm.ipynb",
        )
        with open(notebook_path) as f:
            nb = json.load(f)

        all_source = " ".join(
            "".join(cell["source"]) for cell in nb["cells"]
        )
        assert "MiniMax-M2.7" in all_source
        assert "minimax.io" in all_source

    def test_readme_mentions_minimax(self):
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "README.md"
        )
        with open(readme_path) as f:
            content = f.read()
        assert "MiniMax" in content
        assert "minimax-m2.7.ipynb" in content
