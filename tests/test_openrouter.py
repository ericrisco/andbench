"""Tests for the OpenRouter provider. No unit test touches the network."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Mapping

import pytest

from andbench.harness.judge import JudgeModel
from andbench.harness.smoke import Completion, SmokeModel
from andbench.providers.openrouter import (
    API_KEY_ENV,
    DEFAULT_TEMPERATURE,
    MeasuredModel,
    OpenRouterClient,
    OpenRouterError,
    TextModel,
    api_key,
    candidate_models,
    extract_json,
    json_text_model,
)


def _response(text: str = "hola", *, prompt_tokens: int = 11, completion_tokens: int = 3) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()


class _Recorder:
    """A transport that records calls and replays a queue of outcomes."""

    def __init__(self, *outcomes: bytes | Exception) -> None:
        self.outcomes = list(outcomes) or [_response()]
        self.calls: list[tuple[str, dict[str, object], Mapping[str, str]]] = []

    def __call__(self, url: str, body: bytes, headers: Mapping[str, str]) -> bytes:
        self.calls.append((url, json.loads(body), headers))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _http_error(code: int, body: bytes = b'{"error": "boom"}') -> urllib.error.HTTPError:
    import io

    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(body))  # type: ignore[arg-type]


def _client(*outcomes: bytes | Exception, **kwargs: object) -> OpenRouterClient:
    recorder = _Recorder(*outcomes)
    slept: list[float] = []
    client = OpenRouterClient(
        model="test/model",
        key="secret-key",
        transport=recorder,
        sleep=slept.append,
        **kwargs,  # type: ignore[arg-type]
    )
    client._recorder = recorder  # type: ignore[attr-defined]
    client._slept = slept  # type: ignore[attr-defined]
    return client


# --- the key --------------------------------------------------------------


def test_key_is_read_from_the_environment() -> None:
    assert api_key({API_KEY_ENV: " k "}) == "k"


@pytest.mark.parametrize("env", [{}, {API_KEY_ENV: ""}, {API_KEY_ENV: "   "}])
def test_a_missing_key_says_where_to_put_it(env: dict[str, str]) -> None:
    with pytest.raises(OpenRouterError, match=r"01-TOOLS/openrouter/\.env"):
        api_key(env)


def test_the_error_never_suggests_passing_the_key_on_the_command_line() -> None:
    with pytest.raises(OpenRouterError, match="never pass it as a command-line flag"):
        api_key({})


def test_from_env_builds_a_client() -> None:
    client = OpenRouterClient.from_env("m", env={API_KEY_ENV: "k"})
    assert client.model == "m"
    assert client.key == "k"


# --- the request ----------------------------------------------------------


def test_the_request_is_deterministic_by_default() -> None:
    """A benchmark number that moves between runs for no stated reason is not one."""
    client = _client()
    client.generate("hi")
    _url, body, _headers = client._recorder.calls[0]  # type: ignore[attr-defined]
    assert body["temperature"] == DEFAULT_TEMPERATURE == 0.0
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["model"] == "test/model"
    assert "seed" not in body


def test_a_seed_is_sent_when_given() -> None:
    client = _client(seed=1234)
    client.generate("hi")
    _url, body, _headers = client._recorder.calls[0]  # type: ignore[attr-defined]
    assert body["seed"] == 1234


def test_the_key_travels_in_the_authorization_header_only() -> None:
    client = _client()
    client.generate("hi")
    _url, body, headers = client._recorder.calls[0]  # type: ignore[attr-defined]
    assert headers["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in json.dumps(body)


def test_the_completion_carries_text_and_token_counts() -> None:
    client = _client(_response("resposta", prompt_tokens=42, completion_tokens=7))
    completion = client.generate("hi")
    assert completion == Completion(text="resposta", prompt_tokens=42, completion_tokens=7)


# --- failure handling -----------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 408])
def test_transient_statuses_are_retried_then_succeed(status: int) -> None:
    client = _client(_http_error(status), _response("ok"))
    assert client.generate("hi").text == "ok"
    assert len(client._recorder.calls) == 2  # type: ignore[attr-defined]


def test_backoff_is_exponential_and_without_jitter() -> None:
    """Deterministic backoff so a schedule can be asserted and a log read back."""
    client = _client(_http_error(429), _http_error(429), _response("ok"), max_attempts=4)
    client.generate("hi")
    assert client._slept == [1.0, 2.0]  # type: ignore[attr-defined]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_fail_immediately(status: int) -> None:
    client = _client(_http_error(status))
    with pytest.raises(OpenRouterError, match=f"HTTP {status}"):
        client.generate("hi")
    assert len(client._recorder.calls) == 1  # type: ignore[attr-defined]


def test_retries_are_bounded() -> None:
    client = _client(_http_error(429), max_attempts=3)
    with pytest.raises(OpenRouterError, match="3 attempt"):
        client.generate("hi")
    assert len(client._recorder.calls) == 3  # type: ignore[attr-defined]


def test_the_failure_message_includes_the_api_detail() -> None:
    client = _client(_http_error(400, b'{"error": "no such model"}'))
    with pytest.raises(OpenRouterError, match="no such model"):
        client.generate("hi")


def test_a_network_error_is_retried_then_reported() -> None:
    client = _client(urllib.error.URLError("down"), max_attempts=2)
    with pytest.raises(OpenRouterError, match="network error"):
        client.generate("hi")


def test_a_network_error_can_recover() -> None:
    client = _client(urllib.error.URLError("blip"), _response("ok"))
    assert client.generate("hi").text == "ok"


def test_a_non_json_response_is_reported() -> None:
    client = _client(b"<html>502</html>")
    with pytest.raises(OpenRouterError, match="not JSON"):
        client.generate("hi")


def test_an_error_body_returned_with_status_200_is_not_treated_as_an_answer() -> None:
    """Otherwise an error string lands in the dataset as if it were a response."""
    client = _client(json.dumps({"error": {"message": "rate limited"}}).encode())
    with pytest.raises(OpenRouterError, match="API error"):
        client.generate("hi")


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"nope": 1},
    ],
)
def test_a_malformed_success_body_is_reported(payload: dict[str, object]) -> None:
    client = _client(json.dumps(payload).encode())
    with pytest.raises(OpenRouterError, match=r"no choices|empty message"):
        client.generate("hi")


def test_a_null_message_is_reported() -> None:
    client = _client(json.dumps({"choices": [{"message": {"content": None}}]}).encode())
    with pytest.raises(OpenRouterError, match="empty message"):
        client.generate("hi")


def test_missing_usage_counts_as_zero_tokens_not_a_crash() -> None:
    client = _client(json.dumps({"choices": [{"message": {"content": "x"}}]}).encode())
    completion = client.generate("hi")
    assert (completion.prompt_tokens, completion.completion_tokens) == (0, 0)


# --- accounting -----------------------------------------------------------


def test_usage_accumulates_across_calls() -> None:
    client = _client(_response(prompt_tokens=10, completion_tokens=5))
    for _ in range(3):
        client.generate("hi")
    assert client.totals == (30, 15)


def test_spend_applies_the_given_prices() -> None:
    client = _client(_response(prompt_tokens=1000, completion_tokens=1000))
    client.generate("hi")
    assert client.spend(0.002, 0.01) == pytest.approx(0.012)


def test_spend_is_zero_before_any_call() -> None:
    assert _client().spend(1.0, 1.0) == 0.0


# --- the adapters satisfy the harness seams -------------------------------


def test_text_model_satisfies_the_judge_seam() -> None:
    model = TextModel(_client(_response("{}")))
    assert isinstance(model, JudgeModel)
    assert model.complete("hi") == "{}"


def test_measured_model_satisfies_the_smoke_seam() -> None:
    model = MeasuredModel(_client(_response("B")))
    assert isinstance(model, SmokeModel)
    assert model.complete("hi").text == "B"


def test_json_text_model_trims_prose_around_the_payload() -> None:
    model = json_text_model("m", env={API_KEY_ENV: "k"})
    model.client.transport = _Recorder(_response('És clar!\n```json\n{"correct": true}\n```'))
    assert json.loads(model.complete("hi")) == {"correct": True}


# --- extract_json ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('prose {"a": 1} more', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('[{"a": 1}]', '[{"a": 1}]'),
        ('Aquí tens:\n[{"a": 1}, {"b": 2}]', '[{"a": 1}, {"b": 2}]'),
        ('{"outer": {"inner": 1}}', '{"outer": {"inner": 1}}'),
    ],
)
def test_extract_json_finds_the_payload(raw: str, expected: str) -> None:
    assert extract_json(raw) == expected


def test_extract_json_leaves_plain_text_alone() -> None:
    assert extract_json("  no json here  ") == "no json here"


# --- the shortlist --------------------------------------------------------


def test_no_judge_candidate_shares_a_lab_with_the_evaluated_family() -> None:
    """Maia is a Gemma-4 fine-tune, so a Google judge risks self-preference."""
    judges = candidate_models()[:3]
    assert judges
    assert not any(model.startswith("google/") for model in judges)


def test_no_draft_candidate_is_from_the_evaluated_family() -> None:
    """Item phrasing must not be biased toward the family under test."""
    drafters = candidate_models()[3:]
    assert drafters
    assert not any("gemma" in model for model in drafters)


# --- reasoning models (found by the live integration probe) ----------------


def _reasoning_response(content: str, reasoning_tokens: int, cost: float | None = 0.001) -> bytes:
    usage: dict[str, object] = {
        "prompt_tokens": 71,
        "completion_tokens": 174,
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
    if cost is not None:
        usage["cost"] = cost
    return json.dumps({"choices": [{"message": {"content": content}}], "usage": usage}).encode()


def test_reasoning_tokens_are_recorded() -> None:
    """They are invisible in the text but not in the bill."""
    client = _client(_reasoning_response('{"ok": true}', 158))
    client.generate("hi")
    assert client.reasoning_tokens == 158


def test_an_empty_message_from_a_reasoning_model_names_the_cause() -> None:
    """The real failure mode: thinking consumed the whole max_tokens budget."""
    client = _client(_reasoning_response("", 596), max_tokens=600)
    with pytest.raises(OpenRouterError, match=r"596 token\(s\) on reasoning"):
        client.generate("hi")


def test_an_empty_message_without_reasoning_stays_a_plain_error() -> None:
    client = _client(_reasoning_response("", 0))
    with pytest.raises(OpenRouterError, match="empty message"):
        client.generate("hi")


def test_the_default_token_cap_leaves_room_for_reasoning() -> None:
    from andbench.providers.openrouter import DEFAULT_MAX_TOKENS

    assert DEFAULT_MAX_TOKENS >= 2048


def test_the_providers_own_cost_figure_is_preferred() -> None:
    client = _client(_reasoning_response('{"ok": true}', 10, cost=0.00026))
    client.generate("hi")
    assert client.reported_cost_usd == pytest.approx(0.00026)


def test_a_missing_cost_makes_the_total_unknown_not_partial() -> None:
    """An under-count read as a total is how a budget gets blown quietly."""
    client = _client(_reasoning_response('{"ok": true}', 10, cost=None))
    client.generate("hi")
    assert client.reported_cost_usd is None


def test_reported_cost_is_unknown_before_any_call() -> None:
    assert _client().reported_cost_usd is None


# --- the convenience constructors -----------------------------------------


def test_judge_and_measured_constructors_read_the_environment() -> None:
    from andbench.providers.openrouter import judge_model, measured_model

    env = {API_KEY_ENV: "k"}
    assert judge_model("m", env=env).client.model == "m"
    assert measured_model("m", env=env).client.key == "k"


def test_an_unmatched_bracket_falls_back_to_the_stripped_text() -> None:
    assert extract_json("  { unclosed  ") == "{ unclosed"


# --- extract_json: the closer must match the opener, not be the last in the text ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}  hope that helps!', '{"a": 1}'),
        ('Here: {"a": 1} — note that {this} matters', '{"a": 1}'),
        ('{"nested": {"deep": 1}} trailing', '{"nested": {"deep": 1}}'),
        ('[{"a": 1}] and later [brackets]', '[{"a": 1}]'),
        (
            '{"s": "a brace } inside a string", "a": 1}',
            '{"s": "a brace } inside a string", "a": 1}',
        ),
        (
            '{"s": "an escaped \\" quote and }", "a": 1} tail',
            '{"s": "an escaped \\" quote and }", "a": 1}',
        ),
    ],
)
def test_extract_json_matches_by_depth_not_by_the_last_bracket(raw: str, expected: str) -> None:
    """A friendly note containing braces would otherwise discard a paid call."""
    assert extract_json(raw) == expected


def test_every_depth_matched_payload_actually_parses() -> None:
    for raw in (
        '{"a": 1} hope that helps!',
        'Here: {"a": 1} — note that {this} matters',
        '{"s": "a brace } inside", "a": 1}',
        '[{"a": 1}] and later [brackets]',
    ):
        assert json.loads(extract_json(raw))


def test_an_unbalanced_payload_still_returns_something_for_the_parser_to_reject() -> None:
    """Better a clear parse error than a silent truncation."""
    assert extract_json('{"a": 1') == '{"a": 1'
