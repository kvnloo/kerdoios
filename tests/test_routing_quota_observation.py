import json
from kerdoios.observed import record_routing_attempt, load_observations


def test_actual_429_retains_identity_reset_counts_without_task_acceptance(tmp_path):
    path = tmp_path / "observed.jsonl"
    row = record_routing_attempt(provider="opencode-go", model="deepseek-test", trace_id="first429",
        http_status=429, input_tokens=12, output_tokens=None,
        observed_at="2026-10-24T12:00:00Z", quota_group="synthetic-account",
        quota_headers={"x-ratelimit-reset-requests-day": "2026-10-25T00:00:00Z",
                       "x-ratelimit-limit-requests-day": "20", "retry-after": "60",
                       "authorization": "never retain", "x-ratelimit-reset-api-key": "secret",
                       "x-ratelimit-remaining-requests": "not numeric secret"}, path=path)
    persisted = json.loads(path.read_text())
    assert persisted == row
    assert row["http_status"] == 429 and row["provider"] == "opencode-go"
    assert row["quota_headers"] == {"x-ratelimit-reset-requests-day": "2026-10-25T00:00:00+00:00",
                                    "x-ratelimit-limit-requests-day": "20", "retry-after": "60"}
    assert row["output_tokens"] is None
    assert load_observations(path=path) == []
