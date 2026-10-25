from dataclasses import replace
import pytest
from kerdoios.routing_objective import plan_routing
from kerdoios.types import ResourceOffer, CapabilityProfile, Capacity, Economics, Telemetry, WorkRequirement


def offer(name):
    return ResourceOffer(name, name, "llm", "synthetic", False,
        CapabilityProfile(coding=.9, reasoning=.9, tool_use=.9), Capacity(),
        Economics(input_token_price=0, output_token_price=0), Telemetry())


def metrics(**changes):
    return {"ttft_seconds": 1., "tokens_per_second": 100., "burn": .01,
            "remaining": 100., "time_to_reset_hours": 1., "p_accept": .9, **changes}


def test_exact_agreed_formula_and_minimization():
    req = WorkRequirement(coding=.5, estimated_input_tokens=20, estimated_output_tokens=10)
    result = plan_routing([offer("groq"), offer("slow")], req,
                          {"groq": metrics(), "slow": metrics(tokens_per_second=1)})
    best = result["ranked"][0]
    assert best.offer.provider == "groq"
    assert best.latency == pytest.approx(1 + 10/100)
    assert best.pace == pytest.approx(.01 * 30/(100/1))
    assert best.score == pytest.approx((1 + 10/100) * (1 + .01*30/(100/1))/.9)
    assert best.inputs["lambda_evidence"]["status"] == "unmeasured"


def test_unknown_does_not_become_zero_or_default_latency():
    result = plan_routing([offer("unknown")], WorkRequirement(coding=.5), {})
    assert result["ranked"] == []
    assert result["unscored"][0]["reason"] == "unknown_objective_inputs"


def test_hard_capability_floor_survives_new_objective():
    weak = replace(offer("fast"), capabilities=CapabilityProfile(coding=.1))
    result = plan_routing([weak], WorkRequirement(coding=.8), {"fast": metrics(ttft_seconds=0)})
    assert not result["ranked"] and not result["unscored"]
    assert result["rejections"]


def test_acceptance_blend_uses_existing_five_observation_floor():
    req = WorkRequirement(coding=.5, capability_id="coding.small")
    observed = {"n": 4, "accepted": 0, "task_class": "coding.small", "verifier_source": "synthetic"}
    first = plan_routing([offer("x")], req, {"x": metrics(observed_acceptance=observed)})["ranked"][0]
    assert first.p_accept == .9
    observed = {**observed, "n": 5}
    second = plan_routing([offer("x")], req, {"x": metrics(observed_acceptance=observed)})["ranked"][0]
    assert second.p_accept < first.p_accept


@pytest.mark.parametrize("value", [None, -1, float("nan"), 0])
def test_invalid_tps_unscored(value):
    result = plan_routing([offer("x")], WorkRequirement(coding=.5), {"x": metrics(tokens_per_second=value)})
    assert not result["ranked"]


def test_nondefault_lambda_requires_measurement_source():
    with pytest.raises(ValueError, match="measured"):
        plan_routing([], WorkRequirement(), {}, lambda_value=2)
