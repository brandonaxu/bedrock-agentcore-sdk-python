"""Tests for RAGASAdapter (direct *_ascore scoring, no datasets/evaluate)."""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import pytest

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.messages import AIMessage, HumanMessage, ToolCall
from ragas.metrics import Faithfulness, SemanticSimilarity
from ragas.metrics.base import MultiTurnMetric, SingleTurnMetric

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas.adapter import RAGASAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spans(user_content: str = "What is AI?", response: str = "AI is artificial intelligence."):
    """CloudWatch split-format spans (metadata entry + log entry)."""
    return [
        {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer"},
            "name": "invoke_agent",
            "kind": "INTERNAL",
            "startTimeUnixNano": 1000000000,
            "endTimeUnixNano": 2000000000,
            "attributes": {"gen_ai.operation.name": "invoke_agent", "session.id": "test-session"},
            "status": {"code": "UNSET"},
        },
        {
            "traceId": "t1",
            "spanId": "s1",
            "scope": {"name": "strands.telemetry.tracer"},
            "timeUnixNano": 2000000000,
            "observedTimeUnixNano": 2000000001,
            "severityNumber": 9,
            "body": {
                "input": {
                    "messages": [
                        {"role": "user", "content": {"content": json.dumps([{"text": user_content}])}}
                    ]
                },
                "output": {"messages": [{"role": "assistant", "content": {"message": response}}]},
            },
        },
    ]


def _make_evaluator_input(spans=None, user_content="What is AI?", response="AI is artificial intelligence.", **kwargs):
    if spans is None:
        spans = _make_spans(user_content=user_content, response=response)
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fake metrics (controlled scores; subclass the real ragas base classes so the
# adapter's isinstance-based mode detection is exercised)
# ---------------------------------------------------------------------------


@dataclass
class FakeSingleTurnMetric(SingleTurnMetric):
    name: str = "fake_single_turn"
    threshold: Optional[float] = 0.5
    fixed_score: float = 0.9
    captured_sample: Optional[SingleTurnSample] = field(default=None, repr=False)

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks) -> float:
        self.captured_sample = sample
        return self.fixed_score


@dataclass
class FakeNoThresholdMetric(SingleTurnMetric):
    """Metric with no threshold attribute at all."""

    name: str = "fake_no_threshold"
    fixed_score: float = 0.6

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks) -> float:
        return self.fixed_score


@dataclass
class FakeRaisingMetric(SingleTurnMetric):
    name: str = "fake_raising"

    def init(self, run_config):
        pass

    async def _single_turn_ascore(self, sample, callbacks) -> float:
        raise RuntimeError("RAGAS timeout")


@dataclass
class FakeMultiTurnMetric(MultiTurnMetric):
    name: str = "fake_multi_turn"
    threshold: Optional[float] = 0.5
    fixed_score: float = 1.0
    captured_sample: Optional[MultiTurnSample] = field(default=None, repr=False)

    def init(self, run_config):
        pass

    async def _multi_turn_ascore(self, sample, callbacks) -> float:
        self.captured_sample = sample
        return self.fixed_score


# ---------------------------------------------------------------------------
# Success paths (single-turn)
# ---------------------------------------------------------------------------


class TestRAGASAdapterSuccess:
    def test_returns_pass_when_score_above_threshold(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=0.9, threshold=0.7))

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.9
        assert result.label == "Pass"

    def test_returns_fail_when_score_below_threshold(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=0.3, threshold=0.7))

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_returns_pass_at_exact_threshold(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=0.7, threshold=0.7))

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    def test_real_exact_match_end_to_end(self):
        """Real ragas ExactMatch through spans with embedded reference."""
        from ragas.metrics import ExactMatch

        adapter = RAGASAdapter(metric=ExactMatch())
        evaluator_input = _make_evaluator_input(
            user_content="What is 2+2?\n\nReference Answer:\n4",
            response="4",
        )

        result = adapter(evaluator_input)

        assert result.value == 1.0
        assert result.label == "Pass"
        assert result.errorCode is None

    def test_custom_mapper_dict(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {
                "user_input": "mapped input",
                "response": "mapped output",
                "retrieved_contexts": ["some context"],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9
        assert metric.captured_sample.user_input == "mapped input"
        assert metric.captured_sample.response == "mapped output"
        assert metric.captured_sample.retrieved_contexts == ["some context"]

    def test_custom_mapper_returns_sample_instance(self):
        metric = FakeSingleTurnMetric()
        sample = SingleTurnSample(user_input="direct", response="sample")
        adapter = RAGASAdapter(metric=metric, custom_mapper=lambda ev: sample)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9
        assert metric.captured_sample is sample

    def test_reference_inputs_populates_reference(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = _make_evaluator_input(
            reference_inputs=[{"expectedResponse": {"text": "AI stands for artificial intelligence."}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.9
        assert metric.captured_sample.reference == "AI stands for artificial intelligence."

    def test_llm_override_sets_metric_llm(self):
        metric = FakeSingleTurnMetric()
        sentinel = object()
        RAGASAdapter(metric=metric, llm=sentinel)

        assert metric.llm is sentinel

    def test_embeddings_override_sets_metric_embeddings(self):
        metric = SemanticSimilarity()
        sentinel = object()
        RAGASAdapter(metric=metric, embeddings=sentinel)

        assert metric.embeddings is sentinel


# ---------------------------------------------------------------------------
# Embedded reference / context parsing (build_adot_docs format)
# ---------------------------------------------------------------------------


class TestRAGASAdapterEmbeddedParsing:
    def test_parses_embedded_reference(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input(user_content="What is 2+2?\n\nReference Answer:\n4"))

        assert result.value == 0.9
        assert metric.captured_sample.user_input == "What is 2+2?"
        assert metric.captured_sample.reference == "4"

    def test_parses_embedded_context(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(
            _make_evaluator_input(user_content="What is AI?\n\nContext:\nAI is a branch of computer science.")
        )

        assert result.value == 0.9
        assert metric.captured_sample.user_input == "What is AI?"
        assert metric.captured_sample.retrieved_contexts == ["AI is a branch of computer science."]
        assert metric.captured_sample.reference_contexts == ["AI is a branch of computer science."]

    def test_parses_combined_context_and_reference(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        user_content = (
            "What is AI?\n\nContext:\nAI is computer science.\n\nReference Answer:\nArtificial Intelligence"
        )
        result = adapter(_make_evaluator_input(user_content=user_content))

        assert result.value == 0.9
        assert metric.captured_sample.user_input == "What is AI?"
        assert metric.captured_sample.retrieved_contexts == ["AI is computer science."]
        assert metric.captured_sample.reference == "Artificial Intelligence"

    def test_no_markers_leaves_input_unchanged(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())  # plain "What is AI?"

        assert result.value == 0.9
        assert metric.captured_sample.user_input == "What is AI?"
        assert metric.captured_sample.reference is None

    def test_reference_inputs_takes_precedence_over_embedded(self):
        metric = FakeSingleTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = _make_evaluator_input(
            user_content="Q\n\nReference Answer:\nembedded ref",
            reference_inputs=[{"expectedResponse": {"text": "service-provided ref"}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.9
        assert metric.captured_sample.reference == "service-provided ref"


# ---------------------------------------------------------------------------
# Multi-turn metrics
# ---------------------------------------------------------------------------


class TestRAGASAdapterMultiTurn:
    def test_real_tool_call_accuracy_with_custom_mapper_sample(self):
        """Real ToolCallAccuracy scored through the adapter via a MultiTurnSample."""
        from ragas.metrics import ToolCallAccuracy

        sample = MultiTurnSample(
            user_input=[
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4", tool_calls=[ToolCall(name="calculator", args={"expression": "2+2"})]),
            ],
            reference_tool_calls=[ToolCall(name="calculator", args={"expression": "2+2"})],
        )
        adapter = RAGASAdapter(metric=ToolCallAccuracy(), custom_mapper=lambda ev: sample)

        result = adapter(_make_evaluator_input())

        assert result.errorCode is None
        assert result.value == 1.0
        assert result.label == "Pass"

    def test_real_tool_call_f1_with_custom_mapper_dict(self):
        """Real ToolCallF1 via a custom mapper returning multi-turn fields."""
        from ragas.metrics import ToolCallF1

        adapter = RAGASAdapter(
            metric=ToolCallF1(),
            custom_mapper=lambda ev: {
                "user_input": [
                    HumanMessage(content="Weather in Tokyo?"),
                    AIMessage(content="Sunny", tool_calls=[ToolCall(name="weather", args={"city": "Tokyo"})]),
                ],
                "reference_tool_calls": [{"name": "weather", "args": {"city": "Tokyo"}}],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.errorCode is None
        assert result.value == 1.0

    def test_builds_conversation_from_default_extraction(self):
        """Plain span text becomes a Human/AI message conversation."""
        metric = FakeMultiTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 1.0
        messages = metric.captured_sample.user_input
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "What is AI?"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "AI is artificial intelligence."

    def test_embedded_json_reference_becomes_reference_tool_calls(self):
        metric = FakeMultiTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        ref = json.dumps([{"name": "calculator", "args": {"expression": "2+2"}}])
        result = adapter(_make_evaluator_input(user_content=f"What is 2+2?\n\nReference Answer:\n{ref}"))

        assert result.value == 1.0
        tool_calls = metric.captured_sample.reference_tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "calculator"
        assert tool_calls[0].args == {"expression": "2+2"}

    def test_embedded_json_string_list_becomes_reference_topics(self):
        metric = FakeMultiTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        ref = json.dumps(["science", "technology"])
        result = adapter(_make_evaluator_input(user_content=f"Tell me about AI\n\nReference Answer:\n{ref}"))

        assert result.value == 1.0
        assert metric.captured_sample.reference_topics == ["science", "technology"]

    def test_embedded_plain_text_reference_stays_reference(self):
        metric = FakeMultiTurnMetric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(
            _make_evaluator_input(user_content="Book a table\n\nReference Answer:\nTable booked at 7pm")
        )

        assert result.value == 1.0
        assert metric.captured_sample.reference == "Table booked at 7pm"


# ---------------------------------------------------------------------------
# Component validation (missing llm / embeddings)
# ---------------------------------------------------------------------------


class TestRAGASAdapterComponentValidation:
    def test_llm_metric_without_llm_returns_missing_component(self):
        adapter = RAGASAdapter(metric=Faithfulness())  # no llm

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_COMPONENT"
        assert "requires an LLM" in result.errorMessage
        assert "faithfulness" in result.errorMessage

    def test_embeddings_metric_without_embeddings_returns_missing_component(self):
        adapter = RAGASAdapter(metric=SemanticSimilarity())  # no embeddings

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "MISSING_COMPONENT"
        assert "requires embeddings" in result.errorMessage


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRAGASAdapterErrors:
    def test_no_agent_spans_returns_error(self):
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "attributes": {"gen_ai.operation.name": "chat"},
                "span_events": [],
            }
        ]
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric())

        result = adapter(_make_evaluator_input(spans=spans))

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode == "FIELD_EXTRACTION_ERROR"
        assert result.label == "Error"

    def test_metric_execution_exception_returns_error(self):
        adapter = RAGASAdapter(metric=FakeRaisingMetric())

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "RAGAS timeout" in result.errorMessage

    def test_nan_score_returns_no_score_found(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=float("nan")))

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "NO_SCORE_FOUND"
        assert "fake_single_turn" in result.errorMessage

    def test_invalid_mapper_result_returns_error(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(), custom_mapper=lambda ev: "not a dict")

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "INVALID_MAPPER_RESULT"

    def test_never_raises(self):
        adapter = RAGASAdapter(metric=FakeRaisingMetric())

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is not None

    def test_missing_input_returns_error(self):
        """Spans with no extractable input should return an extraction error."""
        spans = [
            {
                "traceId": "t1",
                "spanId": "s1",
                "scope": {"name": "strands.telemetry.tracer", "version": ""},
                "attributes": {"gen_ai.operation.name": "invoke_agent"},
                "span_events": [
                    {
                        "body": {
                            "output": {"messages": [{"role": "assistant", "content": "answer"}]},
                        }
                    }
                ],
            }
        ]
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric())

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.errorMessage


# ---------------------------------------------------------------------------
# Threshold handling
# ---------------------------------------------------------------------------


class TestRAGASAdapterThreshold:
    def test_threshold_none_defaults_to_half_pass(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=0.6, threshold=None))

        result = adapter(_make_evaluator_input())

        assert result.value == 0.6
        assert result.label == "Pass"  # 0.6 >= 0.5
        assert "threshold=0.5" in result.explanation

    def test_threshold_none_below_default_fails(self):
        adapter = RAGASAdapter(metric=FakeSingleTurnMetric(fixed_score=0.3, threshold=None))

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    def test_missing_threshold_attribute_defaults(self):
        adapter = RAGASAdapter(metric=FakeNoThresholdMetric(fixed_score=0.6))

        result = adapter(_make_evaluator_input())

        assert result.value == 0.6
        assert result.label == "Pass"


# ---------------------------------------------------------------------------
# Import hygiene
# ---------------------------------------------------------------------------


class TestImportHygiene:
    def test_adapter_importable_without_strands_evals(self):
        """RAGASAdapter must import when strands-evals is absent (custom_mapper flows)."""
        code = (
            "import sys; sys.modules['strands_evals'] = None; "
            "from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas "
            "import RAGASAdapter; print('ok')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout

    def test_adapter_module_does_not_import_datasets(self):
        """The datasets/pyarrow stack must not be required by the adapter module."""
        code = (
            "import sys; sys.modules['datasets'] = None; "
            "from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas "
            "import RAGASAdapter; print('ok')"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

        assert proc.returncode == 0, proc.stderr
        assert "ok" in proc.stdout
