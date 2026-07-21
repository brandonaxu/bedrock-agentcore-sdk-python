"""Tests for RAGASAdapter."""

from unittest.mock import MagicMock, patch

import pytest

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas.adapter import RAGASAdapter


def _make_evaluator_input(spans=None):
    """Build an EvaluatorInput with agent-level spans (CloudWatch split format)."""
    if spans is None:
        spans = [
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
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}}]},
                    "output": {"messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]},
                },
            },
        ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
    )


def _mock_ragas_metric(name="faithfulness", threshold=0.5):
    """Create a mock RAGAS metric with a .name attribute."""
    metric = MagicMock()
    metric.name = name
    metric.threshold = threshold
    metric.required_columns = {"SINGLE_TURN": {"user_input", "response"}}
    return metric


def _mock_evaluate_result(metric_name, score=0.85):
    """Create a mock ragas evaluate() return that produces a DataFrame with the score."""
    import pandas as pd

    df = pd.DataFrame({
        "user_input": ["What is AI?"],
        "response": ["AI is artificial intelligence."],
        metric_name: [score],
    })

    result = MagicMock()
    result.to_pandas.return_value = df
    return result


RAGAS_MODULE = "bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas.adapter"


class TestRAGASAdapterSuccess:
    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_returns_pass_when_score_above_threshold(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.9)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.7)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.9
        assert result.label == "Pass"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_returns_fail_when_score_below_threshold(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.3)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.7)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_returns_pass_at_exact_threshold(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.7)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.7)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_handles_column_name_with_suffix(self, mock_evaluate):
        """RAGAS may produce columns like 'factual_correctness(mode=f1)'."""
        import pandas as pd

        df = pd.DataFrame({
            "user_input": ["test"],
            "response": ["answer"],
            "factual_correctness(mode=f1)": [0.85],
        })
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df
        mock_evaluate.return_value = mock_result

        metric = _mock_ragas_metric(name="factual_correctness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.85
        assert result.label == "Pass"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_custom_mapper(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.9)
        metric = _mock_ragas_metric(name="faithfulness")
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

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_reference_inputs_populates_reference(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("answer_correctness", 0.8)
        metric = _mock_ragas_metric(name="answer_correctness")
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=[
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
                        "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "What is AI?"}]'}}]},
                        "output": {"messages": [{"role": "assistant", "content": {"message": "AI is artificial intelligence."}}]},
                    },
                },
            ],
            target_trace_id="t1",
            reference_inputs=[{"expectedResponse": {"text": "AI stands for artificial intelligence."}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.8

    def test_llm_override_sets_metric_llm(self):
        metric = _mock_ragas_metric()
        mock_llm = MagicMock()
        RAGASAdapter(metric=metric, llm=mock_llm)

        assert metric.llm == mock_llm

    def test_embeddings_override_sets_metric_embeddings(self):
        metric = _mock_ragas_metric()
        metric.embeddings = None
        mock_embeddings = MagicMock()
        RAGASAdapter(metric=metric, embeddings=mock_embeddings)

        assert metric.embeddings == mock_embeddings


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
        metric = _mock_ragas_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode == "FIELD_EXTRACTION_ERROR"
        assert result.label == "Error"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_metric_execution_exception_returns_error(self, mock_evaluate):
        mock_evaluate.side_effect = RuntimeError("RAGAS timeout")
        metric = _mock_ragas_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "METRIC_ERROR"
        assert "RAGAS timeout" in result.errorMessage

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_no_score_column_returns_error(self, mock_evaluate):
        import pandas as pd

        df = pd.DataFrame({"user_input": ["test"], "response": ["answer"]})
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = df
        mock_evaluate.return_value = mock_result

        metric = _mock_ragas_metric(name="nonexistent_metric")
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.errorCode == "NO_SCORE_FOUND"
        assert "nonexistent_metric" in result.errorMessage

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_never_raises(self, mock_evaluate):
        mock_evaluate.side_effect = Exception("unexpected catastrophic failure")
        metric = _mock_ragas_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.errorCode is not None


class TestRAGASAdapterEdgeCases:
    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_metric_score_zero(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.0)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.0
        assert result.label == "Fail"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_metric_score_one(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 1.0)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 1.0
        assert result.label == "Pass"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_default_threshold_when_missing(self, mock_evaluate):
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.6)
        metric = _mock_ragas_metric(name="faithfulness")
        del metric.threshold
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.label == "Pass"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_custom_mapper_with_ragas_columns(self, mock_evaluate):
        """custom_mapper returns RAGAS-native column names directly."""
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.9)
        metric = _mock_ragas_metric(name="faithfulness")
        adapter = RAGASAdapter(
            metric=metric,
            custom_mapper=lambda ev: {
                "user_input": "question",
                "response": "answer",
                "retrieved_contexts": ["context 1", "context 2"],
            },
        )

        result = adapter(_make_evaluator_input())

        assert result.value == 0.9

    def test_missing_input_returns_error(self):
        """Spans with no extractable input should return MISSING_REQUIRED_FIELD."""
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
        metric = _mock_ragas_metric()
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input(spans=spans))

        assert result.errorCode in ("MISSING_REQUIRED_FIELD", "FIELD_EXTRACTION_ERROR")
        assert result.errorMessage


def _make_evaluator_input_with_embedded_text(user_content: str):
    """Build an EvaluatorInput with a specific user message content (for embedded parsing tests)."""
    spans = [
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
                "input": {"messages": [{"role": "user", "content": {"content": f'[{{"text": "{user_content}"}}]'}}]},
                "output": {"messages": [{"role": "assistant", "content": {"message": "The answer is 42."}}]},
            },
        },
    ]
    return EvaluatorInput(
        evaluation_level="TRACE",
        session_spans=spans,
        target_trace_id="t1",
    )


class TestRAGASAdapterEmbeddedReferenceParsing:
    """Tests for parsing reference text embedded in user_input via build_adot_docs() format."""

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_parses_embedded_reference_from_user_input(self, mock_evaluate):
        """Reference embedded as '{question}\\n\\nReference Answer:\\n{reference}' is extracted."""
        mock_evaluate.return_value = _mock_evaluate_result("exact_match", 1.0)
        metric = _mock_ragas_metric(name="exact_match", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        user_content = "What is 2+2?\\n\\nReference Answer:\\n4"
        evaluator_input = _make_evaluator_input_with_embedded_text(user_content)
        result = adapter(evaluator_input)

        assert result.value == 1.0
        # Verify the dataset passed to evaluate had the reference extracted
        call_kwargs = mock_evaluate.call_args[1]
        dataset = call_kwargs["dataset"]
        assert "reference" in dataset.column_names
        assert dataset["reference"][0] == "4"
        assert dataset["user_input"][0] == "What is 2+2?"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_parses_embedded_context_from_user_input(self, mock_evaluate):
        """Context embedded as '{question}\\n\\nContext:\\n{context}' is extracted."""
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.9)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        user_content = "What is AI?\\n\\nContext:\\nAI is a branch of computer science."
        evaluator_input = _make_evaluator_input_with_embedded_text(user_content)
        result = adapter(evaluator_input)

        assert result.value == 0.9
        call_kwargs = mock_evaluate.call_args[1]
        dataset = call_kwargs["dataset"]
        assert "retrieved_contexts" in dataset.column_names
        assert dataset["retrieved_contexts"][0] == ["AI is a branch of computer science."]
        assert "reference_contexts" in dataset.column_names
        assert dataset["reference_contexts"][0] == ["AI is a branch of computer science."]
        assert dataset["user_input"][0] == "What is AI?"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_parses_combined_context_and_reference(self, mock_evaluate):
        """Combined format '{question}\\n\\nContext:\\n{ctx}\\n\\nReference Answer:\\n{ref}' is parsed."""
        mock_evaluate.return_value = _mock_evaluate_result("context_precision", 0.75)
        metric = _mock_ragas_metric(name="context_precision", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        user_content = "What is AI?\\n\\nContext:\\nAI is computer science.\\n\\nReference Answer:\\nArtificial Intelligence"
        evaluator_input = _make_evaluator_input_with_embedded_text(user_content)
        result = adapter(evaluator_input)

        assert result.value == 0.75
        call_kwargs = mock_evaluate.call_args[1]
        dataset = call_kwargs["dataset"]
        assert dataset["user_input"][0] == "What is AI?"
        assert dataset["retrieved_contexts"][0] == ["AI is computer science."]
        assert dataset["reference"][0] == "Artificial Intelligence"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_no_embedded_markers_leaves_input_unchanged(self, mock_evaluate):
        """Plain user_input without markers passes through unchanged."""
        mock_evaluate.return_value = _mock_evaluate_result("faithfulness", 0.8)
        metric = _mock_ragas_metric(name="faithfulness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        evaluator_input = _make_evaluator_input()  # plain "What is AI?"
        result = adapter(evaluator_input)

        assert result.value == 0.8
        call_kwargs = mock_evaluate.call_args[1]
        dataset = call_kwargs["dataset"]
        assert dataset["user_input"][0] == "What is AI?"
        assert "reference" not in dataset.column_names

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_reference_inputs_takes_precedence_over_embedded(self, mock_evaluate):
        """reference_inputs from the service override embedded reference."""
        mock_evaluate.return_value = _mock_evaluate_result("answer_correctness", 0.9)
        metric = _mock_ragas_metric(name="answer_correctness", threshold=0.5)
        adapter = RAGASAdapter(metric=metric)

        # Span has embedded reference, but reference_inputs also provided
        spans = [
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
                    "input": {"messages": [{"role": "user", "content": {"content": '[{"text": "Q\\n\\nReference Answer:\\nembedded ref"}]'}}]},
                    "output": {"messages": [{"role": "assistant", "content": {"message": "answer"}}]},
                },
            },
        ]
        evaluator_input = EvaluatorInput(
            evaluation_level="TRACE",
            session_spans=spans,
            target_trace_id="t1",
            reference_inputs=[{"expectedResponse": {"text": "service-provided ref"}}],
        )

        result = adapter(evaluator_input)

        assert result.value == 0.9
        call_kwargs = mock_evaluate.call_args[1]
        dataset = call_kwargs["dataset"]
        # reference_inputs takes precedence
        assert dataset["reference"][0] == "service-provided ref"


class TestRAGASAdapterThresholdNone:
    """Tests for handling metrics where threshold is explicitly None."""

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_threshold_none_defaults_to_half(self, mock_evaluate):
        """When metric.threshold is explicitly None, default to 0.5."""
        mock_evaluate.return_value = _mock_evaluate_result("semantic_similarity", 0.6)
        metric = _mock_ragas_metric(name="semantic_similarity")
        metric.threshold = None  # Explicitly set to None like SemanticSimilarity does
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.6
        assert result.label == "Pass"  # 0.6 >= 0.5

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_threshold_none_score_below_default(self, mock_evaluate):
        """Score below 0.5 with threshold=None results in Fail."""
        mock_evaluate.return_value = _mock_evaluate_result("semantic_similarity", 0.3)
        metric = _mock_ragas_metric(name="semantic_similarity")
        metric.threshold = None
        adapter = RAGASAdapter(metric=metric)

        result = adapter(_make_evaluator_input())

        assert result.value == 0.3
        assert result.label == "Fail"

    @patch(f"{RAGAS_MODULE}.evaluate")
    def test_threshold_none_does_not_crash(self, mock_evaluate):
        """Ensure no TypeError when comparing score >= None."""
        mock_evaluate.return_value = _mock_evaluate_result("semantic_similarity", 0.85)
        metric = _mock_ragas_metric(name="semantic_similarity")
        metric.threshold = None
        adapter = RAGASAdapter(metric=metric)

        # This would previously raise TypeError: '>=' not supported between float and NoneType
        result = adapter(_make_evaluator_input())

        assert isinstance(result, EvaluatorOutput)
        assert result.value == 0.85
        assert result.label == "Pass"
        assert "threshold=0.5" in result.explanation
