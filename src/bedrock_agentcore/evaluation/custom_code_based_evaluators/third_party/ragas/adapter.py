"""RAGAS adapter for AgentCore code-based evaluators."""

import asyncio
import inspect
import json
import logging
import math
from typing import Any, Callable, Dict, List, Optional

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)

# Separators used by build_adot_docs() to embed reference/context in user messages.
_REFERENCE_SEPARATOR = "\n\nReference Answer:\n"
_CONTEXT_SEPARATOR = "\n\nContext:\n"

_DEFAULT_THRESHOLD = 0.5


class RAGASAdapter(BaseAdapter):
    """Adapter that runs a RAGAS metric against AgentCore evaluation events.

    Scores samples directly via the metric's ``single_turn_ascore()`` /
    ``multi_turn_ascore()`` (legacy metrics) or ``ascore()`` (collections
    metrics) instead of ``ragas.evaluate()``. This avoids the heavyweight
    ``datasets``/pyarrow dependency (important for Lambda package size) and
    enables multi-turn metrics such as ToolCallAccuracy, ToolCallF1,
    AgentGoalAccuracyWithReference/WithoutReference, and TopicAdherenceScore.

    Example (default span mapping)::

        from ragas.metrics import Faithfulness
        from langchain_aws import ChatBedrockConverse
        from ragas.llms import LangchainLLMWrapper
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        eval_llm = LangchainLLMWrapper(ChatBedrockConverse(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
        ))
        adapter = RAGASAdapter(metric=Faithfulness(), llm=eval_llm)

    Example (custom mapper returning RAGAS sample fields)::

        from typing import Dict, Any

        def my_mapper(ev: EvaluatorInput) -> Dict[str, Any]:
            return {
                "user_input": ev.session_spans[0]["attributes"]["question"],
                "response": ev.session_spans[0]["attributes"]["answer"],
                "retrieved_contexts": ["some context"],
            }

        adapter = RAGASAdapter(
            metric=Faithfulness(),
            llm=eval_llm,
            custom_mapper=my_mapper,
        )

    Multi-turn metrics: the reference for a multi-turn metric is interpreted
    by shape. If the extracted/embedded reference text parses as a JSON list of
    ``{"name": ..., "args": {...}}`` objects it becomes ``reference_tool_calls``
    (ToolCallAccuracy, ToolCallF1); a JSON list of strings becomes
    ``reference_topics`` (TopicAdherenceScore); any other text is used as the
    plain ``reference`` string (AgentGoalAccuracyWithReference). A
    ``custom_mapper`` may also return a ``MultiTurnSample`` (or
    ``SingleTurnSample``) instance directly for full control.
    """

    def __init__(
        self,
        metric: Any,
        custom_mapper: Optional[Callable[[EvaluatorInput], Any]] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A RAGAS metric instance (e.g., Faithfulness(), ToolCallAccuracy()).
                Legacy metrics (ragas.metrics.*) and collections metrics
                (ragas.metrics.collections.*) are both supported.
            custom_mapper: Optional callable that receives the EvaluatorInput and
                returns either a dict of RAGAS sample fields (user_input, response,
                reference, retrieved_contexts, reference_contexts, reference_tool_calls,
                reference_topics) or a SingleTurnSample/MultiTurnSample instance.
                Bypasses default span mapping when provided.
            llm: Optional LLM wrapper to set on the metric (e.g., LangchainLLMWrapper
                for legacy metrics). Required for LLM-based metrics when not using OpenAI.
            embeddings: Optional embeddings wrapper to set on the metric.
                Required for embedding-based metrics (SemanticSimilarity, AnswerCorrectness).
        """
        self.metric = metric
        self.custom_mapper = custom_mapper

        if llm is not None:
            self.metric.llm = llm
        if embeddings is not None and hasattr(self.metric, "embeddings"):
            self.metric.embeddings = embeddings

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def _run(self, evaluator_input: EvaluatorInput) -> EvaluatorOutput:
        """Run the RAGAS metric pipeline."""
        mode = self._scoring_mode()

        component_error = self._validate_components()
        if component_error is not None:
            return component_error

        # Build the sample (or raw field dict for collections metrics)
        sample_or_fields, error = self._build_sample(evaluator_input, mode)
        if error is not None:
            return error

        self._init_metric()

        score = self._score(sample_or_fields, mode)

        if score is None or (isinstance(score, float) and math.isnan(score)):
            return EvaluatorOutput(
                label="Error",
                errorCode="NO_SCORE_FOUND",
                errorMessage=(
                    f"RAGAS metric '{self._metric_name()}' returned no score "
                    f"(got {score!r}). Check that the sample contains the fields "
                    f"the metric requires."
                ),
            )

        threshold = getattr(self.metric, "threshold", None)
        if threshold is None:
            # e.g. SemanticSimilarity sets threshold=None by default
            threshold = _DEFAULT_THRESHOLD
        label = "Pass" if score >= threshold else "Fail"
        reason = f"RAGAS {self._metric_name()}: {score:.4f} (threshold={threshold})"

        return EvaluatorOutput(value=score, label=label, explanation=reason)

    # ------------------------------------------------------------------
    # Metric introspection / validation
    # ------------------------------------------------------------------

    def _metric_name(self) -> str:
        return getattr(self.metric, "name", type(self.metric).__name__)

    def _scoring_mode(self) -> str:
        """Determine how to score with this metric.

        Returns:
            "single_turn": legacy SingleTurnMetric -> single_turn_ascore(sample)
            "multi_turn":  legacy MultiTurnMetric  -> multi_turn_ascore(sample)
            "keyword":     collections metric      -> ascore(**fields)
        """
        try:
            from ragas.metrics.base import MultiTurnMetric, SingleTurnMetric

            is_single = isinstance(self.metric, SingleTurnMetric)
            is_multi = isinstance(self.metric, MultiTurnMetric)
            if is_multi and not is_single:
                return "multi_turn"
            if is_single:
                return "single_turn"
        except ImportError:  # pragma: no cover - ragas always present at runtime
            pass

        if callable(getattr(self.metric, "ascore", None)):
            # Collections metrics (ragas.metrics.collections.*) expose
            # keyword-based ascore(user_input=..., response=..., ...)
            return "keyword"

        raise TypeError(
            f"Unsupported RAGAS metric type {type(self.metric).__name__}: expected a "
            f"legacy metric (SingleTurnMetric/MultiTurnMetric) or a collections "
            f"metric with an ascore() method."
        )

    def _validate_components(self) -> Optional[EvaluatorOutput]:
        """Fail fast with a clear error when a required llm/embeddings is missing.

        Without this, downstream code paths can fall back to constructing an
        OpenAI client, which crashes in Lambda with a confusing error because
        no OPENAI_API_KEY is configured.
        """
        try:
            from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
        except ImportError:  # pragma: no cover
            return None

        if isinstance(self.metric, MetricWithLLM) and self.metric.llm is None:
            return EvaluatorOutput(
                label="Error",
                errorCode="MISSING_COMPONENT",
                errorMessage=(
                    f"RAGAS metric '{self._metric_name()}' requires an LLM but none was "
                    f"provided. Pass llm=LangchainLLMWrapper(ChatBedrockConverse(...)) "
                    f"to RAGASAdapter."
                ),
            )
        if isinstance(self.metric, MetricWithEmbeddings) and self.metric.embeddings is None:
            return EvaluatorOutput(
                label="Error",
                errorCode="MISSING_COMPONENT",
                errorMessage=(
                    f"RAGAS metric '{self._metric_name()}' requires embeddings but none "
                    f"were provided. Pass embeddings=LangchainEmbeddingsWrapper(...) "
                    f"to RAGASAdapter."
                ),
            )
        return None

    def _init_metric(self) -> None:
        """Initialize legacy metrics (validates components, sets run config)."""
        init = getattr(self.metric, "init", None)
        if not callable(init):
            return
        try:
            from ragas.run_config import RunConfig

            init(RunConfig())
        except TypeError:
            # Some metric implementations take no arguments or are not
            # legacy-style init(run_config); scoring works without it.
            logger.debug("Metric init() signature mismatch; skipping", exc_info=True)

    # ------------------------------------------------------------------
    # Field extraction and sample construction
    # ------------------------------------------------------------------

    def _extract_fields(self, evaluator_input: EvaluatorInput) -> Any:
        """Extract RAGAS sample fields from the evaluation event.

        Returns either a dict of sample fields, or whatever the custom_mapper
        returned (which may be a ready-made RAGAS sample instance).
        """
        if self.custom_mapper is not None:
            return self.custom_mapper(evaluator_input)

        result = self._default_extract(evaluator_input)
        if not result.input or not result.actual_output:
            missing: List[str] = []
            if not result.input:
                missing.append("input")
            if not result.actual_output:
                missing.append("actual_output")
            metric_name = type(self.metric).__name__
            return EvaluatorOutput(
                label="Error",
                errorCode="MISSING_REQUIRED_FIELD",
                errorMessage=f"Field(s) {missing} required by {metric_name} but not found in evaluation event. "
                f"Provide a custom_mapper or ensure spans contain the necessary data.",
            )

        # Parse embedded reference and context from the user_input field.
        # build_adot_docs() embeds these as:
        #   "{user_input}\n\nContext:\n{context}\n\nReference Answer:\n{reference}"
        user_input = result.input
        embedded_reference: Optional[str] = None
        embedded_context: Optional[str] = None

        # Extract reference (must come after context if both present)
        if _REFERENCE_SEPARATOR in user_input:
            user_input, embedded_reference = user_input.split(_REFERENCE_SEPARATOR, 1)

        # Extract context from whatever remains as user_input
        if _CONTEXT_SEPARATOR in user_input:
            user_input, embedded_context = user_input.split(_CONTEXT_SEPARATOR, 1)

        fields: Dict[str, Any] = {
            "user_input": user_input,
            "response": result.actual_output,
        }

        # Reference priority: reference_inputs > embedded > assertions
        if result.expected_output:
            fields["reference"] = result.expected_output
        elif embedded_reference:
            fields["reference"] = embedded_reference
        elif result.assertions:
            fields["reference"] = "\n".join(result.assertions)

        # Retrieval context priority: span tool results > embedded > SpanMapResult.context
        if result.retrieval_context:
            fields["retrieved_contexts"] = result.retrieval_context
            fields["reference_contexts"] = result.retrieval_context
        elif embedded_context:
            fields["retrieved_contexts"] = [embedded_context]
            fields["reference_contexts"] = [embedded_context]
        elif result.context:
            fields["retrieved_contexts"] = result.context
            fields["reference_contexts"] = result.context

        # Tool calls made by the agent (used for multi-turn metrics)
        if result.tools_called:
            fields["_tools_called"] = result.tools_called
        if result.expected_tools:
            fields["_expected_tools"] = result.expected_tools

        return fields

    def _build_sample(self, evaluator_input: EvaluatorInput, mode: str):
        """Build the object to score: a RAGAS sample or a field dict.

        Returns:
            (sample_or_fields, error): exactly one is non-None.
        """
        extracted = self._extract_fields(evaluator_input)
        if isinstance(extracted, EvaluatorOutput):
            return None, extracted

        # custom_mapper may hand back a ready-made sample
        if self._is_ragas_sample(extracted):
            return extracted, None

        if not isinstance(extracted, dict):
            return None, EvaluatorOutput(
                label="Error",
                errorCode="INVALID_MAPPER_RESULT",
                errorMessage=(
                    f"custom_mapper must return a dict of RAGAS sample fields or a "
                    f"SingleTurnSample/MultiTurnSample instance, got {type(extracted).__name__}."
                ),
            )

        if mode == "multi_turn":
            return self._build_multi_turn_sample(extracted), None
        if mode == "single_turn":
            return self._build_single_turn_sample(extracted), None
        # keyword mode (collections metrics): pass fields directly
        return extracted, None

    @staticmethod
    def _is_ragas_sample(obj: Any) -> bool:
        try:
            from ragas.dataset_schema import MultiTurnSample, SingleTurnSample

            return isinstance(obj, (SingleTurnSample, MultiTurnSample))
        except ImportError:  # pragma: no cover
            return False

    def _build_single_turn_sample(self, fields: Dict[str, Any]):
        from ragas.dataset_schema import SingleTurnSample

        allowed = set(SingleTurnSample.model_fields)
        kwargs = {k: v for k, v in fields.items() if k in allowed and v is not None}
        return SingleTurnSample(**kwargs)

    def _build_multi_turn_sample(self, fields: Dict[str, Any]):
        from ragas.dataset_schema import MultiTurnSample
        from ragas.messages import AIMessage, HumanMessage

        user_input = fields.get("user_input")
        if isinstance(user_input, str):
            # Build a minimal conversation: the user's question and the agent's
            # response, including any tool calls the agent made.
            tool_calls = self._coerce_tool_calls(fields.get("_tools_called"))
            messages = [
                HumanMessage(content=user_input),
                AIMessage(content=fields.get("response") or "", tool_calls=tool_calls or None),
            ]
        else:
            # custom_mapper provided ragas message objects directly
            messages = user_input

        kwargs: Dict[str, Any] = {"user_input": messages}

        # Interpret the reference by shape (see class docstring)
        reference = fields.get("reference")
        if isinstance(reference, str) and reference.strip():
            interpreted_key, interpreted_value = self._interpret_multi_turn_reference(reference)
            kwargs[interpreted_key] = interpreted_value
        elif fields.get("_expected_tools"):
            kwargs["reference_tool_calls"] = self._coerce_tool_calls(fields["_expected_tools"])

        # Explicit multi-turn fields from custom mappers take precedence
        allowed = set(MultiTurnSample.model_fields)
        for key in ("reference_tool_calls", "reference_topics"):
            if key in fields and fields[key] is not None:
                value = fields[key]
                if key == "reference_tool_calls":
                    value = self._coerce_tool_calls(value)
                kwargs[key] = value

        kwargs = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        return MultiTurnSample(**kwargs)

    def _interpret_multi_turn_reference(self, reference: str):
        """Interpret a reference string for multi-turn metrics by shape.

        - JSON list of {"name": ..., "args": {...}} -> ("reference_tool_calls", [ToolCall])
        - JSON list of strings -> ("reference_topics", [str])
        - anything else -> ("reference", str)
        """
        try:
            parsed = json.loads(reference)
        except (json.JSONDecodeError, ValueError):
            return "reference", reference

        if isinstance(parsed, list) and parsed:
            if all(isinstance(item, dict) and "name" in item for item in parsed):
                return "reference_tool_calls", self._coerce_tool_calls(parsed)
            if all(isinstance(item, str) for item in parsed):
                return "reference_topics", parsed

        return "reference", reference

    @staticmethod
    def _coerce_tool_calls(items: Optional[List[Any]]) -> List[Any]:
        """Coerce a list of dicts (or ToolCall instances) to ragas ToolCall objects."""
        if not items:
            return []
        from ragas.messages import ToolCall

        coerced = []
        for item in items:
            if isinstance(item, ToolCall):
                coerced.append(item)
            elif isinstance(item, dict) and item.get("name"):
                args = item.get("args") or item.get("arguments") or item.get("input_parameters") or {}
                coerced.append(ToolCall(name=item["name"], args=args))
        return coerced

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, sample_or_fields: Any, mode: str) -> Optional[float]:
        """Score the sample with the metric, returning a float."""
        if mode == "keyword":
            # Collections metrics: ascore(**fields), pass only accepted kwargs
            sig = inspect.signature(self.metric.ascore)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            fields = {k: v for k, v in sample_or_fields.items() if not k.startswith("_")}
            if not accepts_kwargs:
                fields = {k: v for k, v in fields.items() if k in sig.parameters}
            result = self._run_async(self.metric.ascore(**fields))
            value = getattr(result, "value", result)
            return float(value) if value is not None else None

        # Legacy metrics: sample-based scoring.
        # A sample from a custom_mapper may not match the metric's turn type;
        # let type errors surface via the base error handler.
        if mode == "multi_turn":
            score = self._run_async(self.metric.multi_turn_ascore(sample_or_fields))
        else:
            score = self._run_async(self.metric.single_turn_ascore(sample_or_fields))
        return float(score) if score is not None else None

    @staticmethod
    def _run_async(coro):
        """Run a coroutine to completion from sync code (Lambda handlers are sync)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "RAGASAdapter cannot be invoked from within a running event loop. "
            "Call it from a synchronous Lambda handler."
        )
