"""RAGAS adapter for AgentCore code-based evaluators."""

import logging
from typing import Any, Callable, Dict, Optional

from datasets import Dataset
from ragas import evaluate

from bedrock_agentcore.evaluation.custom_code_based_evaluators.models import EvaluatorInput, EvaluatorOutput
from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.base import BaseAdapter

logger = logging.getLogger(__name__)


class RAGASAdapter(BaseAdapter):
    """Adapter that runs a RAGAS metric against AgentCore evaluation events.

    Example::

        from ragas.metrics import Faithfulness
        from langchain_aws import ChatBedrockConverse
        from ragas.llms import LangchainLLMWrapper
        from bedrock_agentcore.evaluation.custom_code_based_evaluators.third_party.ragas import RAGASAdapter

        eval_llm = LangchainLLMWrapper(ChatBedrockConverse(
            model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            region_name="us-east-1",
        ))
        adapter = RAGASAdapter(metric=Faithfulness(), llm=eval_llm)
    """

    def __init__(
        self,
        metric: Any,
        field_mapper: Optional[Callable[[EvaluatorInput], Dict[str, Any]]] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        """Initialize the adapter.

        Args:
            metric: A RAGAS metric instance (e.g., Faithfulness(), ContextRecall()).
                Must have a .name attribute and support ragas.evaluate().
            field_mapper: Optional callable that receives the EvaluatorInput and
                returns a dict with RAGAS-compatible keys. Bypasses default span
                parsing when provided.
            llm: Optional LLM wrapper to set on the metric (e.g., LangchainLLMWrapper).
                Required for most RAGAS metrics when not using OpenAI.
            embeddings: Optional embeddings wrapper to set on the metric.
                Required for embedding-based metrics (AnswerSimilarity, AnswerCorrectness).
        """
        super().__init__(field_mapper=field_mapper)
        self.metric = metric

        if llm is not None:
            self.metric.llm = llm
        if embeddings is not None and hasattr(self.metric, "embeddings"):
            self.metric.embeddings = embeddings

    def validate_fields(self, fields: Dict[str, Any]) -> None:
        """Validate that required fields for this RAGAS metric are present.

        The span parser produces fields with keys: 'input', 'actual_output',
        'expected_output', 'context', 'retrieval_context'. We map these to
        RAGAS column names in execute().
        """
        if not fields.get("input") and not fields.get("actual_output"):
            metric_name = type(self.metric).__name__
            raise ValueError(
                f"Neither 'input' nor 'actual_output' found in evaluation event. "
                f"{metric_name} requires at minimum a user input and agent response. "
                f"Provide a field_mapper or ensure spans contain the necessary data."
            )

    def execute(self, fields: Dict[str, Any]) -> EvaluatorOutput:
        """Run the RAGAS metric and return formatted results.

        Maps the standard span parser fields to RAGAS dataset columns:
            input          → user_input
            actual_output  → response
            expected_output → reference
            retrieval_context → retrieved_contexts
            context        → reference_contexts (fallback)
        """

        # Map span parser fields → RAGAS dataset columns
        dataset_dict: Dict[str, list] = {
            "user_input": [fields.get("input", "")],
            "response": [fields.get("actual_output", "")],
        }

        # Add optional columns based on what's available
        expected = fields.get("expected_output")
        if expected:
            dataset_dict["reference"] = [expected]

        retrieval_ctx = fields.get("retrieval_context")
        if retrieval_ctx:
            # retrieval_context from span parser may be a list or a single string
            if isinstance(retrieval_ctx, str):
                retrieval_ctx = [retrieval_ctx]
            dataset_dict["retrieved_contexts"] = [retrieval_ctx]
            # reference_contexts defaults to retrieved_contexts for metrics that need it
            dataset_dict["reference_contexts"] = [retrieval_ctx]

        context = fields.get("context")
        if context and "retrieved_contexts" not in dataset_dict:
            if isinstance(context, str):
                context = [context]
            dataset_dict["retrieved_contexts"] = [context]
            dataset_dict["reference_contexts"] = [context]

        dataset = Dataset.from_dict(dataset_dict)

        # Run RAGAS evaluation
        eval_result = evaluate(dataset=dataset, metrics=[self.metric])
        df = eval_result.to_pandas()

        # Find the score column — RAGAS uses metric.name as column prefix,
        # sometimes with a suffix like "(mode=f1)"
        score_col = None
        for col in df.columns:
            if col.startswith(self.metric.name):
                score_col = col
                break

        if score_col is None:
            # Fallback: find any numeric column that's not an input field
            input_cols = {"user_input", "response", "reference", "retrieved_contexts", "reference_contexts"}
            for col in df.columns:
                if col not in input_cols and df[col].dtype in ("float64", "int64", "float32"):
                    score_col = col
                    break

        if score_col is None:
            return EvaluatorOutput(
                label="Error",
                errorCode="NO_SCORE_FOUND",
                errorMessage=(
                    f"RAGAS evaluate() produced no score column for metric "
                    f"'{self.metric.name}'. Columns: {list(df.columns)}"
                ),
            )

        score = float(df[score_col].iloc[0])
        threshold = getattr(self.metric, "threshold", 0.5)
        label = "Pass" if score >= threshold else "Fail"
        reason = f"RAGAS {self.metric.name}: {score:.4f} (threshold={threshold})"

        return EvaluatorOutput(value=score, label=label, explanation=reason)
