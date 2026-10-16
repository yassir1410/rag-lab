import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from app.rag.generation.base import BaseLLMProvider, GenerationRequest
from app.rag.generation.registry import llm_registry
from app.rag.agents.schemas import (
    ToolCallRequest,
    ToolDefinitionSchema,
    AgentActionType,
)
from app.rag.agents.state import AgentState
from app.rag.agents.evidence import EvidenceCollector
from app.rag.agents.registry import AgentToolRegistry


class AgentController:
    """
    LLM-powered Agent Controller for Plan-Act-Observe decision loops.
    Orchestrates tool selection, query decomposition, and final answer synthesis.
    Includes a robust structured-decision fallback parser with strict schema validation.
    """

    def __init__(
        self,
        provider_id: str = "mock",
        model_id: str = "mock-instruct-v1",
        tool_registry: Optional[AgentToolRegistry] = None,
    ):
        self.provider_id = provider_id
        self.model_id = model_id
        self.tool_registry = tool_registry

    def _get_provider(self) -> BaseLLMProvider:
        return llm_registry.get_provider(self.provider_id)

    def decide_next_action(
        self,
        state: AgentState,
        evidence_collector: EvidenceCollector,
        available_tools: List[ToolDefinitionSchema],
    ) -> Tuple[AgentActionType, Optional[str], Optional[ToolCallRequest]]:
        """
        Queries the LLM or structured decision fallback to determine the next action:
        - TOOL_CALL: Execute an approved tool
        - SYNTHESIZE / STOP: Evidence is sufficient, proceed to answer
        """
        tools_summary = "\n".join(
            [f"- {t.name}: {t.description}" for t in available_tools]
        )

        evidence_summary = f"{evidence_collector.total_evidence_count} evidence item(s) collected."
        if evidence_collector.total_evidence_count > 0:
            evidence_summary += "\n" + "\n".join(
                [f"  * {item.source_id}: {item.document_name} - {item.content[:100]}..."
                 for item in evidence_collector.items[-3:]]
            )

        prompt = f"""You are an Agentic Retrieval Controller. Your goal is to gather verifiable evidence to answer the user question.

Question: "{state.question}"

Available Read-Only Tools:
{tools_summary}

Current Evidence State:
{evidence_summary}

Step count: {state.iteration_count + 1} of {state.max_iterations}.
Tool calls so far: {state.tool_call_count} of {state.max_tool_calls}.

Respond strictly in JSON format with one of the following schemas:

Option 1: Call a retrieval tool
{{
  "action": "tool_call",
  "thought": "Reasoning for why this tool and query will find missing evidence",
  "tool_name": "tool_name_here",
  "arguments": {{"query": "search query", "top_k": 5}}
}}

Option 2: Finish and synthesize answer (evidence is sufficient or limits reached)
{{
  "action": "stop",
  "thought": "Explanation of why existing evidence is sufficient or no more evidence is needed"
}}
"""

        provider = self._get_provider()
        response_text = ""
        try:
            req = GenerationRequest(
                prompt=prompt,
                system_prompt="You are a strict, helpful AI Agent Controller. Output only valid JSON.",
                temperature=0.0,
                max_tokens=350,
            )
            resp = provider.generate(req)
            response_text = resp.content
            state.tokens_used += resp.total_tokens or (len(resp.content.split()) + len(prompt.split()))
        except Exception:
            pass

        # Parse LLM response
        decision = self._parse_decision_json(response_text)
        if decision:
            action = decision.get("action")
            thought = decision.get("thought", "")
            if action == "stop":
                return AgentActionType.STOP, thought, None
            elif action == "tool_call":
                t_name = decision.get("tool_name")
                t_args = decision.get("arguments", {})
                if t_name and isinstance(t_args, dict):
                    return AgentActionType.TOOL_CALL, thought, ToolCallRequest(tool_name=t_name, arguments=t_args)

        # Structured-Decision Fallback Heuristic
        return self._heuristic_fallback_decision(state, evidence_collector)

    def _parse_decision_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from text."""
        if not text:
            return None
        try:
            # Try direct JSON parse
            return json.loads(text.strip())
        except Exception:
            pass

        # Try regex search for markdown fenced JSON
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        # Try finding outermost braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass

        return None

    def _heuristic_fallback_decision(
        self,
        state: AgentState,
        evidence_collector: EvidenceCollector,
    ) -> Tuple[AgentActionType, str, Optional[ToolCallRequest]]:
        """
        Deterministic, robust fallback decision logic when LLM output lacks valid JSON.
        Guarantees that deterministic mock runs and unit tests execute reliably.
        """
        q_lower = state.question.lower()

        # Stop condition: if we already have sufficient evidence (e.g. >= 2 items)
        if evidence_collector.total_evidence_count >= 2:
            return AgentActionType.STOP, "Sufficient evidence collected from multiple passages.", None

        # If question asks about document metadata / files
        if any(term in q_lower for term in ["page", "pages", "file", "filename", "size", "document count", "format"]):
            if "inspect_document_metadata" not in [t.tool_name for t in state.tool_calls_history]:
                return (
                    AgentActionType.TOOL_CALL,
                    "Inspecting document metadata to answer document properties question.",
                    ToolCallRequest(tool_name="inspect_document_metadata", arguments={}),
                )

        # If question has technical acronyms or specific keywords -> hybrid search
        if any(term in q_lower for term in ["react", "python", "fastapi", "postgresql", "docker", "redis", "socket"]):
            if "hybrid_search" not in [t.tool_name for t in state.tool_calls_history]:
                return (
                    AgentActionType.TOOL_CALL,
                    "Executing hybrid search to combine keyword and vector matching for technical terms.",
                    ToolCallRequest(tool_name="hybrid_search", arguments={"query": state.question, "strategy": "hybrid_rrf", "top_k": 5}),
                )

        # Default: dense search
        if "dense_search" not in [t.tool_name for t in state.tool_calls_history]:
            return (
                AgentActionType.TOOL_CALL,
                "Executing dense semantic search for the primary question.",
                ToolCallRequest(tool_name="dense_search", arguments={"query": state.question, "top_k": 5}),
            )

        # If we have 1 item, try inspect_evidence
        if evidence_collector.total_evidence_count == 1:
            sources = [evidence_collector.items[0].source_id]
            if "inspect_evidence" not in [t.tool_name for t in state.tool_calls_history]:
                return (
                    AgentActionType.TOOL_CALL,
                    "Verifying coverage of initial evidence.",
                    ToolCallRequest(tool_name="inspect_evidence", arguments={"source_ids": sources, "question": state.question}),
                )

        # Otherwise halt
        return AgentActionType.STOP, "Completed available search steps.", None

    def plan_sub_questions(self, question: str) -> List[str]:
        """
        Decomposes a complex multi-hop question into 2-3 focused sub-questions.
        """
        provider = self._get_provider()
        prompt = f"""Decompose the following complex user question into 2 or 3 distinct, search-friendly sub-questions.

Question: "{question}"

Output strictly a JSON list of strings, e.g.:
["sub-question 1", "sub-question 2"]
"""
        try:
            req = GenerationRequest(
                prompt=prompt,
                system_prompt="You are a query decomposition specialist. Output only a valid JSON array of strings.",
                temperature=0.0,
                max_tokens=250,
            )
            resp = provider.generate(req)
            text = resp.content.strip()

            # Try parsing JSON array
            if text.startswith("[") and text.endswith("]"):
                sub_qs = json.loads(text)
                if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                    return [str(q).strip() for q in sub_qs]

            # Try markdown block
            match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
            if match:
                sub_qs = json.loads(match.group(1))
                if isinstance(sub_qs, list) and len(sub_qs) >= 2:
                    return [str(q).strip() for q in sub_qs]
        except Exception:
            pass

        # Heuristic fallback decomposition
        if " and " in question.lower():
            parts = re.split(r"\s+and\s+", question, flags=re.IGNORECASE)
            if len(parts) >= 2:
                return [p.strip() for p in parts[:3]]

        return [question]

    def synthesize_answer(
        self,
        question: str,
        evidence_collector: EvidenceCollector,
        strategy_label: str = "Agentic RAG",
    ) -> str:
        """
        Generates the final cited answer using the gathered documentary evidence.
        """
        if evidence_collector.total_evidence_count == 0:
            return (
                "Based on the available documents, no relevant evidence was found to answer this question. "
                "The search was halted because no matching documents or sections were located."
            )

        context_block = evidence_collector.build_evidence_context(max_tokens=3000)

        prompt = f"""You are a grounded Agentic Assistant. Answer the user question based STRICTLY and ONLY on the provided documentary evidence.

INSTRUCTIONS:
1. Cite your sources using exact brackets like [S1], [S2] corresponding to the sources in the evidence.
2. Every factual claim must be backed by a source marker.
3. If the evidence does not contain sufficient details to answer part of the question, explicitly state the limitation.
4. Do NOT fabricate citations or cite sources not present in the evidence block.

DOCUMENTARY EVIDENCE:
{context_block}

USER QUESTION:
{question}

GROUNDED ANSWER WITH CITATIONS:"""

        provider = self._get_provider()
        try:
            req = GenerationRequest(
                prompt=prompt,
                system_prompt="You are an expert AI answering assistant that strictly grounds all assertions with [S#] citations.",
                temperature=0.0,
                max_tokens=600,
            )
            resp = provider.generate(req)
            return resp.content.strip()
        except Exception as e:
            # Fallback generation
            first_src = evidence_collector.items[0].source_id
            return f"Based on the retrieved evidence, the information was located in {first_src}."
