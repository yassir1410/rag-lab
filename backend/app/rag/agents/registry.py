from typing import Dict, List, Optional
from app.rag.agents.tools.base import BaseAgentTool
from app.rag.agents.tools.dense_search import DenseSearchTool
from app.rag.agents.tools.hybrid_search import HybridSearchTool
from app.rag.agents.tools.document_reader import ReadDocumentSectionTool
from app.rag.agents.tools.metadata_search import InspectDocumentMetadataTool
from app.rag.agents.tools.evidence_checker import InspectEvidenceTool
from app.rag.agents.schemas import ToolDefinitionSchema


class AgentToolRegistry:
    """
    Allowlisted registry of approved read-only tools for Agentic RAG.
    Guarantees that an LLM can NEVER dynamically define or execute arbitrary tools.
    """

    def __init__(self):
        self._tools: Dict[str, BaseAgentTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        tools = [
            DenseSearchTool(),
            HybridSearchTool(),
            ReadDocumentSectionTool(),
            InspectDocumentMetadataTool(),
            InspectEvidenceTool(),
        ]
        for t in tools:
            self._tools[t.name] = t

    def get_tool(self, name: str) -> Optional[BaseAgentTool]:
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> List[ToolDefinitionSchema]:
        return [tool.get_definition() for tool in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())


# Singleton instance
agent_tool_registry = AgentToolRegistry()
