from abc import ABC, abstractmethod
from typing import Dict, Any, Type
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.rag.agents.schemas import ToolDefinitionSchema, ToolResult


class BaseAgentTool(ABC):
    """
    Abstract Base Class for all approved read-only Agentic RAG tools.
    Every tool must define strict typed arguments via a Pydantic schema,
    refuse mutation, and return a standardized ToolResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name, e.g. 'dense_search'."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed description explaining when and how to call this tool."""
        pass

    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Pydantic model validating tool arguments."""
        pass

    @property
    def is_read_only(self) -> bool:
        """All Agentic RAG tools are strictly read-only."""
        return True

    @property
    def category(self) -> str:
        return "retrieval"

    def get_definition(self) -> ToolDefinitionSchema:
        """Exports JSON schema compatible with OpenAPI / LLM function calling."""
        schema = self.args_schema.model_json_schema()
        return ToolDefinitionSchema(
            name=self.name,
            description=self.description,
            parameters_schema=schema,
            is_read_only=self.is_read_only,
            category=self.category,
        )

    @abstractmethod
    def execute(self, session: Session, **kwargs) -> ToolResult:
        """Executes the tool logic with the provided database session and typed arguments."""
        pass
