from abc import abstractmethod
from typing import Any, AsyncGenerator, Callable, Generator, Optional, Sequence, cast
from enum import Enum
from pydantic import BaseModel, Field


class ChatRole(str, Enum):
    """Message role of chat."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Message(BaseModel):
    """Chat message."""

    role: ChatRole = ChatRole.USER
    content: Optional[str] = ""
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.role.value}: {self.content}"

class ChatResponse(BaseModel):
    """Chat response."""

    message: Message
    raw: Optional[dict] = None
    delta: Optional[str] = None
    additional_kwargs: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return str(self.message)

ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]
    
class CompletionResponse(BaseModel):
    """Completion response."""

    text: str
    additional_kwargs: dict = Field(default_factory=dict)
    raw: Optional[dict] = None
    delta: Optional[str] = None

    def __str__(self) -> str:
        return self.text

class LLM(BaseModel):
    """LLM interface."""


    @abstractmethod
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        pass

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Completion endpoint for LLM."""
        pass

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        """Streaming chat endpoint for LLM."""
        pass