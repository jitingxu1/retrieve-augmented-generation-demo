

from src.llms.base_llm import LLM, Message, ChatResponse, CompletionResponse, ChatRole
from typing import Any, Sequence, Callable


def messsages_to_prompt(messages: Sequence[Message]):
    """Convert messages to a prompt string."""
    prompt_messages = [
        f"{m.role.value}: {m.content}" + (f"\n{m.additional_kwargs}" if m.additional_kwargs else "")
        for m in messages
    ]
    prompt_messages.append(f"{ChatRole.ASSISTANT.value}: ")
    return "\n".join(messages)

def completion_response_to_chat_response(
    completion_response: CompletionResponse,
) -> ChatResponse:
    """Convert a completion response to a chat response."""
    return ChatResponse(
        message=Message(
            role=ChatRole.ASSISTANT,
            content=completion_response.text,
            additional_kwargs=completion_response.additional_kwargs,
        ),
        raw=completion_response.raw,
    )

def completion_to_chat_decorator(
        func: Callable[..., CompletionResponse],        
):
    """Convert a completion function to a chat function."""
    def warpper(messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        prompt = messsages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        return completion_response_to_chat_response(completion_response)


class CustomLLM(LLM):
    """custom abstract class for custom LLMs。

    Subclasses must implement the `__init__`, `complete`,
        `stream_complete`, and `metadata` methods.
    """

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        chat_fn = completion_to_chat_decorator(self.complete)
        return chat_fn(messages, **kwargs)