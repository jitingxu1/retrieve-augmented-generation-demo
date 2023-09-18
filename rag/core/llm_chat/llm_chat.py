from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.openai import ChatOpenAI
from langchain.chat_models.anthropic import ChatAnthropic


def load_llm(llm_name: str) -> BaseChatModel:
    # input api key
    if llm_name == "gpt-3.5-turbo":
        return ChatOpenAI(model_name="gpt-3.5-turbo")
    elif llm_name == "ChatAnthropic": # TODO: change to a proper name       
        return ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    else:
        raise Exception("Unknown LLM type: " + llm_name)
