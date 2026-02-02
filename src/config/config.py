import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


load_dotenv()


class Config() :
    """ Configuration class for RAG System """

    api_key = os.getenv("claudeAPI")

    LLM_MODEL = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        api_key=api_key,
    )

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Default URLs
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    
    @classmethod
    def get_llm(cls):
        """Return the LLM model (ChatAnthropic) defined for this config."""
        return cls.LLM_MODEL



