import logging
from openai import OpenAI, OpenAIError
from configs.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Production-ready LLM client using Groq's OpenAI-compatible API.
    Enforces deterministic, low-temperature generation.
    """

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_API_BASE
        self.model = settings.LLM_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: dict) -> str:
        """Generate an answer strictly grounded in the provided prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )

            return response.choices[0].message.content.strip()

        except OpenAIError as e:
            logger.error("LLM API Error: %s", e)
            raise
        except Exception as e:
            logger.error("LLM Unexpected Error: %r", e)
            raise
