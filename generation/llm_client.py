import time
import logging
from openai import OpenAI, OpenAIError
from configs.settings import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Production-ready LLM client using Groq's OpenAI-compatible API.
    Enforces deterministic, low-temperature generation with retry logic.
    """

    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds

    def __init__(self):
        self.api_key = settings.LLM_API_KEY
        self.base_url = settings.LLM_API_BASE
        self.model = settings.LLM_MODEL

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: dict) -> str:
        """Generate an answer strictly grounded in the provided prompt.

        Retries up to MAX_RETRIES times with exponential backoff on
        transient API errors.
        """
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
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
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM API error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt, self.MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM API error after %d attempts: %s", self.MAX_RETRIES, e)
            except Exception as e:
                logger.error("LLM unexpected error: %r", e)
                raise

        raise last_error  # type: ignore[misc]
