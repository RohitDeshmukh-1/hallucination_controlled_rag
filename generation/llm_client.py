import os
from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    Production-ready LLM client using Groq's OpenAI-compatible API.
    Enforces deterministic, low-temperature generation.
    """

    def __init__(self):
        # Load environment variables once at startup
        load_dotenv()

        api_key = os.getenv("LLM_API_KEY")
        api_base = os.getenv("LLM_API_BASE")
        model = os.getenv("LLM_MODEL")

        if not api_key or not api_base or not model:
            raise RuntimeError(
                "Missing LLM configuration. "
                "Ensure LLM_API_KEY, LLM_API_BASE, and LLM_MODEL are set in .env"
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

        self.model = model

    def generate(self, prompt: dict) -> str:
        """
        Generate an answer strictly grounded in the provided prompt.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=0.0,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            # This log is CRITICAL for debugging Groq issues
            print("LLM ERROR:", repr(e))
            raise
