import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """
    Production-grade LLM client using Groq's OpenAI-compatible API.

    Responsibilities:
    - Load credentials from environment variables
    - Execute a single LLM call
    - Return raw model output only

    Design principles:
    - Stateless
    - Deterministic
    - No prompt construction logic
    - No retrieval or verification logic
    """

    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("LLM_API_KEY")
        self.api_base = os.getenv("LLM_API_BASE")
        self.model = os.getenv("LLM_MODEL")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))

        if not self.api_key or not self.api_base or not self.model:
            raise RuntimeError(
                "Missing required environment variables: "
                "LLM_API_KEY, LLM_API_BASE, LLM_MODEL"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def generate(self, prompt: Dict[str, str]) -> str:
        """
        Execute a grounded LLM call.

        Input:
            prompt = {
                "system": system_prompt,
                "user": user_prompt
            }

        Output:
            Raw string response from the LLM.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content.strip()
