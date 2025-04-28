from openai import OpenAI
from anthropic import Anthropic, NotGiven
import google.generativeai as genai
from together import Together
from evaluation.constant import LLMClientType

class LLMClient:
    def __init__(
        self,
        llm_client_type: LLMClientType,
        llm_name: str,
        llm_server_url: str | None = None,
        llm_server_api_key: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ):
        if llm_client_type == LLMClientType.OpenAI:
            self.client = OpenAI(
                base_url=llm_server_url,
                api_key=llm_server_api_key,
            )
        elif llm_client_type == LLMClientType.TogetherAI:
            self.client = Together(
                api_key=llm_server_api_key,
            )
        elif llm_client_type == LLMClientType.Gemini:
            genai.configure(api_key=llm_server_api_key)
            self.client = genai.GenerativeModel(
                model_name=llm_name,
                system_instruction=system_prompt,
            )
        elif llm_client_type == LLMClientType.Claude:
            self.client = Anthropic(api_key=llm_server_api_key)

        self.llm_name = llm_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            self.test_connection()
        except Exception as e:
            raise e

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, str]:
        if isinstance(self.client, genai.GenerativeModel):
            response = self.client.start_chat(history=[]).send_message(
                content=messages[-1]['content'],
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ),
            )
            model_output = response.text
            finish_reason = response._result.candidates[0].finish_reason.name.lower()
        elif isinstance(self.client, Anthropic):
            system = (
                messages[0]['content'] if messages[0]['role'] == 'system' else NotGiven
            )
            response = self.client.messages.create(
                system=system,
                model=self.llm_name,
                max_tokens=self.max_tokens,
                messages=[messages[-1]],
                temperature=self.temperature,
            )
            model_output = response.content
            finish_reason = (
                'stop' if response.stop_reason == 'stop_sequence' else 'unexpected'
            )
        elif isinstance(self.client, OpenAI) or isinstance(self.client, Together):
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                n=1,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            model_output = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

        return model_output, finish_reason

    def test_connection(self):
        self.generate(
            messages=[{
                'role': 'user',
                'content': 'say hi'
            }]
        )
