import os

from openai import AzureOpenAI

from aisuite.provider import Provider


class AzureopenaiProvider(Provider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_OPENAI_BASE_URL")
        self.api_version = config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
        self.api_key = config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")

        if not self.api_key or not self.base_url or not self.api_version:
            raise ValueError("For Azure OpenAI, api_key, azure_endpoint, api_version are required.")

        self.client = AzureOpenAI(api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.base_url)


    def chat_completions_create(self, deployment_name, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        response = self.client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )

        return response
