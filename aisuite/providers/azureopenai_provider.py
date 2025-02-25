import os

from openai import AzureOpenAI, AsyncAzureOpenAI, DEFAULT_MAX_RETRIES

from aisuite.provider import Provider


class AzureopenaiProvider(Provider):
    def __init__(self, **config):
        self.base_url = config.get("base_url") or os.getenv("AZURE_OPENAI_BASE_URL")
        self.api_version = config.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION")
        self.api_key = config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")

        self.max_retries = config.get("max_retries") or os.getenv("AZURE_OPENAI_MAX_RETRIES")
        if not self.max_retries:
            self.max_retries = DEFAULT_MAX_RETRIES

        if not self.api_key or not self.base_url or not self.api_version:
            raise ValueError("For Azure OpenAI, api_key, azure_endpoint, api_version are required.")


        self.client = AzureOpenAI(api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.base_url, max_retries= self.max_retries)
        self.async_client = AsyncAzureOpenAI(api_key=self.api_key, api_version=self.api_version, azure_endpoint=self.base_url, max_retries= self.max_retries)


    async def async_chat_completions_create(self, deployment_name, messages, **kwargs):
        response = await self.async_client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )

        return response

    def chat_completions_create(self, deployment_name, messages, **kwargs):
        # Any exception raised by OpenAI will be returned to the caller.
        # Maybe we should catch them and raise a custom LLMError.
        response = self.client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            **kwargs  # Pass any additional arguments to the OpenAI API
        )

        return response
