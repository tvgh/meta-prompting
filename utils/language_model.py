# Import the necessary libraries
import os
import openai
from openai import OpenAI
from typing import Any, Dict, List, Optional, Union
import retry


class LanguageModel:
    """Abstract class for a language model."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.stop_tokens = stop_tokens or []
        self.kwargs = kwargs

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        raise NotImplementedError("generate() not implemented.")

    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_name

    def get_stop_tokens(self) -> List[str]:
        """Get the stop tokens."""
        return self.stop_tokens

    def get_kwargs(self) -> Any:
        """Get the kwargs."""
        return self.kwargs

    def set_kwargs(self, kwargs: Any) -> None:
        """Set the kwargs."""
        self.kwargs = kwargs


class DummyLanguageModel(LanguageModel):
    """A dummy language model that just returns the prompt."""

    def __init__(
        self,
        model_name: str,
        stop_tokens: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, stop_tokens, **kwargs)

    def generate(
        self,
        prompt: str,
        max_length: int = 128,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> List[str]:
        """Generate text based on a prompt."""
        return [prompt] * num_return_sequences


class OpenAI_LanguageModel(LanguageModel):
    """A language model from OpenAI's API."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_type: str,
        api_version: str,
        api_base: str,
        stop_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the OpenAI API.

        Args:
            model_name (str): The name of the model to use.
            api_key (str): The API key to use.
            api_type (str): The API type to use.
            api_version (str): The API version to use.
            api_base (str): The API base to use.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            None
        """
        # Set the OpenAI API parameters
        self.model_name = model_name
        self.api_key = api_key
        self.api_type = api_type
        self.api_version = api_version
        self.api_base = api_base
        self.stop_tokens = stop_tokens

        # Set the model type
        self.model_type = "chat"

        # Set the client to None (by default)
        self.client = None

        placeholder_keys = {None, "", "YOUR_API_KEY"}
        placeholder_bases = {None, "", "YOUR_API_BASE"}
        placeholder_versions = {None, "", "YOUR_API_VERSION"}

        ## OLD AZURE API CODE
        if self.api_type == "azure":
            effective_api_key = (
                None if self.api_key in placeholder_keys else self.api_key
            ) or os.environ.get("OPENAI_API_KEY")
            effective_api_base = (
                None if self.api_base in placeholder_bases else self.api_base
            ) or os.environ.get("OPENAI_BASE_URL")
            effective_api_version = (
                None if self.api_version in placeholder_versions else self.api_version
            ) or os.environ.get("OPENAI_API_VERSION")

            if not all([effective_api_key, effective_api_base, effective_api_version]):
                raise ValueError(
                    "Azure OpenAI usage requires api_key, api_base, and api_version to be provided via arguments or environment variables."
                )

            openai.api_key = effective_api_key
            openai.api_type = self.api_type
            openai.api_version = effective_api_version
            openai.api_base = effective_api_base

            if self.model_name in [
                "code-davinci-002",
                "text-davinci-002",
                "text-davinci-003",
            ]:
                self.model_type = "completion"
            elif self.model_name in [
                "gpt-4",
                "gpt-4-32k",
                "gpt-35-turbo",
                "gpt-4-0314",
                "gpt-4-32k-0314",
                "gpt-35-turbo-0314",
                "gpt-4-0613",
                "gpt-4-32k-0613",
                "gpt-35-turbo-0613",
                "gpt-35-turbo",
            ]:
                self.model_type = "chat"
            else:
                raise ValueError(f"Model {self.model_name} not supported.")
        else:
            ## NEW OPENAI API CODE SUPPORT
            # Skipping model name validation for now

            # Fallback to environment variables if explicit arguments are not provided
            effective_api_key = (
                None if self.api_key in placeholder_keys else self.api_key
            ) or os.environ.get("OPENAI_API_KEY")
            if effective_api_key is None:
                raise ValueError(
                    "An OpenAI API key must be supplied either through the api_key argument or the OPENAI_API_KEY environment variable."
                )

            effective_base_url = (
                None if self.api_base in placeholder_bases else self.api_base
            ) or os.environ.get("OPENAI_BASE_URL")

            # Set up the client, allowing usage of OpenAI compatible endpoints via base_url
            if effective_base_url:
                self.client = OpenAI(
                    api_key=effective_api_key,
                    base_url=effective_base_url,
                )
            else:
                self.client = OpenAI(
                    api_key=effective_api_key,
                )

    @retry.retry(tries=3, delay=1)
    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 512,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate text based on a prompt or messages.

        Args:
            prompt_or_messages (Union[str, List[Dict[str, str]]]): The prompt or messages to generate text from.
            stop_tokens (Optional[List[str]], optional): The stop tokens to use. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 1.
            temperature (float, optional): The temperature to use. Defaults to 0.7.
            top_p (float, optional): The top p to use. Defaults to 1.0.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[str]: The list of generated texts based on the prompt or messages.
        """
        # Set the stop tokens
        stop_tokens = stop_tokens or self.stop_tokens

        print(f"Calling the language model with prompt: {prompt_or_messages}")

        if self.api_type == "azure":
            ## OLD AZURE API CODE
            # Generate the text
            if self.model_type == "chat":
                response = openai.ChatCompletion.create(
                    engine=self.model_name,
                    messages=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of messages
                return [message["message"]["content"] for message in response.choices]
            else:
                response = openai.Completion.create(
                    engine=self.model_name,
                    prompt=prompt_or_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=num_return_sequences,
                    stop=stop_tokens,
                    **kwargs,
                )
                # Return the list of outputs
                return [output["text"] for output in response.choices]
        else:
            ## NEW OPENAI API CODE SUPPORT
            # Generate the text
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_or_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_return_sequences,
                stop=stop_tokens,
                **kwargs,
            )
            return [output.message.content for output in response.choices]
