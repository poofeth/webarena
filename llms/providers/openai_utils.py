"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
import openai.error
from tqdm.asyncio import tqdm_asyncio


def _read_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _configure_openai(
    api_key_env_var: str = "OPENAI_API_KEY",
    api_base_env_var: str = "OPENAI_API_BASE",
    organization_env_var: str = "OPENAI_ORGANIZATION",
    fallback_api_key_env_var: str | None = None,
    fallback_api_base_env_var: str | None = None,
    fallback_organization_env_var: str | None = None,
) -> None:
    api_key = _read_env_var(api_key_env_var)
    if api_key is None and fallback_api_key_env_var is not None:
        api_key = _read_env_var(fallback_api_key_env_var)
    if api_key is None:
        expected_env = api_key_env_var
        if fallback_api_key_env_var is not None:
            expected_env = f"{api_key_env_var} or {fallback_api_key_env_var}"
        raise ValueError(
            f"{expected_env} environment variable must be set when using OpenAI API."
        )

    openai.api_key = api_key
    organization = _read_env_var(organization_env_var)
    if organization is None and fallback_organization_env_var is not None:
        organization = _read_env_var(fallback_organization_env_var)
    openai.organization = organization or ""

    api_base = _read_env_var(api_base_env_var)
    if api_base is None and fallback_api_base_env_var is not None:
        api_base = _read_env_var(fallback_api_base_env_var)
    if api_base is not None:
        openai.api_base = api_base
    else:
        openai.api_base = "https://api.openai.com/v1"


def _supports_custom_temperature(model: str) -> bool:
    model_name = model.lower()
    return not (
        model_name.startswith("gpt-5")
        or model_name.startswith("o1")
        or model_name.startswith("o3")
        or model_name.startswith("o4")
    )


def _chat_completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_token: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    if _supports_custom_temperature(model):
        kwargs["temperature"] = temperature
    if stop_token:
        kwargs["stop"] = [stop_token]
    return kwargs


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.Completion.acreate(  # type: ignore
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    _configure_openai()

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    _configure_openai()
    response = openai.Completion.create(  # type: ignore
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(  # type: ignore
                    **_chat_completion_kwargs(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    ),
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    _configure_openai()

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
    api_key_env_var: str = "OPENAI_API_KEY",
    api_base_env_var: str = "OPENAI_API_BASE",
    organization_env_var: str = "OPENAI_ORGANIZATION",
    fallback_api_key_env_var: str | None = None,
    fallback_api_base_env_var: str | None = None,
    fallback_organization_env_var: str | None = None,
) -> str:
    _configure_openai(
        api_key_env_var=api_key_env_var,
        api_base_env_var=api_base_env_var,
        organization_env_var=organization_env_var,
        fallback_api_key_env_var=fallback_api_key_env_var,
        fallback_api_base_env_var=fallback_api_base_env_var,
        fallback_organization_env_var=fallback_organization_env_var,
    )

    response = openai.ChatCompletion.create(  # type: ignore
        **_chat_completion_kwargs(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop_token=stop_token,
        ),
    )
    answer: str = response["choices"][0]["message"]["content"]
    return answer


def generate_from_openai_eval_chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int = 768,
    top_p: float = 1.0,
    context_length: int = 0,
    stop_token: str | None = None,
) -> str:
    return generate_from_openai_chat_completion(
        messages=messages,
        model=model or os.environ.get("WEBARENA_EVAL_MODEL", "gpt-4-1106-preview"),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        context_length=context_length,
        stop_token=stop_token,
        api_key_env_var="WEBARENA_EVAL_OPENAI_API_KEY",
        api_base_env_var="WEBARENA_EVAL_OPENAI_API_BASE",
        organization_env_var="WEBARENA_EVAL_OPENAI_ORGANIZATION",
        fallback_api_key_env_var="OPENAI_API_KEY",
        fallback_api_base_env_var="OPENAI_API_BASE",
        fallback_organization_env_var="OPENAI_ORGANIZATION",
    )


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    _configure_openai()
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
