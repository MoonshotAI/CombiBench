import asyncio
import logging
import os
from urllib.parse import urljoin, urlparse, urlunparse
import uuid

import aiohttp
from loguru import logger
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)


class Lean4Client(object):
    """Client for interacting with the Lean 4 verification server.

    This client handles communication with a Lean 4 server for verifying proofs
    and retrieving results. It handles authentication, connection testing,
    and provides methods for synchronous and asynchronous verification.
    """

    def __init__(self, base_url, api_key=None, disable_cache=False) -> None:
        """Initialize the Lean4Client.

        Args:
            base_url (str): Base URL of the Lean 4 server.
            api_key (str, optional): API key for authentication. If None, will try
                to load from LEANSERVER_API_KEY environment variable. Defaults to None.
            disable_cache (bool, optional): Whether to disable result and header caching. Defaults to False.

        Raises:
            Exception: If the Lean server cannot be connected to or is unavailable.
        """
        self.url = base_url
        if api_key is None:
            api_key = os.getenv('LEANSERVER_API_KEY')

        self.api_key = api_key
        self.disable_cache = disable_cache

        self._test_connection()

    def verify(self, codes, timeout=60, infotree_type=None):
        """Synchronous wrapper for verifying proof codes.

        This is a convenience method that wraps the async_verify method
        in an asyncio event loop for synchronous usage.

        Args:
            codes (list): The list of Lean 4 code to verify.
                Each code is a dict containing:
                    - code: The Lean 4 code to verify.
                    - custom_id: The custom id of the proof.
            timeout (int): The timeout in seconds.
            infotree_type (str, optional): Type of info tree to use. Defaults to None.

        Returns:
            dict: The response from the server with verification results.
        """
        return asyncio.run(self.async_verify(codes, timeout, infotree_type))

    async def async_verify(self, codes, timeout, infotree_type=None):
        """verify the proof code and get result

        Args:
            codes (list): The list of lena 4 code to verify.
                Each code is a dict of:
                    - code: The lena 4 code to verify.
                    - custom_id: The custom id of the proof.
            timeout (int): The timeout in seconds.

        Returns:
            response (dict): The response from the server.
                It contains a  key results, which is a list of dictionaries.
                Each dictionary contains the following keys:
                    - code: The custom id of the proof.
                    - error: A string with the error message from the lean server.
                    - response: A dictionary with the response from the LEAN REPL.

        Example:
            >>> client.one_pass_verify("import Mathlib\n\nexample : 2 = 2 := rfl", timeout=60)
            {'results': [{'code': 'test_connection', 'error': None, 'response': {'env': 0}}]}
        """
        json_data = {
            'codes': codes,
            'timeout': timeout,
            'infotree_type': infotree_type,
            'disable_cache': self.disable_cache,
        }
        response = await self._query('post', '/verify', json_data)
        return response

    async def _query(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        n_retries: int = 3,
    ) -> dict:
        """
        One single method for sending all requests, with retry behavior controlled by the caller.

        Args:
            method: The HTTP method to use (e.g., "get", "post").
            endpoint: The endpoint to call.
            json_data: The data to send in the request.
            n_retries: Number of retry attempts.

        Returns:
            response: The response from the server.
        """

        # Create retry decorator with dynamic n_retries
        @retry(
            stop=stop_after_attempt(
                n_retries
            ),  # Dynamic retries based on the caller's argument
            wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
            before_sleep=before_sleep_log(
                logger, logging.ERROR
            ),  # Optional logging of each retry
        )
        async def query_with_retries(url):
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            }

            # Create a session with trust_env set to True
            async with aiohttp.ClientSession(
                trust_env=True, timeout=aiohttp.ClientTimeout(total=3600)
            ) as session:
                async with session.request(
                    method,
                    self._ensure_url_has_scheme(str(urljoin(url, endpoint))),
                    headers=headers,
                    json=json_data,  # Directly send the JSON data
                ) as response:
                    # Get the response body asynchronously and parse it as JSON
                    res = await response.json()

            return res

        # Call the query function with retries
        return await query_with_retries(self.url)

    def _ensure_url_has_scheme(self, url, default_scheme='http'):
        """Ensure URL has a scheme (http/https) prefix.

        Args:
            url (str): The URL to check and potentially modify.
            default_scheme (str, optional): The scheme to add if none exists. Defaults to "http".

        Returns:
            str: URL with a scheme.
        """
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f'{default_scheme}://{url}')
        return urlunparse(parsed)

    def _test_connection(self):
        """Test the connection to the Lean server.

        Sends a simple GET request to the root endpoint to verify
        that the server is available and responsive.

        Raises:
            Exception: If the server cannot be connected to or returns a non-ok status.

        Returns:
            bool: True if connection test passed.
        """
        try:
            response = asyncio.run(self._query('get', '/'))
        except RetryError:
            raise Exception(f'The lean server {self.url} cannot be connected.')

        if response.get('status') != 'ok':
            raise Exception(
                f'The lean server {self.url} cannot be available. {response}'
            )


def is_error(
    feedback: dict,
    accept_sorry: bool = True,
    return_error_messages: bool = False,
):
    """
    Checks if the Lean feedback contains an error.

    Args:
    - feedback: The Lean feedback as a dictionary.
    - accept_sorry: Whether to accept "sorry" statements as "not an error".
    By default, "sorry" statements are not considered errors.
    """

    if 'error' in feedback:
        return (True, [feedback['error']]) if return_error_messages else True

    if 'stderr' in feedback:
        return (True, [feedback['stderr']]) if return_error_messages else True

    has_error = False
    error_data_values = []
    sorry_data_values = []
    if 'messages' in feedback:
        error_data_values = [
            message['data']
            for message in feedback.get('messages', [])
            if message.get('severity') == 'error'
        ]
        has_error = bool(error_data_values)

        if not accept_sorry:
            warning_data_values = [
                message['data']
                for message in feedback.get('messages', [])
                if message.get('severity') == 'warning'
            ]
            sorry_data_values = [
                warning_data
                for warning_data in warning_data_values
                if "declaration uses 'sorry'" in warning_data
            ]
            has_error = has_error or bool(sorry_data_values)

    if return_error_messages:
        return has_error, error_data_values + sorry_data_values
    return has_error


def verify(
    code: str,
    lean4_client: Lean4Client,
    accept_sorry: bool = False,
) -> tuple[bool, dict]:
    """
    Verify the proof of the given task.

    Returns:
        is_valid: bool, whether the proof is valid.
        lean_feedback: dict, the feedback from the Lean 4 evaluator.
    """

    try:
        res = lean4_client.verify([{'proof': code, 'custom_id': str(uuid.uuid4())}], timeout=60)
        res = res['results'][0]
        return (
            res['error'] is None and not is_error(res['response'], accept_sorry=accept_sorry),
            res,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(e)
        return False, {}


if __name__ == '__main__':
    lean_client = Lean4Client(
        base_url='http://localhost:12332',
        api_key=None,
    )

    print(verify('import Mathlib4\n', lean_client))
