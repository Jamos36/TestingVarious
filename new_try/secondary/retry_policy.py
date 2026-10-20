"""Bounded retries for transient transport failures; validation is never retried."""
import time
from urllib.error import HTTPError, URLError


def call_with_retry(function, *, attempts=1, on_retry=None):
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except (HTTPError, URLError, TimeoutError, ConnectionError) as error:
            transient = not isinstance(error, HTTPError) or error.code in {
                408, 429, 500, 502, 503, 504}
            if not transient or attempt == attempts:
                raise
            if on_retry:
                on_retry(attempt)
            time.sleep(min(2 ** attempt, 30))
