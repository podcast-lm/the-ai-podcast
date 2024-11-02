import hashlib
import json
import os
import re
from typing import Dict, List, Optional

import anthropic


class LLM:
    """Helper class to manage LLM interactions"""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20240620",
        cache_dir: str = ".cache",
    ):
        """
        Initialize LLM helper.

        Args:
            api_key: API key for the LLM provider
            model: Model name to use
            cache_dir: Directory to store cached responses
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def load_template(self, template_path: str) -> str:
        """
        Load prompt template from file.

        Args:
            template_path: Path to template file

        Returns:
            str: Template string
        """
        with open(template_path, "r") as f:
            return f.read()

    def fill_template(self, template: str, replacements: Dict[str, str]) -> str:
        """
        Fill template with replacements.

        Args:
            template: Template string
            replacements: Dict of placeholder->value pairs

        Returns:
            str: Filled template string
        """
        filled = template
        for key, value in replacements.items():
            filled = filled.replace(f"{{{{{key}}}}}", value)
        return filled

    def _compute_cache_key(
        self,
        prompt: str,
        response_prefill: Optional[str],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Compute cache key from prompt and parameters.

        Args:
            prompt: Prompt text
            response_prefill: Optional response prefix
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens to generate

        Returns:
            str: Cache key hash
        """
        cache_data = {
            "prompt": prompt,
            "response_prefill": response_prefill,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get cache file path for a given key.

        Args:
            cache_key: Cache key hash

        Returns:
            str: Path to cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.txt")

    def generate(
        self,
        template_path: str,
        max_tokens: int = 8192,
        temperature: float = 0.5,
        response_prefill: Optional[str] = None,
        cache: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            template_path: Path to prompt template file
            max_tokens: Max tokens to generate
            temperature: Temperature parameter
            response_prefill: Optional response prefix
            cache: Whether to use caching
            **kwargs: Template variable replacements

        Returns:
            str: Generated text
        """
        # Load and fill template
        template = self.load_template(template_path)
        prompt = self.fill_template(template, kwargs)

        if cache:
            # Check cache
            cache_key = self._compute_cache_key(
                prompt, response_prefill, self.model, temperature, max_tokens
            )
            cache_path = self._get_cache_path(cache_key)

            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    return f.read()

        messages = [{"role": "user", "content": prompt}]
        if response_prefill:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_prefill}],
                }
            )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        result = response.content[0].text

        if cache:
            # Save to cache
            with open(cache_path, "w") as f:
                f.write(result)

        return result

    def extract_tag(self, text: str, tag: str) -> str:
        """
        Extract content between XML-style tags.

        Args:
            text: Text to extract from
            tag: Tag name to extract

        Returns:
            str: Extracted content

        Raises:
            ValueError: If tag is not found in text
        """
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError(f"Tag <{tag}> not found in text")

    def extract_all_tags(self, text: str, tag: str) -> List[str]:
        """
        Extract all occurrences of content between XML-style tags.

        Args:
            text: Text to extract from
            tag: Tag name to extract

        Returns:
            List[str]: List of extracted content strings

        Raises:
            ValueError: If no matches found for tag
        """
        pattern = f"<{tag}>(.*?)</{tag}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            raise ValueError(f"No matches found for tag <{tag}> in text")
        return [match.strip() for match in matches]
