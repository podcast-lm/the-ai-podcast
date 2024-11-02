import hashlib
import io
import json
import os
from typing import Dict

import requests
from pydub import AudioSegment


def fetch_voice_ids(elevenlabs_api_key: str) -> Dict[str, str]:
    """
    Fetches available voice IDs from ElevenLabs API.

    Args:
        elevenlabs_api_key: ElevenLabs API key

    Returns:
        Mapping of voice names to voice IDs
    """
    url = "https://api.elevenlabs.io/v1/voices"

    headers = {
        "Accept": "application/json",
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    voice_map = {}
    for voice in data["voices"]:
        voice_map[voice["name"]] = voice["voice_id"]

    return voice_map


def generate_audio(
    script: str,
    output_dir: str,
    elevenlabs_api_key: str,
    voice: str,
    use_cache: bool = True,
    cache_dir: str = ".cache",
) -> None:
    """Generate audio from script using ElevenLabs API.

    Args:
        script: Text script to convert to audio
        output_dir: Directory to save audio file
        elevenlabs_api_key: ElevenLabs API key
        voice: Name of voice to use
        use_cache: Whether to use caching
        cache_dir: Directory to store cached audio files

    Raises:
        ValueError: If specified voice is not found
        Exception: If audio generation fails
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    voice_map = fetch_voice_ids(elevenlabs_api_key)

    if voice not in voice_map:
        raise ValueError(
            f"Voice '{voice}' not found. Available voices: {list(voice_map.keys())}"
        )

    voice_id = voice_map[voice]
    OUTPUT_PATH = os.path.join(output_dir, "audio.mp3")

    # Split script into paragraphs
    paragraphs = script.split("\n\n")

    # Compute cache key
    cache_data = {"script": script, "voice_id": voice_id}
    cache_str = json.dumps(cache_data, sort_keys=True)
    cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.mp3")

    # Check cache
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as src, open(OUTPUT_PATH, "wb") as dst:
            dst.write(src.read())
        return

    segments = []
    previous_request_ids = []
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"Accept": "application/json", "xi-api-key": elevenlabs_api_key}

    for i, paragraph in enumerate(paragraphs):
        is_first_paragraph = i == 0
        is_last_paragraph = i == len(paragraphs) - 1

        data = {
            "text": paragraph,
            "model_id": "eleven_multilingual_v2",
            # "previous_request_ids": previous_request_ids[-3:],
            "previous_text": None if is_first_paragraph else " ".join(paragraphs[:i]),
            "next_text": None if is_last_paragraph else " ".join(paragraphs[i + 1 :]),
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }

        response = requests.post(tts_url, headers=headers, json=data)

        if not response.ok:
            raise Exception("Failed to generate audio")

        previous_request_ids.append(response.headers["request-id"])
        segments.append(AudioSegment.from_mp3(io.BytesIO(response.content)))

    # Combine all segments
    final_audio = segments[0]
    for segment in segments[1:]:
        final_audio = final_audio + segment

    # Save to both cache and output if caching is enabled
    if use_cache:
        final_audio.export(cache_path, format="mp3")
        final_audio.export(OUTPUT_PATH, format="mp3")
    else:
        final_audio.export(OUTPUT_PATH, format="mp3")
