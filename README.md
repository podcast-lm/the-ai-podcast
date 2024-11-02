# the-ai-podcast
An open-source AI podcast creator

To use this script, you'll need your own Anthropic and ElevenLabs API keys. To create a podcast, follow these steps:

```
pip install pydub anthropic 
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
export ELEVENLABS_API_KEY=<your-elevenlabs-api-key>
python podcast.py \
    --input the-ai-podcast/episode-01-audio-lm/audio-lm.txt \
    --output-dir the-ai-podcast/episode-01-audio-lm \
    --model-name claude-3-5-sonnet-20240620
```

You can find an example of a generated podcast at [`the-ai-podcast/episode-01-audio-lm/audio.mp3`](the-ai-podcast/episode-01-audio-lm/audio.mp3).

## Limitations

- Currently, only one source document is supported for podcast creation. If you want to create a podcast from multiple sources, combine them into a single document.
- We only support Anthropic for script generation and ElevenLabs for voice generation. However, swapping out LLM and voice providers is straightforward.

## Responsible Use

- Be mindful of the generated content, as you are responsible for what it communicates.
- Inform your audience that the podcast is generated by AI.