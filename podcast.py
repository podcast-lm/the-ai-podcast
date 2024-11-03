import argparse
import logging
import os
import re

from tts import generate_audio


def clean_script(script: str) -> str:
    """
    Clean the script by removing annotations in parentheses and square brackets.

    Args:
        script: The script text to clean

    Returns:
        str: The cleaned script with annotations removed
    """
    # Remove text within parentheses and square brackets
    cleaned = re.sub(r"\([^)]*\)", "", script)  # Remove (text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)  # Remove [text]
    return cleaned


def create_podcast_script(
    input_file: str,
    output_dir: str,
    llm_provider: str,
    model_name: str,
    use_cache: bool = True,
    save_traces: bool = False,
) -> str:
    """
    Create a podcast script from input material using LLM generation.

    Args:
        input_file: Path to input text file
        output_dir: Directory to write output files
        llm_provider: LLM provider to use ('google' or 'anthropic')
        model_name: Name of the model to use
        use_cache: Whether to use cached LLM responses
        save_traces: Whether to save LLM generation traces to files

    Returns:
        str: The generated podcast script
    """
    logging.info("Initializing LLM helper")
    # Initialize LLM helper based on provider
    if llm_provider == "google":
        from gemini import LLM

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Google provider"
            )
        llm = LLM(api_key=api_key, model=model_name)
    elif llm_provider == "anthropic":
        from claude import LLM

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
        llm = LLM(api_key=api_key, model=model_name)
    else:
        raise ValueError("llm_provider must be either 'google' or 'anthropic'")

    # Create traces directory if saving traces
    traces_dir = os.path.join(output_dir, "traces") if save_traces else None
    if save_traces:
        os.makedirs(traces_dir, exist_ok=True)
        logging.info(f"Writing traces to {traces_dir}")

    logging.info("Reading input document")
    # Read input document
    with open(input_file, "r") as f:
        document = f.read().strip()

    logging.info("Getting source metadata")
    # Get source info
    response = llm.generate(
        "prompts/metadata_note_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "01_metadata") if save_traces else None,
        DOCUMENT=document,
    )
    source_info = llm.extract_tag(response, "research_note").strip()

    logging.info("Getting source summary")
    # Get source summary
    response = llm.generate(
        "prompts/summary_note_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "02_summary") if save_traces else None,
        DOCUMENT=document,
        META=source_info,
    )
    source_summary = llm.extract_tag(response, "research_note")

    logging.info("Getting deep dive questions")
    # Get deep dive questions
    response = llm.generate(
        "prompts/deep_dive_questions_prompt.txt",
        cache=use_cache,
        response_prefill="<analysis>",
        trace_prefix=os.path.join(traces_dir, "03_questions") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
    )
    questions_list = llm.extract_all_tags(response, "question")

    logging.info("Getting deep dive answers")
    # Get deep dive answers
    answers_list = []

    for i, question in enumerate(questions_list, 1):
        logging.info(f"Processing answer {i} of {len(questions_list)}")
        response = llm.generate(
            "prompts/deep_dive_answer_prompt.txt",
            response_prefill="The following is the original question, my analysis and answer:",
            cache=use_cache,
            trace_prefix=(
                os.path.join(traces_dir, f"04_answer_{i:02d}") if save_traces else None
            ),
            DOCUMENT=document,
            METADATA=source_info,
            SUMMARY=source_summary,
            QUESTION=question,
        )
        answers_list.append(response.strip())

    answers = "\n\n".join(answers_list)

    logging.info("Getting podcast monologue plan")
    # Get podcast monologue plan
    response = llm.generate(
        "prompts/podcast_plan_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "05_plan") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
    )
    monologue_plan = llm.extract_tag(response, "plan")

    logging.info("Getting initial monologue draft")
    # Get initial monologue draft
    response = llm.generate(
        "prompts/podcast_draft_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "06_draft") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        PLAN=monologue_plan,
    )
    monologue_draft = llm.extract_tag(response, "script")

    logging.info("Getting improved monologue")
    # Get improved monologue
    response = llm.generate(
        "prompts/podcast_revise_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "07_revised") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        DRAFT=monologue_draft,
    )
    monologue_improved = llm.extract_tag(response, "script")

    logging.info("Getting final monologue")
    # Get final monologue
    response = llm.generate(
        "prompts/podcast_final_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "08_final") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        SCRIPT=monologue_improved,
    )
    monologue_final = llm.extract_tag(response, "script")

    # Clean the final script
    monologue_final = clean_script(monologue_final)

    logging.info("Writing script to file")
    script_path = os.path.join(output_dir, "script.txt")
    with open(script_path, "w") as f:
        f.write(monologue_final)

    return monologue_final


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate an AI podcast from input material"
    )
    parser.add_argument("--input", required=True, help="Input material text file")
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for podcast files"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching of LLM responses"
    )
    parser.add_argument(
        "--voice", default="Callum", help="Voice to use for audio generation"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["google", "anthropic"],
        default="anthropic",
        help="LLM provider to use (google or anthropic)",
    )
    parser.add_argument(
        "--model-name",
        default="claude-3-sonnet-20240620",
        choices=[
            "gemini-1.5-pro-002",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
        ],
        help="Name of the model to use (gemini or claude)",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save LLM generation traces to files",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to generate podcast script and audio."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logging.info("Starting podcast generation")
    args = parse_args()

    logging.info("Configuration:")
    logging.info(f"  Input file: {args.input}")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Cache enabled: {not args.no_cache}")
    logging.info(f"  Voice: {args.voice}")
    logging.info(f"  LLM Provider: {args.llm_provider}")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Save traces: {args.save_traces}")

    logging.info("Getting API keys from environment")
    # Get API keys from environment
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not elevenlabs_api_key:
        logging.error("Missing ELEVENLABS_API_KEY environment variable")
        raise ValueError("ELEVENLABS_API_KEY environment variable is required")

    logging.info("Creating output directory")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Generating podcast script")
    # Generate podcast script and audio
    script = create_podcast_script(
        input_file=args.input,
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        use_cache=not args.no_cache,
        save_traces=args.save_traces,
    )

    logging.info("Generating audio from script")
    # Generate audio from script
    generate_audio(
        script=script,
        output_dir=args.output_dir,
        elevenlabs_api_key=elevenlabs_api_key,
        voice=args.voice,
        use_cache=not args.no_cache,
    )

    logging.info("Podcast generation complete")


if __name__ == "__main__":
    main()
