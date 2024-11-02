import argparse
import logging
import os

from llm import LLM
from tts import generate_audio


def create_podcast_script(
    input_file: str,
    output_dir: str,
    anthropic_api_key: str,
    model_name: str,
    use_cache: bool = True,
    save_traces: bool = False,
) -> str:
    """
    Create a podcast script from input material using LLM generation.

    Args:
        input_file: Path to input text file
        output_dir: Directory to write output files
        anthropic_api_key: Anthropic API key
        model_name: Name of the model to use
        use_cache: Whether to use cached LLM responses
        save_traces: Whether to save LLM generation traces to files

    Returns:
        str: The generated podcast script
    """
    logging.info("Initializing LLM helper")
    # Initialize LLM helper
    llm = LLM(api_key=anthropic_api_key, model=model_name)

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
        trace_prefix=os.path.join(traces_dir, "metadata") if save_traces else None,
        DOCUMENT=document,
    )
    source_info = llm.extract_tag(response, "research_note").strip()

    logging.info("Getting source summary")
    # Get source summary
    response = llm.generate(
        "prompts/summary_note_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "summary") if save_traces else None,
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
        trace_prefix=os.path.join(traces_dir, "questions") if save_traces else None,
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
                os.path.join(traces_dir, f"answer_{i}") if save_traces else None
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
        trace_prefix=os.path.join(traces_dir, "plan") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
    )
    monologue_plan = llm.extract_tag(response, "monologue_planning")

    logging.info("Getting initial monologue draft")
    # Get initial monologue draft
    response = llm.generate(
        "prompts/podcast_draft_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "draft") if save_traces else None,
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
        trace_prefix=os.path.join(traces_dir, "revised") if save_traces else None,
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
        trace_prefix=os.path.join(traces_dir, "final") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        SCRIPT=monologue_improved,
    )
    monologue_final = llm.extract_tag(response, "script")

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
        "--model-name",
        default="claude-3-5-sonnet-20241022",
        help="Name of the model to use",
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
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Save traces: {args.save_traces}")

    logging.info("Getting API keys from environment")
    # Get API keys from environment
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not anthropic_api_key:
        logging.error("Missing ANTHROPIC_API_KEY environment variable")
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
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
        anthropic_api_key=anthropic_api_key,
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
