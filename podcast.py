import argparse
import logging
import os
import re

from llm import LLM
from tts import generate_audio


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
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required for Google provider"
            )
        llm = LLM(api_key=api_key, provider="gemini", model=model_name)
    elif llm_provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )
        llm = LLM(api_key=api_key, provider="claude", model=model_name)
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
    with open("prompts/show_format.txt") as f:
        show_format = f.read()
    response = llm.generate(
        "prompts/podcast_plan_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "05_plan") if save_traces else None,
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        SHOW_FORMAT=show_format,
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
    with open("prompts/host_background.txt") as f:
        host_background = f.read()
    response = llm.generate(
        "prompts/podcast_final_prompt.txt",
        cache=use_cache,
        trace_prefix=(
            os.path.join(traces_dir, "08_final_before_feedback")
            if save_traces
            else None
        ),
        DOCUMENT=document,
        METADATA=source_info,
        SUMMARY=source_summary,
        DEEPDIVE=answers,
        SCRIPT=monologue_improved,
        HOST_BACKGROUND=host_background,
    )
    monologue_final = llm.extract_tag(response, "script")

    for iteration in range(3):
        logging.info(f"Getting audience feedback, iteration {iteration+1}")
        # Get 3 audience feedbacks
        feedbacks = []
        for i in range(3):
            logging.info(f"Getting feedback #{i+1}")
            response = llm.generate(
                "prompts/audience_feedback_prompt.txt",
                cache=use_cache,
                temperature=1.0,
                trace_prefix=(
                    os.path.join(traces_dir, f"09_feedback_{iteration+1}_{i+1}")
                    if save_traces
                    else None
                ),
                SCRIPT=monologue_final,
            )
            feedback = llm.extract_tag(response, "feedback")
            feedbacks.append(feedback)

        logging.info("Improving script based on feedback")
        response = llm.generate(
            "prompts/podcast_final_w_feedback_prompt.txt",
            cache=use_cache,
            trace_prefix=(
                os.path.join(traces_dir, f"10_final_after_feedback_{iteration+1}")
                if save_traces
                else None
            ),
            DOCUMENT=document,
            METADATA=source_info,
            SUMMARY=source_summary,
            DEEPDIVE=answers,
            SCRIPT=monologue_final,
            HOST_BACKGROUND=host_background,
            FEEDBACK="\n\n".join(
                f"Feedback #{i}:\n{feedback}" for i, feedback in enumerate(feedbacks, 1)
            ),
        )
        monologue_final = llm.extract_tag(response, "script")

    # Clean the final script
    response = llm.generate(
        "prompts/cleanup_script_prompt.txt",
        cache=use_cache,
        trace_prefix=os.path.join(traces_dir, "11_cleanup") if save_traces else None,
        SCRIPT=monologue_final,
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
        "--voice", default="Chris", help="Voice to use for audio generation"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["google", "anthropic"],
        default="google",
        help="LLM provider to use (google or anthropic)",
    )
    parser.add_argument(
        "--model-name",
        default="gemini-1.5-flash-002",
        choices=[
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            # "claude-3-5-haiku-20241022",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-002",
        ],
        help="Name of the model to use",
    )
    parser.add_argument(
        "--save-traces",
        action="store_true",
        help="Save LLM generation traces to files",
    )
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Only generate script without audio",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only generate audio from existing script.txt",
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

    if args.script_only and args.audio_only:
        logging.error("Cannot use both --script-only and --audio-only")
        raise ValueError("Cannot use both --script-only and --audio-only")

    logging.info("Configuration:")
    logging.info(f"  Input file: {args.input}")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Cache enabled: {not args.no_cache}")
    logging.info(f"  Voice: {args.voice}")
    logging.info(f"  LLM Provider: {args.llm_provider}")
    logging.info(f"  Model: {args.model_name}")
    logging.info(f"  Save traces: {args.save_traces}")
    logging.info(f"  Script only: {args.script_only}")
    logging.info(f"  Audio only: {args.audio_only}")

    logging.info("Getting API keys from environment")
    # Get API keys from environment
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

    if not args.script_only and not elevenlabs_api_key:
        logging.error("Missing ELEVENLABS_API_KEY environment variable")
        raise ValueError("ELEVENLABS_API_KEY environment variable is required")

    logging.info("Creating output directory")
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    script = None
    if args.audio_only:
        script_path = os.path.join(args.output_dir, "script.txt")
        if not os.path.exists(script_path):
            logging.error(f"Script file not found: {script_path}")
            raise FileNotFoundError(f"Script file not found: {script_path}")
        with open(script_path) as f:
            script = f.read()
    else:
        logging.info("Generating podcast script")
        script = create_podcast_script(
            input_file=args.input,
            output_dir=args.output_dir,
            llm_provider=args.llm_provider,
            model_name=args.model_name,
            use_cache=not args.no_cache,
            save_traces=args.save_traces,
        )

    if not args.script_only:
        logging.info("Generating audio from script")
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
