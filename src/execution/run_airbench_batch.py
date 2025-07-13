#!/usr/bin/env python3
"""
AIR-Bench Batch Runner
Automatically runs language model on AIR-Bench harmful prompt variation files.

This runner focuses on generating responses to harmful prompts. The evaluation of these responses
against safety criteria will be handled by a separate judge model using the provided judge templates.

Example usage:
python run_airbench_batch.py --batch_size 5 --max_retries 5
python run_airbench_batch.py --model llama_3_3_70b --max_tokens 512
python run_airbench_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

# No direct shared_metrics function for AIR-Bench correctness, as it relies on an external judge
from dotenv import load_dotenv
from src.execution.batch_runner_base import BatchRunnerBase
from src.constants import MODEL_SHORT_NAMES, get_model_dir_name

load_dotenv()


class AirbenchBatchRunner(BatchRunnerBase):
    """Batch runner for AIR-Bench harmful prompt tasks."""

    def __init__(self):
        super().__init__(
            task_name="AIR-Bench Harmful Prompts",
            data_dir_name="airbench",
            file_pattern="airbench_*.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from AIR-Bench filename."""
        if filename.startswith('airbench_') and filename.endswith('.json'):
            # Example: airbench_harmful_prompts.json -> harmful_prompts
            # Example: airbench_prompts.json -> prompts
            # Adjust slice for new pattern: remove 'airbench_' (9 chars) and '.json' (5 chars)
            # We need to handle cases like airbench_prompts.json and airbench_harmful_prompts.json
            # The original logic `filename[9:-16]` was for `_variations.json` (16 chars).
            # Now it's just `.json` (5 chars), so it should be `filename[9:-5]`
            identifier = filename[9:-5]
            if identifier.endswith('_variations'):  # Handle legacy names during transition
                identifier = identifier[:-12]  # Remove _variations
            return identifier
        return filename

    def get_display_name(self, identifier: str) -> str:
        """Convert identifier to display name."""
        if identifier == "harmful_prompts":
            return "AIR-Bench Harmful Prompts"
        return identifier.replace("_", " ").title()

    def get_metrics_function(self) -> Optional[Callable]:
        """
        Return None - AIR-Bench metrics are calculated separately in evaluation phase.
        """
        return None

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """
        Create a basic metrics function for AIR-Bench that extracts gold answer and performs simple matching.
        For AIR-Bench, the gold field is typically 'category' which is metadata, not a direct answer.
        """
        return self.create_basic_metrics_function_with_gold_field(gold_field)

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """
        Create a result dictionary with AIR-Bench specific fields.
        The 'category' field, if present, refers to the category of the harmful prompt,
        not a gold answer or expected model response.
        """
        result = {
            "dataset": identifier,
            "status": status,
            "duration": duration
        }

        if variations_processed is not None:
            result["variations_processed"] = variations_processed
        if output_file is not None:
            result["output_file"] = output_file
        if error is not None:
            result["error"] = error

        return result


def print_airbench_summary(results_dir: Path, model_short: str) -> None:
    """
    Print a summary of AIR-Bench results.
    This summary focuses on completion status rather than specific metrics, as evaluation
    is done by an external judge.
    """
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No AIR-Bench data available")
        return

    json_files = list(model_dir.glob("*.json"))
    if not json_files:
        print("📊 No AIR-Bench data available")
        return

    total_files = len(json_files)
    total_variations_processed = 0
    successful_files = 0
    categories_encountered = set()

    print(f"\n📊 AIR-Bench Results Summary for model: {model_short}")
    print(f"   Total processed files: {total_files}")
    print("=" * 60)

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)

            dataset_name = AirbenchBatchRunner().extract_identifier_from_filename(json_file.name)
            display_name = AirbenchBatchRunner().get_display_name(dataset_name)

            variations_in_file = len(dataset_results)
            total_variations_processed += variations_in_file
            successful_files += 1

            # Collect categories from results. 'category' here refers to the category of the
            # harmful prompt, not a gold answer for the model's response.
            for result in dataset_results:
                if 'category' in result:
                    categories_encountered.add(result['category'])

            print(f"✅ {display_name}: Processed {variations_in_file} variations")

        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")

    print("=" * 60)
    print(f"\nTotal variations processed across all files: {total_variations_processed}")
    print(f"Successfully processed files: {successful_files}/{total_files}")
    if categories_encountered:
        print(f"Categories encountered: {', '.join(sorted(list(categories_encountered)))}")
    else:
        print("No categories found in processed results.")


def main():
    """Main function to run language model on all AIR-Bench files."""
    runner = AirbenchBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on AIR-Bench harmful prompt variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent.parent / "data" / "generated_data" / "airbench")
    runner.setup_common_args(parser, default_data_dir)

    # Add AIR-Bench specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets harmful_prompts)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--all", action="store_true",
                        help="Process all AIR-Bench categories")

    # Gold field is implicitly 'category' for metrics, but not a direct answer field for the LLM
    # No specific --gold_field argument needed for this runner, as the 'gold' is meta-data

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        airbench_dir = Path(args.airbench_dir).resolve()
        if not airbench_dir.exists():
            print(f"❌ AIR-Bench directory not found: {airbench_dir}")
            return

        airbench_files = runner.find_variation_files(airbench_dir)
        if not airbench_files:
            print(f"❌ No AIR-Bench variation files found in: {airbench_dir}")
            print("   Expected files matching pattern: airbench_*.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in airbench_files]
        datasets.sort()

        print(f"📚 Available AIR-Bench datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            display_name = runner.get_display_name(dataset)
            print(f"   {i:2d}. {display_name} ({dataset})")
        return

    # Get the full model name based on platform and model key
    full_model_name = MODEL_SHORT_NAMES.get(args.model, args.model)
    print(f"🤖 Using model: {full_model_name} on {args.platform}")

    # Find and filter AIR-Bench files
    airbench_dir = Path(args.airbench_dir).resolve()
    if not airbench_dir.exists():
        print(f"❌ AIR-Bench directory not found: {airbench_dir}")
        return

    airbench_files = runner.find_variation_files(airbench_dir)
    if not airbench_files:
        print(f"❌ No AIR-Bench variation files found in: {airbench_dir}")
        return

    # Filter datasets if specified
    if args.datasets:
        if args.all:
            print(
                "⚠️ Warning: --all flag is set. Ignoring --datasets argument and processing all available categories.")
        else:
            datasets_to_include = set(args.datasets)
            airbench_files = [f for f in airbench_files
                              if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
            if not airbench_files:
                print(f"❌ No files found for specified datasets: {args.datasets}")
                return
    elif not args.all:
        # If neither --all nor specific datasets are given, behave as --all
        print("No specific datasets provided and --all not specified. Processing all available AIR-Bench categories.")

    # Print header and process files
    runner.print_header(args, full_model_name, airbench_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, airbench_file in enumerate(airbench_files, 1):
        dataset = runner.extract_identifier_from_filename(airbench_file.name)
        display_name = runner.get_display_name(dataset)
        print(f"\n📚 Processing dataset {i}/{len(airbench_files)}: {display_name}")

        result = runner.run_language_model_on_file(airbench_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {display_name} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {display_name} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(airbench_files):
            runner.print_progress_summary(results, i, len(airbench_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(args.model, args.model)
    model_dir_name = get_model_dir_name(model_short, args.quantization)
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "airbench"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 AIR-Bench Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print AIR-Bench specific summary (not accuracy)
    print_airbench_summary(results_dir, model_dir_name)


if __name__ == "__main__":
    main()
