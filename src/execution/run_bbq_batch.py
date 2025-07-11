#!/usr/bin/env python3
"""
BBQ Batch Runner
Automatically runs language model on BBQ variation files.

Example usage:
python run_bbq_batch.py --batch_size 5 --max_retries 5
python run_bbq_batch.py --model llama_3_3_70b --max_tokens 512
python run_bbq_batch.py --no_resume  # Start fresh
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from batch_runner_base import BatchRunnerBase
from src.constants import MODEL_SHORT_NAMES, get_model_dir_name
from src.model_client import get_supported_platforms


class BBQBatchRunner(BatchRunnerBase):
    """Batch runner for BBQ tasks."""

    def __init__(self):
        super().__init__(
            task_name="BBQ Bias in QA",
            data_dir_name="bbq",
            file_pattern="bbq_variations.json"
        )

    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract dataset name from BBQ filename."""
        if filename == "bbq_variations.json":
            return "variations"
        # If we later add more specific BBQ datasets, this logic can be extended
        if filename.startswith('bbq_') and filename.endswith('_variations.json'):
            return filename[4:-16]  # Remove 'bbq_' (4 chars) and '_variations.json' (16 chars)
        return filename

    def get_display_name(self, identifier: str) -> str:
        """Convert identifier to display name."""
        if identifier == "variations":  # For the default single file
            return "BBQ Dataset"
        return identifier.replace("_", " ").title()

    def get_metrics_function(self) -> Optional[Callable]:
        """
        Return None - BBQ metrics are calculated separately in evaluation phase.
        """
        return None

    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """
        Create a basic metrics function for BBQ that extracts gold answer and performs simple matching.
        For more advanced BBQ-specific evaluation, use the separate evaluation phase.
        """
        return self.create_basic_metrics_function_with_gold_field(gold_field)

    def create_result_dict(self, identifier: str, status: str, duration: float,
                           variations_processed: int = None, output_file: str = None,
                           error: str = None) -> Dict[str, Any]:
        """Create a result dictionary with BBQ-specific fields."""
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


def print_bbq_accuracy_summary(results_dir: Path, model_short: str) -> None:
    """
    Print BBQ accuracy summary.
    """
    model_dir = results_dir / model_short
    if not model_dir.exists():
        print("📊 No BBQ data available")
        return

    json_files = list(model_dir.glob("*.json"))
    if not json_files:
        print("📊 No BBQ data available")
        return

    total_responses = 0
    total_correct = 0
    dataset_accuracies = {}

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                dataset_results = json.load(f)

            dataset_name = BBQBatchRunner().extract_identifier_from_filename(json_file.name)
            dataset_total = len(dataset_results)
            dataset_correct = sum(1 for result in dataset_results if result.get('is_correct', False))
            dataset_accuracy = (dataset_correct / dataset_total * 100) if dataset_total > 0 else 0.0

            total_responses += dataset_total
            total_correct += dataset_correct
            dataset_accuracies[dataset_name] = {
                "accuracy": dataset_accuracy,
                "correct": dataset_correct,
                "total": dataset_total
            }
        except Exception as e:
            print(f"⚠️  Error reading {json_file}: {e}")

    if total_responses == 0:
        print("📊 No BBQ data available")
        return

    overall_accuracy = (total_correct / total_responses * 100)

    print(f"\n📊 BBQ Results Summary:")
    print(f"   Total datasets: {len(dataset_accuracies)}")
    print(f"   Total responses: {total_responses}")
    print(f"   ✅ Total correct: {total_correct}")
    print(f"   📈 Overall accuracy: {overall_accuracy:.2f}%")

    # Show dataset performance
    if dataset_accuracies:
        sorted_datasets = sorted(dataset_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print(f"\n🏆 Dataset performance:")
        for i, (dataset, data) in enumerate(sorted_datasets, 1):
            display_name = BBQBatchRunner().get_display_name(dataset)
            print(f"   {i}. {display_name}: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})")


def main():
    """Main function to run language model on all BBQ files."""
    runner = BBQBatchRunner()

    parser = argparse.ArgumentParser(description="Run language model on BBQ variations")

    # Setup common arguments
    default_data_dir = str(Path(__file__).parent.parent.parent / "data" /"generated_data"/ "bbq")
    runner.setup_common_args(parser, default_data_dir)

    # Add BBQ-specific arguments
    parser.add_argument("--datasets", nargs="+",
                        help="Run only specific datasets (e.g., --datasets variations)")
    parser.add_argument("--list_datasets", action="store_true",
                        help="List available datasets and exit")
    parser.add_argument("--all", action="store_true",
                        help="Process all BBQ datasets (default: True)", default=True)

    # Add gold_field with BBQ-specific default
    runner.add_gold_field_with_default(parser, "label",
                                       "Field name in gold_updates containing the correct label (default: 'label')")

    args = parser.parse_args()

    # Handle list datasets option
    if args.list_datasets:
        bbq_dir = Path(args.bbq_dir).resolve()
        if not bbq_dir.exists():
            print(f"❌ BBQ directory not found: {bbq_dir}")
            return

        bbq_files = runner.find_variation_files(bbq_dir)
        if not bbq_files:
            print(f"❌ No BBQ variation files found in: {bbq_dir}")
            print("   Expected files matching pattern: bbq_*_variations.json")
            return

        datasets = [runner.extract_identifier_from_filename(f.name) for f in bbq_files]
        datasets.sort()

        print(f"📚 Available BBQ datasets ({len(datasets)}):")
        for i, dataset in enumerate(datasets, 1):
            display_name = runner.get_display_name(dataset)
            print(f"   {i:2d}. {display_name} ({dataset})")
        return

    # Get the full model name based on platform and model key
    full_model_name = MODEL_SHORT_NAMES.get(args.model, args.model)
    print(f"🤖 Using model: {full_model_name} on {args.platform}")

    # Find and filter BBQ files
    bbq_dir = Path(args.bbq_dir).resolve()
    if not bbq_dir.exists():
        print(f"❌ BBQ directory not found: {bbq_dir}")
        return

    bbq_files = runner.find_variation_files(bbq_dir)
    if not bbq_files:
        print(f"❌ No BBQ variation files found in: {bbq_dir}")
        return

    # Filter datasets if specified or if --all is not set (i.e. if specific datasets are implied)
    if args.datasets and not args.all:
        datasets_to_include = set(args.datasets)
        bbq_files = [f for f in bbq_files
                     if runner.extract_identifier_from_filename(f.name) in datasets_to_include]
        if not bbq_files:
            print(f"❌ No files found for specified datasets: {args.datasets}")
            return
    elif not args.all and not args.datasets:  # If neither --all nor specific datasets are given, behave as --all
        print("No specific datasets provided and --all not specified. Processing all available BBQ datasets.")
    elif args.all and args.datasets:  # Warn if both --all and --datasets are specified
        print("⚠️ Warning: --all flag is set. Ignoring --datasets argument and processing all available datasets.")

    # Print header and process files
    runner.print_header(args, full_model_name, bbq_files)

    # Process files
    results = []
    total_start_time = time.time()

    for i, bbq_file in enumerate(bbq_files, 1):
        dataset = runner.extract_identifier_from_filename(bbq_file.name)
        display_name = runner.get_display_name(dataset)
        print(f"\n📚 Processing dataset {i}/{len(bbq_files)}: {display_name}")

        result = runner.run_language_model_on_file(bbq_file, args)
        results.append(result)

        # Print result
        if result["status"] == "success":
            variations_count = result.get("variations_processed", "unknown")
            print(f"✅ {display_name} completed in {result['duration']:.1f}s ({variations_count} variations)")
        else:
            print(f"❌ {display_name} failed: {result.get('error', 'Unknown error')} in {result['duration']:.1f}s")

        # Show progress
        if i < len(bbq_files):
            runner.print_progress_summary(results, i, len(bbq_files))

    # Save summary and print final results
    total_duration = time.time() - total_start_time
    model_short = MODEL_SHORT_NAMES.get(args.model, args.model)
    model_dir_name = get_model_dir_name(model_short, args.quantization)
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "bbq"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Print simple summary
    successful = len([r for r in results if r["status"] == "success"])
    print(f"\n🎉 BBQ Processing Completed!")
    print(f"   ✅ Successful: {successful}/{len(results)}")
    print(f"   ⏱️  Total time: {total_duration:.1f}s")

    # Print BBQ accuracy summary
    print_bbq_accuracy_summary(results_dir, model_dir_name)


if __name__ == "__main__":
    main()
