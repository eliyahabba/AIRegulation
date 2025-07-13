#!/usr/bin/env python3
"""
Base class for batch runners to reduce code duplication.
"""

import argparse
import json
import sys
import time
import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Callable

from src.model_client import get_model_response, get_batch_model_responses
from src.constants import (
    LM_DEFAULT_MAX_TOKENS, LM_DEFAULT_PLATFORM, LM_DEFAULT_TEMPERATURE,
    LM_DEFAULT_PARALLEL_WORKERS, LM_DEFAULT_MAX_RETRIES, LM_DEFAULT_RETRY_SLEEP,
    LM_DEFAULT_BATCH_SIZE, LM_DEFAULT_INFERENCE_BATCH_SIZE, LM_DEFAULT_QUANTIZATION,
    PLATFORMS, MODEL_SHORT_NAMES, MODELS, get_model_dir_name
)



def load_variations_file(file_path: str) -> List[Dict[str, Any]]:
    """Load variations from a JSON file."""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            variations = json.load(f)
        print(f"✅ Loaded {len(variations)} variations from {file_path}")
        return variations
    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []


def filter_variations_by_rows_and_variations(variations: List[Dict[str, Any]],
                                             max_rows: int = None,
                                             max_variations_per_row: int = None) -> List[Dict[str, Any]]:
    """Filter variations based on row and variation limits."""
    if max_rows is None and max_variations_per_row is None:
        return variations

    # Group variations by original row index
    row_groups = {}
    for variation in variations:
        row_idx = variation.get('original_row_index', 0)
        if row_idx not in row_groups:
            row_groups[row_idx] = []
        row_groups[row_idx].append(variation)

    # Sort rows and limit them
    sorted_rows = sorted(row_groups.keys())
    if max_rows is not None:
        sorted_rows = sorted_rows[:max_rows]

    # Filter variations
    filtered_variations = []
    for row_idx in sorted_rows:
        row_variations = row_groups[row_idx]
        if max_variations_per_row is not None:
            row_variations = row_variations[:max_variations_per_row]
        filtered_variations.extend(row_variations)

    print(f"🔍 Filtered to {len(filtered_variations)} variations from {len(row_groups)} rows")
    return filtered_variations


def get_model_response_with_retry(conversation: List[Dict[str, Any]],
                                  model_name: str,
                                  max_tokens: int,
                                  platform: str,
                                  temperature: float = LM_DEFAULT_TEMPERATURE,
                                  max_retries: int = LM_DEFAULT_MAX_RETRIES,
                                  base_sleep_time: int = LM_DEFAULT_RETRY_SLEEP,
                                  quantization: Optional[str] = None) -> str:
    """Get model response with retry logic for rate limit errors."""
    for attempt in range(max_retries + 1):
        try:
            response = get_model_response(messages=conversation, model_name=model_name,
                                          max_tokens=max_tokens, platform=platform, temperature=temperature,
                                          quantization=quantization)
            
            # Ensure we return a string, not a dict
            if isinstance(response, dict):
                return response.get("parsed_response", response.get("full_response", ""))
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except Exception as e:
            if attempt < max_retries and "rate limit" in str(e).lower():
                sleep_time = base_sleep_time * (attempt + 1)
                print(f"⏳ Rate limit hit, sleeping {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            else:
                raise e

    raise Exception("Max retries exceeded")


def get_model_name(platform: str, model_key: str) -> str:
    """Get the full model name based on platform and model key."""
    if platform not in MODELS:
        raise ValueError(f"Unsupported platform: {platform}. Supported platforms: {list(MODELS.keys())}")
    if model_key not in MODELS[platform]:
        raise ValueError(f"Unsupported model '{model_key}' for platform '{platform}'. Available models: {list(MODELS[platform].keys())}")
    return MODELS[platform][model_key]


def extract_gold_answer_with_field(variation: Dict[str, Any], gold_field: str) -> tuple[str, bool, Dict[str, Any]]:
    """
    Extract gold answer from variation using the specified gold_field.
    Handles both simple answers and multiple choice questions.
    
    Returns:
        tuple: (gold_answer_text, is_extractable, extra_info)
    """
    try:
        gold_updates = variation.get('gold_updates', {})
        
        # Get the gold value from the specified field
        gold_value = gold_updates.get(gold_field)
        if gold_value is None:
            return f"No gold answer in gold_updates['{gold_field}']", False, {}
        
        # Check if this looks like a multiple choice question (numeric index)
        try:
            gold_index = int(gold_value)
            
            # Get the choices from field_values
            field_values = variation.get('configuration', {}).get('field_values', {})
            choices_str = field_values.get('choices', '')
            
            if not choices_str:
                # If no choices found, treat as simple answer
                return str(gold_value), True, {}
            
            # Split choices by lines and clean them
            choices = [choice.strip() for choice in choices_str.split('\n') if choice.strip()]
            
            # Get the correct answer by index
            if 0 <= gold_index < len(choices):
                gold_answer_text = choices[gold_index]
                return gold_answer_text, True, {'choice_index': gold_index, 'total_choices': len(choices)}
            else:
                return f"Invalid choice index {gold_index} (max: {len(choices)-1})", False, {}
                
        except (ValueError, TypeError):
            # Not a numeric index, treat as simple text answer
            return str(gold_value), True, {}
            
    except Exception as e:
        return f"Error extracting gold answer: {str(e)}", False, {}


def extract_gold_answer_default(variation: Dict[str, Any]) -> tuple[str, bool, Dict[str, Any]]:
    """Extract gold answer using default logic (try common field names)."""
    gold_updates = variation.get('gold_updates', {})
    if not gold_updates:
        return "N/A", False, {}
    
    # Try common field names, fallback to first available field
    for field in ['label', 'answer', 'gold_answer', 'correct_answer']:
        if field in gold_updates:
            return extract_gold_answer_with_field(variation, field)
    
    # If no common field found, use the first available field
    first_field = next(iter(gold_updates.keys()))
    return extract_gold_answer_with_field(variation, first_field)


def create_result_entry(variation: Dict[str, Any], response: str, model_name: str, 
                       gold_answer_text: str, is_correct: bool, extra_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create a result entry dictionary."""
    result = {
        'variation_index': variation.get('variation_count', 0),
        'original_row_index': variation.get('original_row_index', 'Unknown'),
        'run_number': variation.get('run_number', 1),
        'unique_run_id': variation.get('unique_run_id', f"{variation.get('original_row_index', 0)}_{variation.get('variation_count', 0)}_1"),
        'model_response': response,
        'model_name': model_name,
        'gold_answer': gold_answer_text,
        'is_correct': is_correct,
        'conversation': variation.get('conversation', []),
        'template_config': variation.get('template_config', {})
    }
    
    # Add extra metrics if available
    if extra_metrics:
        result.update(extra_metrics)
    
    return result


def process_variations_batch(variations_batch: List[Dict[str, Any]],
                            model_name: str,
                            max_tokens: int,
                            platform: str,
                            temperature: float,
                            max_retries: int,
                            retry_sleep: int,
                            batch_start_num: int,
                            total_variations: int,
                            metrics_function=None,
                            quantization: Optional[str] = None) -> List[Dict[str, Any]]:
    """Process a batch of variations together (more efficient for local models)."""
    results = []
    
    try:
        # For local models with batch size > 1, use batch processing
        if platform == "local" and len(variations_batch) > 1:
            conversations = []
            for variation in variations_batch:
                conversation = variation.get('conversation', [])
                conversations.append(conversation)
            
            # Get batch responses
            batch_responses = get_batch_model_responses(
                conversations, model_name, max_tokens=max_tokens, 
                platform=platform, temperature=temperature,
                quantization=quantization
            )
            
            # Process each response
            for i, (variation, response) in enumerate(zip(variations_batch, batch_responses)):
                # Extract gold answer and check correctness
                if metrics_function:
                    gold_answer_text, is_correct, extra_metrics = metrics_function(variation, response)
                else:
                    gold_answer_text, is_extractable, extra_info = extract_gold_answer_default(variation)
                    is_correct = None
                    extra_metrics = extra_info
                
                # Create result entry
                result = create_result_entry(variation, response, model_name, gold_answer_text, is_correct, extra_metrics)
                results.append(result)
                
                print(f"✅ Completed {batch_start_num + i + 1}/{total_variations} (variation {variation.get('variation_count')})")
        else:
            # Fall back to individual processing
            for i, variation in enumerate(variations_batch):
                result = process_single_variation(
                    variation, model_name, max_tokens, platform, temperature,
                    max_retries, retry_sleep, batch_start_num + i + 1, total_variations, metrics_function,
                    quantization
                )
                results.append(result)
                
        return results
        
    except Exception as e:
        print(f"❌ Error processing batch: {e}")
        # Fall back to individual processing on error
        for i, variation in enumerate(variations_batch):
            try:
                result = process_single_variation(
                    variation, model_name, max_tokens, platform, temperature,
                    max_retries, retry_sleep, batch_start_num + i + 1, total_variations, metrics_function,
                    quantization
                )
                results.append(result)
            except Exception as inner_e:
                print(f"❌ Error processing variation {batch_start_num + i + 1}: {inner_e}")
                results.append(None)
        
        return results


def process_single_variation(variation: Dict[str, Any],
                           model_name: str,
                           max_tokens: int,
                           platform: str,
                           temperature: float,
                           max_retries: int,
                           retry_sleep: int,
                           variation_num: int,
                           total_variations: int,
                           metrics_function=None,
                           quantization: Optional[str] = None) -> Dict[str, Any]:
    """Process a single variation and return the result."""
    try:
        # Get conversation from variation
        conversation = variation.get('conversation', [])
        
        print(f"🔄 Processing {variation_num}/{total_variations} (variation {variation.get('variation_count')})")

        # Run the model with conversation format, max_tokens, platform, and temperature (with retry logic)
        response = get_model_response_with_retry(
            conversation, model_name, max_tokens=max_tokens, platform=platform,
            temperature=temperature, max_retries=max_retries, base_sleep_time=retry_sleep,
            quantization=quantization
        )

        # Extract gold answer and check correctness using custom or default function
        if metrics_function:
            gold_answer_text, is_correct, extra_metrics = metrics_function(variation, response)
        else:
            # Extract gold answer using improved logic
            gold_answer_text, is_extractable, extra_info = extract_gold_answer_default(variation)
            is_correct = None  # No correctness check during execution - will be done in evaluation phase
            extra_metrics = extra_info

        # Create result entry
        result = create_result_entry(variation, response, model_name, gold_answer_text, is_correct, extra_metrics)
        
        print(f"✅ Completed {variation_num}/{total_variations} (variation {variation.get('variation_count')})")
        return result

    except Exception as e:
        print(f"❌ Error processing variation {variation_num}: {e}")
        # Extract gold answer even for errors using custom or default function
        if metrics_function:
            gold_answer_text, is_correct, extra_metrics = metrics_function(variation, f"ERROR: {str(e)}")
        else:
            # Extract gold answer using improved logic
            gold_answer_text, is_extractable, extra_info = extract_gold_answer_default(variation)
            is_correct = None  # No correctness check during execution - will be done in evaluation phase
            extra_metrics = extra_info

        # Create error result entry
        error_response = f"ERROR: {str(e)}"
        return create_result_entry(variation, error_response, model_name, gold_answer_text, is_correct, extra_metrics)


def get_processed_variation_indices(results: List[Dict[str, Any]]) -> set:
    """Get set of (original_row_index, variation_index, run_number) tuples that have already been processed."""
    processed = set()
    for result in results:
        row_idx = result['original_row_index']
        var_idx = result['variation_index']
        run_num = result['run_number']
        processed.add((row_idx, var_idx, run_num))
    return processed


def run_model_on_variations(variations: List[Dict[str, Any]],
                            model_name: str,
                            max_tokens: int,
                            platform: str,
                            output_file: str,
                            temperature: float = LM_DEFAULT_TEMPERATURE,
                            max_retries: int = LM_DEFAULT_MAX_RETRIES,
                            retry_sleep: int = LM_DEFAULT_RETRY_SLEEP,
                            batch_size: int = LM_DEFAULT_BATCH_SIZE,
                            resume: bool = True,
                            parallel_workers: int = LM_DEFAULT_PARALLEL_WORKERS,
                            metrics_function=None,
                            runs_per_sample: int = 1,
                            inference_batch_size: int = LM_DEFAULT_INFERENCE_BATCH_SIZE,
                            quantization: Optional[str] = None,
                            ) -> None:
    """Run the language model on variations and save results."""
    print(f"🤖 Using model: {model_name}")
    print(f"📦 Batch size: {batch_size}, Resume: {resume}, Workers: {parallel_workers}")
    if platform == "local" and inference_batch_size > 1:
        print(f"🚀 Inference batch size: {inference_batch_size} (for local model efficiency)")

    # Load existing results if resume mode is enabled
    results = []
    processed_indices = set()
    if resume:
        results = load_existing_results(output_file)
        if results:
            processed_indices = get_processed_variation_indices(results)
            if processed_indices:
                print(f"📋 Found {len(processed_indices)} already processed variations")

    # Create multiple copies of each variation for multiple runs
    variations_to_process = []
    for variation in variations:
        row_idx = variation.get('original_row_index', 0)
        variation_index = variation.get('variation_count')
        
        for run_number in range(1, runs_per_sample + 1):
            # Create unique key for each run
            variation_key = (row_idx, variation_index, run_number)
            
            # Check if this specific run has already been processed
            if variation_key not in processed_indices:
                # Create a copy of variation with run number
                variation_copy = variation.copy()
                variation_copy['run_number'] = run_number
                variation_copy['unique_run_id'] = f"{row_idx}_{variation_index}_{run_number}"
                variations_to_process.append(variation_copy)

    if not variations_to_process:
        print("✅ All variations and runs already processed!")
        return

    total_original_variations = len(variations)
    total_runs = total_original_variations * runs_per_sample
    print(f"🔄 Processing {len(variations_to_process)} remaining runs ({total_original_variations} variations × {runs_per_sample} runs = {total_runs} total runs)")

    # Thread-safe results list and counter
    results_lock = Lock()

    def add_result_and_save(result: Dict[str, Any], current_count: int):
        """Thread-safe function to add results and save batches."""
        with results_lock:
            if result is not None:
                results.append(result)
            
            # Save batch results
            if len(results) % batch_size == 0 or current_count == len(variations_to_process):
                progress_pct = (current_count / len(variations_to_process)) * 100
                print(f"💾 Saving batch ({len(results)} total results, {progress_pct:.1f}% complete)...")
                save_batch_results(results, output_file)

    if parallel_workers > 1:
        # Parallel processing
        print(f"🚀 Starting parallel processing with {parallel_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Submit all tasks
            future_to_variation = {}
            for i, variation in enumerate(variations_to_process, 1):
                future = executor.submit(
                    process_single_variation,
                    variation, model_name, max_tokens, platform, temperature,
                    max_retries, retry_sleep, i, len(variations_to_process), metrics_function,
                    quantization
                )
                future_to_variation[future] = i

            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_variation):
                completed_count += 1
                try:
                    result = future.result()
                    add_result_and_save(result, completed_count)
                except Exception as e:
                    print(f"❌ Unexpected error in parallel processing: {e}")

    else:
        # Sequential processing with inference batching
        print("🔄 Processing variations sequentially...")
        
        # Group variations into inference batches
        for i in range(0, len(variations_to_process), inference_batch_size):
            batch_variations = variations_to_process[i:i + inference_batch_size]
            
            # Process the batch
            batch_results = process_variations_batch(
                batch_variations, model_name, max_tokens, platform, temperature,
                max_retries, retry_sleep, i, len(variations_to_process), metrics_function,
                quantization
            )
            
            # Add results and save
            for j, result in enumerate(batch_results):
                add_result_and_save(result, i + j + 1)

    # Final save
    print(f"💾 Results saved to: {output_file}")
    csv_file = str(output_file).replace('.json', '.csv')
    print(f"📊 CSV saved to: {csv_file}")
    print(f"📊 Total processed: {len(results)} variations")


def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    """Load existing results from CSV file if it exists (for resume functionality)."""
    # Check for CSV file first (faster loading for resume)
    csv_file = str(output_file).replace('.json', '.csv')
    if os.path.exists(csv_file):
        try:
            # Use pandas to read CSV - much faster and cleaner
            df = pd.read_csv(csv_file)
            # Convert to list of dictionaries
            results = df.to_dict('records')
            print(f"📂 Loaded {len(results)} existing results from {csv_file}")
            return results
        except Exception as e:
            print(f"⚠️  Error loading CSV results, trying JSON: {e}")
    
    # Fallback to JSON if CSV doesn't exist or failed to load
    if not os.path.exists(output_file):
        return []

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"📂 Loaded {len(results)} existing results from {output_file}")
        return results
    except Exception as e:
        print(f"⚠️  Error loading existing results: {e}")
        return []


def save_batch_results(results: List[Dict[str, Any]], output_file: str) -> None:
    """Save results to JSON and CSV files."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    save_results_as_csv(results, str(output_file).replace('.json', '.csv'))


def save_results_as_csv(results: List[Dict[str, Any]], csv_file: str) -> None:
    """Save results as CSV with essential information."""
    if not results:
        return
    
    # Define base columns
    base_columns = ['variation_index', 'original_row_index', 'run_number', 'unique_run_id', 'model_name', 'model_response', 'gold_answer', 'is_correct']
    
    # Add metric columns if they exist in any result
    metric_columns = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'sacrebleu', 'predicted_score', 'mse', 'mae', 'absolute_error', 'parsed_answer', 'gold_numeric_answer']
    available_metrics = [col for col in metric_columns if any(col in result for result in results)]
    
    columns = base_columns + available_metrics

    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            # Create row with default values for missing fields
            csv_row = {col: result.get(col, 0.0 if col in metric_columns else '') for col in columns}
            writer.writerow(csv_row)


class BatchRunnerBase:
    """Base class for batch processing of language model tasks."""
    
    def __init__(self, task_name: str, data_dir_name: str, file_pattern: str):
        """
        Initialize the batch runner.
        
        Args:
            task_name: Human-readable name of the task (e.g., "MMLU", "Translation")
            data_dir_name: Name of the data directory (e.g., "mmlu", "translation")
            file_pattern: Glob pattern to match variation files (e.g., "mmlu_*_variations.json")
        """
        self.task_name = task_name
        self.data_dir_name = data_dir_name
        self.file_pattern = file_pattern
    
    def find_variation_files(self, data_dir: Path) -> List[Path]:
        """Find all variation files matching the pattern."""
        return sorted(data_dir.glob(self.file_pattern))
    
    def extract_identifier_from_filename(self, filename: str) -> str:
        """Extract identifier from filename. Override in subclasses."""
        return filename
    
    def get_display_name(self, identifier: str) -> str:
        """Convert identifier to display name. Override in subclasses if needed."""
        return identifier.replace('_', ' ').title()
    
    def create_result_dict(self, identifier: str, status: str, duration: float, 
                          variations_processed: int = None, output_file: str = None, 
                          error: str = None) -> Dict[str, Any]:
        """Create a result dictionary. Override in subclasses for custom fields."""
        result = {
            "identifier": identifier,
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
    
    def get_metrics_function(self) -> Optional[Callable]:
        """Return the metrics function for this task type. Override in subclasses."""
        return None
    
    def create_metrics_function_with_gold_field(self, gold_field: str) -> Optional[Callable]:
        """Create a metrics function that uses the specified gold_field. Override in subclasses."""
        return None

    def create_basic_metrics_function_with_gold_field(self, gold_field: str) -> Callable:
        """
        Create a basic metrics function that extracts gold answer using the specified field
        and performs simple text matching for correctness.
        """
        def metrics_function(variation: Dict[str, Any], model_response: str) -> tuple[str, bool, Dict[str, Any]]:
            # Extract gold answer using the specified field
            gold_answer_text, is_extractable, extra_info = extract_gold_answer_with_field(variation, gold_field)
            
            if not is_extractable:
                return gold_answer_text, False, extra_info
            
            # Simple text matching for correctness
            model_response_clean = model_response.strip().lower()
            gold_answer_clean = gold_answer_text.strip().lower()
            
            # Check for exact match or if model response contains the gold answer
            is_correct = (model_response_clean == gold_answer_clean or
                         gold_answer_clean in model_response_clean or
                         model_response_clean in gold_answer_clean)
            
            return gold_answer_text, is_correct, extra_info
        
        return metrics_function

    def run_language_model_on_file(self, file_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
        """Run the language model on a single file."""
        identifier = self.extract_identifier_from_filename(file_path.name)
        display_name = self.get_display_name(identifier)
        
        print(f"🚀 Processing: {file_path.name} ({display_name})")
        
        start_time = time.time()
        try:
            # Get the full model name
            full_model_name = get_model_name(args.platform, args.model)
            
            # Create output file path
            main_dir = Path(__file__).parent.parent.parent  # Go to project root
            results_dir = main_dir / "data" / "results" / self.data_dir_name
            model_short = MODEL_SHORT_NAMES.get(full_model_name, "unknown")
            model_dir_name = get_model_dir_name(model_short, args.quantization)
            results_dir = results_dir / model_dir_name
            results_dir.mkdir(parents=True, exist_ok=True)
            output_file = results_dir / f"{file_path.stem}.json"
            
            # Load variations
            variations = load_variations_file(str(file_path))
            if not variations:
                return self.create_result_dict(
                    identifier, "error", time.time() - start_time, 
                    error="No variations found"
                )
            
            # Filter variations based on row and variation limits
            filtered_variations = filter_variations_by_rows_and_variations(
                variations,
                max_rows=args.rows,
                max_variations_per_row=args.variations
            )
            
            if not filtered_variations:
                return self.create_result_dict(
                    identifier, "error", time.time() - start_time,
                    error="No variations to process after filtering"
                )
            
            # Get metrics function - check if gold_field is specified
            metrics_function = None
            if hasattr(args, 'gold_field') and args.gold_field:
                metrics_function = self.create_metrics_function_with_gold_field(args.gold_field)
            else:
                metrics_function = self.get_metrics_function()
            
            # Get runs_per_sample if available (only for code generation)
            runs_per_sample = getattr(args, 'runs_per_sample', 1)
            
            # Run model on variations
            run_model_on_variations(
                filtered_variations,
                full_model_name,
                args.max_tokens,
                args.platform,
                str(output_file),
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                batch_size=args.batch_size,
                resume=not args.no_resume,
                parallel_workers=args.parallel_workers,
                metrics_function=metrics_function,
                runs_per_sample=runs_per_sample,
                inference_batch_size=args.inference_batch_size,
                quantization=args.quantization
            )
            
            return self.create_result_dict(
                identifier, "success", time.time() - start_time,
                variations_processed=len(filtered_variations),
                output_file=str(output_file)
            )
            
        except Exception as e:
            return self.create_result_dict(
                identifier, "error", time.time() - start_time,
                error=str(e)
            )
    
    def print_progress_summary(self, results: List[Dict[str, Any]], current: int, total: int) -> None:
        """Print a progress summary."""
        successful = len([r for r in results if r["status"] == "success"])
        failed = len([r for r in results if r["status"] != "success"])
        
        print(f"\n📈 Progress: {current}/{total} items (✅{successful} ❌{failed})")
        
        if results:
            avg_duration = sum(r["duration"] for r in results) / len(results)
            remaining = total - current
            estimated_remaining = remaining * avg_duration
            print(f"   ⏱️  ETA: {estimated_remaining / 60:.1f} minutes")
    
    def setup_common_args(self, parser: argparse.ArgumentParser, default_data_dir: str) -> None:
        """Setup common command line arguments for all batch runners."""
        # Input/output options
        parser.add_argument(f"--{self.data_dir_name}_dir",
                           default=default_data_dir,
                           help=f"Directory containing {self.task_name} variation files")

        # Model configuration
        parser.add_argument("--model", default="default",
                           help="Model key to use (e.g., 'default', 'gpt_4o_mini', 'llama_3_3_70b')")
        parser.add_argument("--platform", choices=list(PLATFORMS.keys()), default=LM_DEFAULT_PLATFORM,
                           help="Platform to use (TogetherAI or OpenAI)")
        parser.add_argument("--max_tokens", type=int, default=LM_DEFAULT_MAX_TOKENS,
                          help=f"Maximum tokens for response (default: {LM_DEFAULT_MAX_TOKENS})")
        parser.add_argument("--temperature", type=float, default=LM_DEFAULT_TEMPERATURE,
                          help=f"Temperature for response generation (default: {LM_DEFAULT_TEMPERATURE})")
        
        # Processing options
        parser.add_argument("--rows", type=int, default=None,
                           help="Maximum number of rows to process per item (None = all rows)")
        parser.add_argument("--variations", type=int, default=None,
                           help="Maximum variations per row to process (None = all variations)")
        parser.add_argument("--quantization", type=str, default=LM_DEFAULT_QUANTIZATION, choices=["8bit", "4bit", "none"],
                           help="Quantization type for local models (8bit, 4bit, or none)")    

        # Retry and batch options
        parser.add_argument("--max_retries", type=int, default=LM_DEFAULT_MAX_RETRIES,
                            help=f"Maximum number of retries for rate limit errors (default: {LM_DEFAULT_MAX_RETRIES})")
        parser.add_argument("--retry_sleep", type=int, default=LM_DEFAULT_RETRY_SLEEP,
                            help=f"Base sleep time in seconds for rate limit retries (default: {LM_DEFAULT_RETRY_SLEEP})")
        parser.add_argument("--batch_size", type=int, default=LM_DEFAULT_BATCH_SIZE,
                            help=f"Number of variations to process before saving intermediate results (default: {LM_DEFAULT_BATCH_SIZE})")
        parser.add_argument("--inference_batch_size", type=int, default=LM_DEFAULT_INFERENCE_BATCH_SIZE,
                            help=f"Number of variations to process together in one model call (for local models) (default: {LM_DEFAULT_INFERENCE_BATCH_SIZE})")

        # Resume options
        parser.add_argument("--no_resume", action="store_true",
                            help="Don't resume from existing results files (start fresh)")

        # Parallel processing options
        parser.add_argument("--parallel_workers", type=int, default=LM_DEFAULT_PARALLEL_WORKERS,
                            help=f"Number of parallel workers for model calls (1=sequential, default: {LM_DEFAULT_PARALLEL_WORKERS})")

        # Note: gold_field is added by each specific batch runner with appropriate defaults

    def add_gold_field_with_default(self, parser: argparse.ArgumentParser, default_value: str, description: str = None) -> None:
        """Add gold_field argument with a specific default value."""
        if description is None:
            description = f"Field name in gold_updates containing the gold answer/label (default: '{default_value}')"
        
        parser.add_argument("--gold_field", type=str, default=default_value, help=description)

    def print_header(self, args: argparse.Namespace, full_model_name: str, files: List[Path]) -> None:
        """Print processing header information."""
        data_dir = getattr(args, f"{self.data_dir_name}_dir")
        
        print(f"🤖 {self.task_name} Batch Language Model Runner")
        print("=" * 60)
        print(f"{self.task_name} directory: {data_dir}")
        print(f"Platform: {args.platform}")
        print(f"Model: {full_model_name}")
        print(f"Max tokens: {args.max_tokens}")
        if args.rows is not None:
            print(f"Max rows per item: {args.rows}")
        if args.variations is not None:
            print(f"Max variations per row: {args.variations}")
        print(f"Max retries: {args.max_retries}")
        print(f"Retry sleep time: {args.retry_sleep} seconds")
        print(f"Batch size: {args.batch_size}")
        if args.platform == "local" and args.inference_batch_size > 1:
            print(f"Inference batch size: {args.inference_batch_size}")
        if args.quantization:
            print(f"Quantization: {args.quantization}")
        resume_mode = not args.no_resume
        print(f"Resume mode: {resume_mode}")
        print(f"Parallel workers: {args.parallel_workers} {'(sequential)' if args.parallel_workers == 1 else '(parallel)'}")
        
        # Show runs per sample if available (only for code generation)
        runs_per_sample = getattr(args, 'runs_per_sample', 1)
        if runs_per_sample > 1:
            print(f"Runs per sample: {runs_per_sample}")
            
        print(f"Found {len(files)} {self.task_name.lower()} items to process:")
        
        for i, file in enumerate(files, 1):
            identifier = self.extract_identifier_from_filename(file.name)
            display_name = self.get_display_name(identifier)
            print(f"  {i:2d}. {display_name}")
        
        print("=" * 60)
