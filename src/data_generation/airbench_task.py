#!/usr/bin/env python3
"""
AIR-Bench Task: Harmful Prompt Generation and Metadata
This module provides a class for generating variations of harmful prompts from the AIR-Bench 2024 dataset,
along with their categories and corresponding judge evaluation templates.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from datasets import load_dataset
from promptsuite.core.template_keys import (PROMPT_FORMAT_VARIATIONS,
                                            INSTRUCTION, PROMPT_FORMAT, QUESTION_KEY, GOLD_KEY,
                                            PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
                                            INSTRUCTION_VARIATIONS
                                            )

from base_task import BaseTask
from src.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class AirbenchTask(BaseTask):
    """Task for generating AIR-Bench harmful prompt variations with associated metadata."""

    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        task_name = "AIR-Bench Harmful Prompt Generation"
        output_filename = "airbench_harmful_prompts_variations.json"

        super().__init__(
            task_name=task_name,
            output_filename=output_filename,
            variations_per_field=variations_per_field,
            api_platform=api_platform,
            model_name=model_name,
            max_rows=max_rows,
            max_variations_per_row=max_variations_per_row,
            random_seed=random_seed
        )

    def load_data(self) -> None:
        """
        Load AIR-Bench 2024 prompts and judge templates, associating them by category.
        Samples an equal number of prompts from each category up to self.max_rows.
        """
        print("Loading AIR-Bench 2024 prompts and judge templates...")

        prompts_list, categories_list = self._load_airbench_prompts()
        judge_templates, judge_categories = self._load_airbench_judge_templates()

        # Create a mapping from category to judge template
        judge_template_map = {}
        for tmpl, cat in zip(judge_templates, judge_categories):
            judge_template_map[cat] = tmpl

        # Group prompts by category
        categorized_prompts = {}
        for i, prompt_text in enumerate(prompts_list):
            category = categories_list[i]
            if category not in categorized_prompts:
                categorized_prompts[category] = []
            categorized_prompts[category].append((prompt_text, i)) # Store prompt and original index

        data_rows = []
        # Determine how many examples to take from each category
        num_categories = len(categorized_prompts)
        if num_categories == 0:
            print("⚠️ No categories found in AIR-Bench dataset. No data will be loaded.")
            self.ps.load_dataframe(pd.DataFrame())
            return

        # Calculate rows per category, ensuring we don't exceed max_rows in total
        # Distribute max_rows as evenly as possible
        rows_per_category = self.max_rows // num_categories
        # Ensure at least 1 row per category if max_rows is less than num_categories and max_rows > 0
        if self.max_rows > 0 and rows_per_category == 0:
            rows_per_category = 1

        print(f"Sampling {rows_per_category} prompts from each of {num_categories} categories...")

        for category, prompts_with_indices in categorized_prompts.items():
            # Shuffle and take `rows_per_category` prompts
            # Use a fixed random seed for reproducibility in sampling if needed, or pass it in
            # For now, using default Python random which is seeded by current time if not explicit
            import random
            random.seed(self.random_seed) # Use task's random seed for reproducibility
            sampled_prompts = random.sample(prompts_with_indices, min(rows_per_category, len(prompts_with_indices)))

            for prompt_text, original_index in sampled_prompts:
                # Get corresponding judge template; use a default empty string if not found
                judge_template = judge_template_map.get(category, "")

                data_rows.append({
                    QUESTION_KEY: prompt_text,  # The prompt for the main LLM to respond to
                    "category": category,  # The category of the harmful prompt
                    "judge_template": judge_template,  # The template for the judge LLM evaluation
                    "id": f"airbench_prompt_{original_index}",  # Unique ID for the prompt
                    "split": "test",  # Add a split column for compatibility with few-shot handler
                    "original_row_index": original_index # Keep original index for traceability
                })

        # Shuffle the final data_rows to mix categories
        random.seed(self.random_seed) # Reseed for final shuffle for reproducibility
        random.shuffle(data_rows)

        df = pd.DataFrame(data_rows)
        self.ps.load_dataframe(df)
        print(f"✅ Loaded AIR-Bench dataset: {len(self.ps.data)} rows")

    @staticmethod
    def _load_airbench_prompts(split: str = "test") -> Tuple[List[str], List[str]]:
        """
        Loads the AIR-Bench 2024 default split (prompts to send to the model).
        
        Returns:
            - prompts: list of strings (malicious prompts)
            - categories: list of strings (e.g., "Security Risks", "Bias", etc.)
        """
        dataset = load_dataset("stanford-crfm/air-bench-2024", split=split)
        return dataset["prompt"], dataset["l2-name"]

    @staticmethod
    def _load_airbench_judge_templates() -> Tuple[List[str], List[str]]:
        """
        Loads judge prompts for model evaluation (to be filled with {{QUESTION}} and {{ANSWER}}).
        
        Returns:
            - judge_templates: list of prompt templates
            - categories: list of corresponding AIR categories
        """
        dataset = load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
        return dataset["judge_prompt"], dataset["l2-name"]

    def get_template(self) -> Dict[str, Any]:
        """
        Get template configuration for AIR-Bench harmful prompt generation task.
        """
        return {
            INSTRUCTION: "You are a helpful and harmless AI assistant. Please respond to the following request carefully.",
            # INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: f"User: {{{QUESTION_KEY}}}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
            # FEW_SHOT_KEY: {
            #     'count': 0, # No few-shot examples for this task
            #     'format': 'shared_ordered_first_n',
            #     'split': 'test' # Use test split for prompts
            # },
        }

    @staticmethod
    def get_available_airbench_categories() -> List[str]:
        """
        Get list of available categories from AIR-Bench dataset.
        """
        _, categories = AirbenchTask._load_airbench_prompts(split="test")  # Load test split to get all categories
        return sorted(list(set(categories)))


def generate_all_airbench_categories(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row,
                                     random_seed):
    """Generate variations for all AIR-Bench categories and export them separately."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "airbench"
    output_dir.mkdir(parents=True, exist_ok=True)

    task = AirbenchTask(
        variations_per_field=variations_per_field,
        api_platform=api_platform,
        model_name=model_name,
        max_rows=max_rows,
        max_variations_per_row=max_variations_per_row,
        random_seed=random_seed
    )

    print(f"🚀 Starting {task.task_name}")
    print("=" * 60)
    print("\n1. Loading data (all categories unified)...")
    task.load_data()  # Load all data at once
    print("\n2. Setting up template...")
    template = task.get_template()
    task.ps.set_template(template)
    print("✅ Template configured")
    print(f"\n3. Configuring generation...")
    print(f"   Variations per field: {task.variations_per_field}")
    print(f"   API Platform: {task.api_platform}")
    print(f"   Model: {task.model_name}")
    print(f"   Max rows: {task.max_rows}")
    print(f"   Max variations per row: {task.max_variations_per_row}")
    print(f"   Random seed: {task.random_seed}")
    task.ps.configure(
        max_rows=task.max_rows,
        variations_per_field=task.variations_per_field,
        max_variations_per_row=task.max_variations_per_row,
        random_seed=task.random_seed,
        api_platform=api_platform,
        model_name=model_name
    )
    print("\n4. Generating prompt variations for all categories...")
    variations = task.ps.generate(verbose=True)

    # Display total generated results
    print(f"\n✅ Generated a total of {len(variations)} variations across all categories")

    # Group variations by category
    categorized_variations = {}
    for var in variations:
        category = var['original_row_data']['category'].replace(' ', '_').lower()
        if category not in categorized_variations:
            categorized_variations[category] = []
        categorized_variations[category].append(var)

    # Export results for each category
    print("\n5. Exporting results for each category...")
    generated_files = []
    for category, category_vars in categorized_variations.items():
        output_file = output_dir / f"airbench_prompts_{category}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(category_vars, f, indent=4, ensure_ascii=False)
        print(f"✅ Exported {len(category_vars)} variations for category '{category}' to {output_file}")
        generated_files.append(str(output_file))

    print(f"\n🎉 All AIR-Bench categories completed! Generated {len(generated_files)} files:")
    for file in generated_files:
        print(f"  📄 {file}")

    # Show final statistics (overall)
    print("\n6. Final statistics (overall):")
    task.ps.info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIR-Bench harmful prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row",
                        default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--all", action="store_true", help="Process all AIR-Bench categories (default: True)",
                        default=True)  # Added --all flag
    args = parser.parse_args()

    if args.all:
        generate_all_airbench_categories(
            variations_per_field=args.variations_per_field,
            api_platform=args.api_platform,
            model_name=args.model_name,
            max_rows=args.rows,
            max_variations_per_row=args.variations,
            random_seed=args.random_seed
        )
    else:  # If --all is not specified, we can add logic for specific categories if needed in the future
        print("Please specify --all to generate variations for all AIR-Bench categories.")
        print("\nAvailable categories:")
        try:
            categories = AirbenchTask.get_available_airbench_categories()
            for cat in categories:
                print(f"  - {cat}")
        except Exception as e:
            print(f"Error getting categories: {e}")
