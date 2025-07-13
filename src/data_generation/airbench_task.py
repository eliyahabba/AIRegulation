#!/usr/bin/env python3
"""
AIR-Bench Task: Harmful Prompt Generation and Metadata
This module provides a class for generating variations of harmful prompts from the AIR-Bench 2024 dataset,
along with their categories and corresponding judge evaluation templates.
"""
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset
from promptsuite.core.template_keys import (PROMPT_FORMAT_VARIATIONS,
                                            INSTRUCTION, PROMPT_FORMAT, PARAPHRASE_WITH_LLM, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
                                            INSTRUCTION_VARIATIONS
                                            )

from base_task import BaseTask
from src.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)

categories = [
    "Child Harm",
    "Hate/Toxicity",
    "Operational Misuses",
    "Deception",
    "Criminal Activities",
    "Fundamental Rights",
    "Violence & Extremism",
    "Sexual Content",
    "Self-harm",
    "Defamation",
    "Security Risks",
    "Privacy",
    "Political Usage",
    "Manipulation",
    "Discrimination/Bias",
    "Economic Harm"
]
class AirbenchTask(BaseTask):
    """Task for generating AIR-Bench harmful prompt variations with associated metadata."""

    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED, target_category: str = None):
        task_name = "AIR-Bench Harmful Prompt Generation"
        if target_category:
            output_filename = f"airbench_{target_category.replace(' ', '_').lower()}_variations.json"
        else:
            output_filename = "airbench_variations.json"

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
        self.target_category = target_category

    def load_data(self) -> None:
        """
        Load AIR-Bench 2024 prompts and judge templates, associating them by category.
        Samples equally from each category up to max_rows.
        """
        print("Loading AIR-Bench 2024 prompts and judge templates...")

        # Load prompts and judge templates
        prompts_df = self._load_airbench_prompts()
        judge_df = self._load_airbench_judge_templates()

        # Merge prompts with judge templates by category
        df = prompts_df.merge(judge_df, on='category', how='left')

        if self.target_category:
            if self.target_category not in df['category'].unique():
                raise ValueError(f"Category '{self.target_category}' not found in the dataset.")
            df = df[df['category'] == self.target_category]
            print(f"✅ Filtered data to category: {self.target_category}. Total rows: {len(df)}")


        # Sample equally from each category - aiming for self.max_rows in the test set per category
        categories_in_data = df['category'].unique()
        rows_to_load_per_category = self.max_rows  # Load twice the desired test set size

        print(f"Loading {rows_to_load_per_category} prompts from each of {len(categories_in_data)} categories...")

        sampled_dfs = []
        for category in categories_in_data:
            category_df = df[df['category'] == category]
            if len(category_df) > rows_to_load_per_category:
                category_sample = category_df.sample(n=rows_to_load_per_category, random_state=self.random_seed)
            else:
                category_sample = category_df
            sampled_dfs.append(category_sample)
            print(f"✅ Loaded {len(category_sample)} examples from {category}")

        # Combine all category samples
        df = pd.concat(sampled_dfs, ignore_index=True)

        df['split'] = 'test'  # Default to test

        # Shuffle the final dataset
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        # Update actual_max_rows to reflect total loaded rows
        self.actual_max_rows = len(df)

        self.ps.load_dataframe(df)
        print(f"✅ Loaded AIR-Bench dataset: {len(df)} rows")

    @staticmethod
    def _load_airbench_prompts(split: str = "test") -> pd.DataFrame:
        """
        Loads the AIR-Bench 2024 default split (prompts to send to the model).
        """
        dataset = load_dataset("stanford-crfm/air-bench-2024", split=split)
        df = dataset.to_pandas()

        # Rename columns to match our needs
        df = df.rename(columns={'l2-name': 'category'})
        df['id'] = [f"airbench_prompt_{i}" for i in range(len(df))]
        df['split'] = split
        # rename the prompt column to ''
        return df

    @staticmethod
    def _load_airbench_judge_templates() -> pd.DataFrame:
        """
        Loads judge prompts for model evaluation (to be filled with {{QUESTION}} and {{ANSWER}}).
        """
        dataset = load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
        df = dataset.to_pandas()

        # Rename columns to match our needs
        df = df.rename(columns={'l2-name': 'category', 'judge_prompt': 'judge_template'})

        return df

    def get_template(self) -> Dict[str, Any]:
        """
        Get template configuration for AIR-Bench harmful prompt generation task.
        """
        return {
            INSTRUCTION: "You are a helpful assistant. Please help me in the task below.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Task: {prompt}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            # "prompt": [TYPOS_AND_NOISE_VARIATION]
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
        df = AirbenchTask._load_airbench_prompts(split="test")  # Load test split to get all categories
        return sorted(df['category'].unique().tolist())


def generate_airbench_variations(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row,
                                 random_seed, target_category=None):
    """Generate variations for AIR-Bench dataset."""
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "data"/ "generated_data" / "airbench"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / (f"airbench_{target_category.replace(' ', '_').lower()}_variations.json" if target_category else "airbench_variations.json")

    task = AirbenchTask(
        variations_per_field=variations_per_field,
        api_platform=api_platform,
        model_name=model_name,
        max_rows=max_rows,
        max_variations_per_row=max_variations_per_row,
        random_seed=random_seed,
        target_category=target_category
    )

    print(f"🚀 Starting {task.task_name}")
    print("=" * 60)
    print("\n1. Loading data...")
    task.load_data()
    print("\n2. Setting up template...")
    template = task.get_template()
    task.ps.set_template(template)
    print("✅ Template configured")
    print(f"\n3. Configuring generation...")
    print(f"   Variations per field: {task.variations_per_field}")
    print(f"   API Platform: {task.api_platform}")
    print(f"   Model: {task.model_name}")
    print(f"   Max rows per category: {task.max_rows}")
    print(f"   Total rows loaded: {task.actual_max_rows}")
    print(f"   Max variations per row: {task.max_variations_per_row}")
    print(f"   Random seed: {task.random_seed}")
    task.ps.configure(
        max_rows=task.actual_max_rows,  # Use actual data size after sampling
        variations_per_field=task.variations_per_field,
        max_variations_per_row=task.max_variations_per_row,
        random_seed=random_seed,
        api_platform=api_platform,
        model_name=model_name,
    )
    print("\n4. Generating prompt variations...")
    variations = task.ps.generate(verbose=True)

    # Display results
    print(f"\n✅ Generated {len(variations)} variations")

    # Show a few examples
    print("\n5. Sample variations:")
    for i, var in enumerate(variations[:3]):
        print(f"\nVariation {i + 1}:")
        print("-" * 50)
        prompt = var.get('prompt', 'No prompt found')
        if len(prompt) > 500:
            prompt = prompt[:500] + "..."
        print(prompt)

    # Export results using the correct path
    print(f"\n6. Exporting results to {output_file}...")
    task.ps.export(str(output_file), format="json")
    print("✅ Export completed!")

    # Show final statistics
    print("\n7. Final statistics:")
    task.ps.info()

    # 8. Detailed statistics
    print("\n8. Detailed statistics:")
    original_df = task.ps.data

    print("\nDataset distribution by Category and Split (original rows):")
    # Assuming 'split' column exists after load_data and 'category' is the category column
    if task.target_category:
        category_split_counts = original_df.groupby(['split']).size().unstack(fill_value=0)
    else:
        category_split_counts = original_df.groupby(['category', 'split']).size().unstack(fill_value=0)
    print(category_split_counts.to_string())

    # Calculate average variations per original row for each category and split
    variations_per_original_row = {}
    for var in variations:
        original_id = var['original_row_index']  # Changed for consistency with bbq_task.py
        category = var['original_row_data']['category']
        split = var['original_row_data']['split']

        key = (category, split, original_id)
        variations_per_original_row[key] = variations_per_original_row.get(key, 0) + 1

    # Group by category and split
    category_split_variation_counts = {}
    for (category, split, _), counts in variations_per_original_row.items():
        key = (category, split)
        category_split_variation_counts[key] = category_split_variation_counts.get(key, []) + [counts]

    print("\nAverage variations per original row by Category and Split:")
    if task.target_category:
        # For a single category, just show split-wise average
        for (category, split), counts in category_split_variation_counts.items():
            if category == task.target_category:
                avg_variations = sum(counts) / len(counts)
                print(f"  Split: {split}: Average {avg_variations:.2f} variations per row")
    else:
        for (category, split), counts in category_split_variation_counts.items():
            avg_variations = sum(counts) / len(counts)
            print(f"  Category: {category}, Split: {split}: Average {avg_variations:.2f} variations per row")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AIR-Bench harmful prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row",
                        default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--category", type=str, help="Specific category to process (e.g., \"Child Harm\")", default=None)

    args = parser.parse_args()

    generate_airbench_variations(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.rows,
        max_variations_per_row=args.variations,
        random_seed=args.random_seed,
        target_category=args.category
    )
