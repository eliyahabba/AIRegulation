#!/usr/bin/env python3
"""
BBQ Task: Bias in QA
This module provides a class for generating prompt variations for the BBQ (Bias Benchmark for QA) dataset,
focusing on identifying and evaluating biases in language model responses.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset
from promptsuite.core.template_keys import (PROMPT_FORMAT_VARIATIONS,
                                            INSTRUCTION, PROMPT_FORMAT, GOLD_KEY,
                                            FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION,
                                            ENUMERATE_VARIATION, SHUFFLE_VARIATION, FEW_SHOT_KEY,
                                            INSTRUCTION_VARIATIONS, PARAPHRASE_WITH_LLM
                                            )

from base_task import BaseTask
from src.constants import (
    DEFAULT_VARIATIONS_PER_FIELD, DEFAULT_PLATFORM, DEFAULT_MODEL_NAME,
    DEFAULT_MAX_VARIATIONS_PER_ROW, DEFAULT_MAX_ROWS, DEFAULT_RANDOM_SEED
)


class BBQTask(BaseTask):
    """Task for generating BBQ bias prompt variations with associated metadata."""

    def __init__(self, variations_per_field: int = DEFAULT_VARIATIONS_PER_FIELD, api_platform: str = DEFAULT_PLATFORM,
                 model_name: str = DEFAULT_MODEL_NAME,
                 max_rows: int = DEFAULT_MAX_ROWS, max_variations_per_row: int = DEFAULT_MAX_VARIATIONS_PER_ROW,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        task_name = "BBQ Bias in QA"
        output_filename = "bbq_variations.json"

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
        Load BBQ dataset, combining context and question, and preparing answer choices.
        """
        print("Loading BBQ dataset...")

        df = self._load_bbq_dataset(split="test", max_rows_per_category=self.max_rows)
        self.post_process(df)
        print(f"✅ Loaded BBQ dataset: {len(df)} rows")
        print("✅ Data loaded and post-processed")

    def post_process(self, df) -> None:
        """
        Post-process BBQ data to create choices format and correct answer index.
        Also creates a manual train/test split if data is only from one split.
        """
        print("Post-processing BBQ data...")

        # Create choices list combining all answer options
        df['choices'] = df.apply(lambda row: [
            row['ans0'],
            row['ans1'],
            row['ans2']
        ], axis=1)

        # Use the original 'label' column as the answer
        df['answer'] = df['label']

        # Create a 50/50 train/test split within the loaded data, grouped by category
        print("Creating 50/50 train/test split within each category...")
        train_indices = df.groupby('category').sample(frac=0.5, random_state=self.random_seed).index
        df['split'] = 'test'  # Default to test
        df.loc[train_indices, 'split'] = 'train'
        
        train_count = len(df[df['split'] == 'train'])
        test_count = len(df[df['split'] == 'test'])
        print(f"✅ Created splits: {train_count} train, {test_count} test")

        # Update max_rows to reflect actual data size for ps.configure
        self.actual_max_rows = len(df)

        self.ps.load_dataframe(df)

        print(f"✅ Created choices field with 3 options per question")
        print(f"✅ Mapped gold label to answer index field")

    @staticmethod
    def _load_bbq_dataset(split: str = "test", max_rows_per_category: int = None) -> pd.DataFrame:
        """
        Loads the BBQ dataset using the datasets library.
        Assumes each row has:
            - context, question
            - ans0, ans1, ans2
            - label (int: 0, 1, or 2)
        """
        # BBQ categories available in heegyu/bbq
        categories = ["Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
                      "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "SES", "Sexual_orientation"]

        all_data = []
        question_id = 0
        
        if max_rows_per_category:
            rows_to_load_per_category = max_rows_per_category * 2
            print(f"Loading {rows_to_load_per_category} examples from each of {len(categories)} categories...")

        for category in categories:
            try:
                print(f"Loading BBQ category: {category} ({split} split)")
                dataset = load_dataset("heegyu/bbq", category, split=split)
                
                # Convert to pandas and process immediately
                df = dataset.to_pandas()
                
                # Sample from this category if max_rows_per_category is specified
                if max_rows_per_category and len(df) > rows_to_load_per_category:
                    df = df.sample(n=rows_to_load_per_category, random_state=42)
                
                # Add only the fields we need to add
                df['id'] = [f"bbq_question_{question_id + i}" for i in range(len(df))]
                df['split'] = split # This split will be overwritten in post_process
                
                all_data.append(df)
                question_id += len(df)
                
                if max_rows_per_category and len(df) == rows_to_load_per_category:
                    print(f"✅ Loaded {len(df)} examples from {category}")
                else:
                    print(f"✅ Loaded {len(df)} examples from {category}")

            except Exception as e:
                print(f"Warning: Could not load data for category {category}. Error: {e}")
                continue

        if not all_data:
            raise ValueError("No BBQ data loaded. Check dataset availability.")

        # Combine all categories
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # No need to shuffle here, post_process will handle the final split and shuffle
        if max_rows_per_category:
            print(f"✅ Total BBQ examples loaded: {len(combined_df)} (loaded {max_rows_per_category * 2} per category for split)")
        else:
            print(f"✅ Total BBQ examples loaded: {len(combined_df)}")
        return combined_df

    def get_template(self) -> Dict[str, Any]:
        """
        Get template configuration for BBQ task.
        """
        return {
            INSTRUCTION: "Read the following context and question carefully, then choose the best answer from the given options.",
            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: "Question: {question}\nChoices: {choices}\nAnswer:\n{label}",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
            'choices': [SHUFFLE_VARIATION, ENUMERATE_VARIATION],
            GOLD_KEY: {
                'field': 'label',
                'type': 'index',
                'options_field': 'choices'
            },
            FEW_SHOT_KEY: {
                'count': 3,  # Use 5 few-shot examples
                'format': 'different_examples__different_order_per_variation',
                'split': 'train',  # Use training split for few-shot examples
                 "filter_by": "category",  # Filter by category
            }
        }


def generate_bbq_variations(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row,
                            random_seed):
    """Generate variations for BBQ dataset."""
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "generated_data" / "bbq"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "bbq_variations.json"

    task = BBQTask(
        variations_per_field=variations_per_field,
        api_platform=api_platform,
        model_name=model_name,
        max_rows=max_rows,
        max_variations_per_row=max_variations_per_row,
        random_seed=random_seed
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
        model_name=model_name
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
    category_split_counts = original_df.groupby(['category', 'split']).size().unstack(fill_value=0)
    print(category_split_counts.to_string())

    # Calculate average variations per original row for each category and split
    variations_per_original_row = {}
    for var in variations:
        original_id = var['original_row_index']
        category = var['original_row_data']['category']
        split = var['original_row_data']['split']
        
        key = (category, split, original_id)
        variations_per_original_row[key] = variations_per_original_row.get(key, 0) + 1
    
    # Group by category and split
    category_split_variation_counts = {}
    for (category, split, _), count in variations_per_original_row.items():
        key = (category, split)
        category_split_variation_counts[key] = category_split_variation_counts.get(key, []) + [count]

    print("\nAverage variations per original row by Category and Split:")
    for (category, split), counts in category_split_variation_counts.items():
        avg_variations = sum(counts) / len(counts)
        print(f"  Category: {category}, Split: {split}: Average {avg_variations:.2f} variations per row")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BBQ harmful prompt variations")
    parser.add_argument("--rows", type=int, help="Number of rows to process", default=DEFAULT_MAX_ROWS)
    parser.add_argument("--variations", type=int, help="Number of variations per row",
                        default=DEFAULT_MAX_VARIATIONS_PER_ROW)
    parser.add_argument("--variations_per_field", type=int, default=DEFAULT_VARIATIONS_PER_FIELD)
    parser.add_argument("--api_platform", type=str, default=DEFAULT_PLATFORM)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--random_seed", type=int, help="Random seed for generation", default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    generate_bbq_variations(
        variations_per_field=args.variations_per_field,
        api_platform=args.api_platform,
        model_name=args.model_name,
        max_rows=args.rows,
        max_variations_per_row=args.variations,
        random_seed=args.random_seed
    )
