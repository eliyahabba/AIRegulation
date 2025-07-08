#!/usr/bin/env python3
"""
BBQ Task: Bias in QA
This module provides a class for generating prompt variations for the BBQ (Bias Benchmark for QA) dataset,
focusing on identifying and evaluating biases in language model responses.
"""

import argparse
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

        questions, answer_choices, correct_labels = self._load_bbq_dataset(split="test")

        data_rows = []
        for i in range(len(questions)):
            data_rows.append({
                QUESTION_KEY: questions[i],
                "ans0": answer_choices[i][0],
                "ans1": answer_choices[i][1],
                "ans2": answer_choices[i][2],
                GOLD_KEY: correct_labels[i],  # The correct label (0, 1, or 2)
                "id": f"bbq_question_{i}",  # Unique ID for the question
                "split": "test"  # Always use test split for now
            })

        df = pd.DataFrame(data_rows)
        self.ps.load_dataframe(df)
        print(f"✅ Loaded BBQ dataset: {len(self.ps.data)} rows")

    @staticmethod
    def _load_bbq_dataset(split: str = "test") -> Tuple[List[str], List[List[str]], List[int]]:
        """
        Loads the BBQ dataset using the datasets library.
        Assumes each row has:
            - context, question
            - ans0, ans1, ans2
            - label (int: 0, 1, or 2)
        """
        # BBQ categories available in heegyu/bbq
        categories = [
            "Age", "Disability_status", "Gender_identity", "Nationality",
            "Physical_appearance", "Race_ethnicity", "Religion", "Sexual_orientation", "Socioeconomic_status"
        ]

        all_questions = []
        all_answer_choices = []
        all_correct_labels = []

        for category in categories:
            try:
                print(f"Loading BBQ category: {category}")
                dataset = load_dataset("heegyu/bbq", category, split=split)

                questions = [f"{ex['context']} {ex['question']}" for ex in dataset]
                answer_choices = [[ex["ans0"], ex["ans1"], ex["ans2"]] for ex in dataset]
                correct_labels = dataset["label"]

                all_questions.extend(questions)
                all_answer_choices.extend(answer_choices)
                all_correct_labels.extend(correct_labels)

                print(f"✅ Loaded {len(questions)} examples from {category}")

            except Exception as e:
                print(f"Warning: Could not load data for category {category}. Error: {e}")
                continue

        if not all_questions:
            raise ValueError("No BBQ data loaded. Check dataset availability.")

        print(f"✅ Total BBQ examples loaded: {len(all_questions)}")
        return all_questions, all_answer_choices, all_correct_labels

    def get_template(self) -> Dict[str, Any]:
        """
        Get template configuration for BBQ task.
        """
        return {
            INSTRUCTION: "Read the following context and question carefully, then choose the best answer from the given options.",
            # INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
            PROMPT_FORMAT: f"Context and Question: {{{QUESTION_KEY}}}\nOptions:\n0: {{ans0}}\n1: {{ans1}}\n2: {{ans2}}\nYour Answer (0, 1, or 2):",
            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
            GOLD_KEY: GOLD_KEY,  # Gold key points to the label column in the dataframe
        }


def generate_bbq_variations(variations_per_field, api_platform, model_name, max_rows, max_variations_per_row,
                            random_seed):
    """Generate variations for BBQ dataset."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / "data" / "bbq"
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
    print(f"   Max rows: {task.max_rows}")
    print(f"   Max variations per row: {task.max_variations_per_row}")
    print(f"   Random seed: {task.random_seed}")
    task.ps.configure(
        max_rows=task.max_rows,
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

        original_row_index = var.get('original_row_index')
        if original_row_index is not None and original_row_index < len(task.ps.data):
            original_gold_label = task.ps.data.loc[original_row_index, GOLD_KEY]
            print(f"Gold Label: {original_gold_label}")
        else:
            print(f"Gold Label: N/A (Original row not found for index {original_row_index})")

        print("-" * 50)

    # Export results using the correct path
    print(f"\n6. Exporting results to {output_file}...")
    task.ps.export(str(output_file), format="json")
    print("✅ Export completed!")

    # Show final statistics
    print("\n7. Final statistics:")
    task.ps.info()


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
