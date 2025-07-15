#!/usr/bin/env python3
"""
Unified Variation Analysis
Creates box plots comparing different models with their variations across AIR-Bench and BBQ datasets.
This focuses on analyzing how different prompt variations affect model performance across datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import json
import seaborn as sns
warnings.filterwarnings('ignore')

# Set font to Times New Roman for academic papers
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# ====== VISUALIZATION PARAMETERS ======
BOX_WIDTH = 0.6
BOX_SPACING = 0.8
FIGURE_WIDTH_PER_MODEL = 2.0
MIN_FIGURE_WIDTH = 14
FIGURE_HEIGHT = 10

# Colors and transparency
BOX_ALPHA = 0.7
MEDIAN_LINE_WIDTH = 2.0

# Font sizes
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
Y_TICK_LABEL_FONT_SIZE = 12
INFO_TEXT_FONT_SIZE = 11

# Models to analyze
MODELS = ['llama_3_3_70b', 'llama3_8b', 'llama_3_8b_8bit',
          'mistral_8b', ]

# Model display names mapping
MODEL_DISPLAY_NAMES = {
    'llama_3_3_70b': 'Llama-3.3-70B',
    'llama3_8b': 'Llama-3-8B',
    'llama_3_8b_8bit': 'Llama-3-8B (8-bit)',
    'mistral_8b': 'Mistral-8B'
}

# Dataset display names
DATASET_DISPLAY_NAMES = {
    'airbench': 'AIR-Bench',
    'bbq': 'BBQ'
}

# Colors for datasets
DATASET_COLORS = {
    'airbench': '#FF6B6B',  # Red
    'bbq': '#4ECDC4'        # Teal
}


def load_airbench_results(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load AIR-Bench results for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        DataFrame with AIR-Bench results or None if not found
    """
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "airbench" / model_name
    csv_file = results_dir / "airbench_variations_evaluated.csv"
    
    if not csv_file.exists():
        print(f"❌ AIR-Bench evaluated results not found for {model_name}: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded AIR-Bench evaluated data for {model_name}: {len(df)} rows")
        
        return df
    except Exception as e:
        print(f"❌ Error loading AIR-Bench data for {model_name}: {e}")
        return None


def load_bbq_results(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load BBQ results for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        DataFrame with BBQ results or None if not found
    """
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "bbq" / model_name
    csv_file = results_dir / "bbq_variations.csv"
    
    if not csv_file.exists():
        print(f"❌ BBQ results not found for {model_name}: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded BBQ data for {model_name}: {len(df)} rows")
        
        # Add category information from original data
        original_data_file = Path(__file__).parent.parent.parent / "data" / "generated_data" / "bbq" / "bbq_variations.json"
        if original_data_file.exists():
            try:
                with open(original_data_file, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                
                # Create mapping from original_row_index to category
                category_map = {}
                for item in original_data:
                    if 'original_row_index' in item and 'original_row_data' in item:
                        if 'category' in item['original_row_data']:
                            category_map[item['original_row_index']] = item['original_row_data']['category']
                
                # Add category column
                df['category'] = df['original_row_index'].map(category_map)
                print(f"✅ Added category information for {len(category_map)} rows")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load category information: {e}")
        
        return df
    except Exception as e:
        print(f"❌ Error loading BBQ data for {model_name}: {e}")
        return None


def clean_airbench_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert judge_score column to numeric values for AIR-Bench.
    Uses the same logic as analyze_airbench_variations.py.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with success_rate column
    """
    if 'judge_score' not in df.columns:
        print("❌ judge_score column not found in AIR-Bench data")
        return df
    
    # Convert to string first to handle any encoding issues
    df['judge_score'] = df['judge_score'].astype(str)
    
    # Extract numeric values from judge_score
    def extract_score(score_str):
        if pd.isna(score_str) or score_str == 'nan':
            return np.nan
        try:
            import re
            numbers = re.findall(r'\d+\.?\d*', str(score_str))
            if numbers:
                return float(numbers[0])
            return np.nan
        except:
            return np.nan
    
    df['judge_score_clean'] = df['judge_score'].apply(extract_score)
    
    # Print score distribution
    print(f"📊 Judge Score Distribution:")
    score_counts = df['judge_score_clean'].value_counts().sort_index()
    for score, count in score_counts.items():
        print(f"   Score {score}: {count} responses ({count/len(df)*100:.1f}%)")
    
    # Calculate success rate based on judge scores (1.0 = success, 0.0 = failure)
    df['success_rate'] = (df['judge_score_clean'] == 1.0).astype(float) * 100
    print(f"✅ Created success rate based on judge scores: {df['success_rate'].mean():.1f}% average")
    
    return df


def analyze_model_variations(model_name: str) -> Dict[str, pd.DataFrame]:
    """
    Analyze variations for a specific model across both datasets.
    
    Args:
        model_name: Name of the model to analyze
        
    Returns:
        Dictionary with variation analysis for each dataset
    """
    print(f"\n🔄 Analyzing variations for model: {model_name}")
    print("-" * 50)
    
    results = {}
    
    # Load and analyze AIR-Bench data
    airbench_df = load_airbench_results(model_name)
    if airbench_df is not None:
        airbench_df = clean_airbench_scores(airbench_df)
        
        # Group by variation index and calculate statistics
        agg_dict = {
            'success_rate': ['count', 'mean', 'std']
        }
        if 'category' in airbench_df.columns:
            agg_dict['category'] = 'first'
        
        variation_stats = airbench_df.groupby('variation_index').agg(agg_dict).round(3)
        
        # Flatten column names
        variation_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in variation_stats.columns]
        variation_stats = variation_stats.reset_index()
        
        # Rename columns for clarity
        variation_stats = variation_stats.rename(columns={
            'success_rate_count': 'count',
            'success_rate_mean': 'mean_score',
            'success_rate_std': 'std_score',
            'category_first': 'category'
        })
        
        results['airbench'] = variation_stats
        print(f"✅ AIR-Bench: {len(variation_stats)} variations, {variation_stats['count'].sum()} total responses")
    
    # Load and analyze BBQ data
    bbq_df = load_bbq_results(model_name)
    if bbq_df is not None:
        # Group by variation index and calculate statistics
        agg_dict = {
            'is_correct': ['count', 'mean', 'std']
        }
        if 'category' in bbq_df.columns:
            agg_dict['category'] = 'first'
        
        variation_stats = bbq_df.groupby('variation_index').agg(agg_dict).round(3)
        
        # Flatten column names
        variation_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in variation_stats.columns]
        variation_stats = variation_stats.reset_index()
        
        # Rename columns for clarity
        variation_stats = variation_stats.rename(columns={
            'is_correct_count': 'count',
            'is_correct_mean': 'mean_score',
            'is_correct_std': 'std_score',
            'category_first': 'category'
        })
        
        # Convert to percentage
        variation_stats['mean_score'] = variation_stats['mean_score'] * 100
        variation_stats['std_score'] = variation_stats['std_score'] * 100
        
        results['bbq'] = variation_stats
        print(f"✅ BBQ: {len(variation_stats)} variations, {variation_stats['count'].sum()} total responses")
    
    return results


def create_dataset_boxplots(all_model_results: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path):
    """
    Create separate box plots for each dataset comparing variations across models.
    
    Args:
        all_model_results: Dictionary with results for all models
        output_dir: Directory to save the plots
    """
    if not all_model_results:
        print("❌ No data available for plotting")
        return
    
    # Create separate plots for each dataset
    for dataset_name in ['airbench', 'bbq']:
        print(f"\n📊 Creating box plot for {dataset_name.upper()}...")
        
        # Prepare data for this dataset
        box_data = []
        labels = []
        colors = []
        info_lines = []
        
        for model_name in MODELS:
            if model_name not in all_model_results:
                continue
                
            model_results = all_model_results[model_name]
            
            variation_stats = model_results[dataset_name]
            
            # Prepare box plot data
            box_data.append(variation_stats['mean_score'].values)
            
            # Create label
            model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            labels.append(model_display)
            
            # Set color based on model (not dataset)
            model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            colors.append(model_colors[len(box_data) - 1])
            
            # Prepare info line
            total_responses = variation_stats['count'].sum()
            total_variations = len(variation_stats)
            avg_score = variation_stats['mean_score'].mean()
            info_lines.append(f"{model_display}: {total_variations} variations, {total_responses} responses, {avg_score:.1f}% avg")
        
        if not box_data:
            print(f"❌ No data available for {dataset_name}")
            continue
        
        # Create the plot
        figure_width = max(10, len(box_data) * 2.5)
        fig, ax = plt.subplots(1, 1, figsize=(figure_width, FIGURE_HEIGHT))
        
        # Create box plot
        positions = np.arange(1, len(box_data) + 1) * BOX_SPACING
        bp = ax.boxplot(box_data, patch_artist=True, labels=labels, widths=BOX_WIDTH, positions=positions)
        
        # Customize appearance
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_alpha(BOX_ALPHA)
        
        # Make median lines black and thicker
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(MEDIAN_LINE_WIDTH)
        
        # Customize plot
        dataset_display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
        ax.set_ylabel('Performance Score (%)', fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_xlabel('Model', fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_title(f'{dataset_display} Performance Variation Analysis\nModel Comparison', 
                     fontsize=TITLE_FONT_SIZE, fontweight='bold')
        
        # Add horizontal grid lines
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Set y-axis limits based on dataset
        if dataset_name == 'bbq':
            ax.set_ylim(20, 60)
        elif dataset_name == 'airbench':
            ax.set_ylim(10, 70)
        else:
            ax.set_ylim(0, 105)
        
        # Customize tick labels
        plt.xticks(positions, labels, rotation=0, fontsize=TICK_LABEL_FONT_SIZE)
        plt.yticks(fontsize=Y_TICK_LABEL_FONT_SIZE)
        
        # Add info text
        if info_lines:
            info_text = '\n'.join(info_lines)
            fig.text(0.02, 0.02, info_text, fontsize=INFO_TEXT_FONT_SIZE, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='lightgray', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        if info_lines:
            plt.subplots_adjust(bottom=0.25)
        
        # Save the plot
        output_filename = f'{dataset_name}_variation_analysis.png'
        plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{dataset_name}_variation_analysis.pdf', dpi=300, bbox_inches='tight')
        
        print(f"📊 {dataset_name} variation analysis plot saved to: {output_dir / output_filename}")
        
        plt.show()
        plt.close()





def generate_variation_report(all_model_results: Dict[str, Dict[str, pd.DataFrame]], output_dir: Path):
    """
    Generate a comprehensive report of the variation analysis.
    
    Args:
        all_model_results: Dictionary with results for all models
        output_dir: Directory to save the report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = []
    report.append("# Unified Variation Analysis Report")
    report.append("")
    report.append("## Overview")
    report.append("This report analyzes how different prompt variations affect model performance across AIR-Bench and BBQ datasets.")
    report.append("")
    
    # Overall statistics
    total_models = len(all_model_results)
    total_datasets = 0
    total_variations = 0
    total_responses = 0
    
    for model_name, model_results in all_model_results.items():
        for dataset_name, variation_stats in model_results.items():
            total_datasets += 1
            total_variations += len(variation_stats)
            total_responses += variation_stats['count'].sum()
    
    report.append(f"- **Models Analyzed:** {total_models}")
    report.append(f"- **Total Dataset-Model Combinations:** {total_datasets}")
    report.append(f"- **Total Variations:** {total_variations}")
    report.append(f"- **Total Responses:** {total_responses:,}")
    report.append("")
    
    # Detailed analysis for each model
    for model_name in MODELS:
        if model_name not in all_model_results:
            continue
            
        model_results = all_model_results[model_name]
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        report.append(f"## {model_display}")
        report.append("")
        
        for dataset_name in ['airbench', 'bbq']:
            if dataset_name not in model_results:
                continue
                
            variation_stats = model_results[dataset_name]
            dataset_display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            
            report.append(f"### {dataset_display}")
            report.append("")
            
            # Calculate statistics
            avg_score = variation_stats['mean_score'].mean()
            std_score = variation_stats['mean_score'].std()
            min_score = variation_stats['mean_score'].min()
            max_score = variation_stats['mean_score'].max()
            total_responses = variation_stats['count'].sum()
            
            report.append(f"- **Variations:** {len(variation_stats)}")
            report.append(f"- **Total Responses:** {total_responses:,}")
            report.append(f"- **Average Score:** {avg_score:.1f}%")
            report.append(f"- **Standard Deviation:** {std_score:.1f}%")
            report.append(f"- **Score Range:** {min_score:.1f}% - {max_score:.1f}%")
            report.append(f"- **Performance Spread:** {max_score - min_score:.1f}%")
            report.append("")
            
            # Top and bottom variations
            top_3 = variation_stats.nlargest(3, 'mean_score')
            bottom_3 = variation_stats.nsmallest(3, 'mean_score')
            
            report.append("#### Top 3 Variations")
            report.append("| Variation | Score | Count |")
            report.append("|-----------|-------|-------|")
            for _, row in top_3.iterrows():
                report.append(f"| {row['variation_index']} | {row['mean_score']:.1f}% | {row['count']} |")
            report.append("")
            
            report.append("#### Bottom 3 Variations")
            report.append("| Variation | Score | Count |")
            report.append("|-----------|-------|-------|")
            for _, row in bottom_3.iterrows():
                report.append(f"| {row['variation_index']} | {row['mean_score']:.1f}% | {row['count']} |")
            report.append("")
    
    # Cross-model comparison
    report.append("## Cross-Model Comparison")
    report.append("")
    
    comparison_data = []
    for model_name in MODELS:
        if model_name not in all_model_results:
            continue
            
        model_results = all_model_results[model_name]
        model_display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        
        for dataset_name in ['airbench', 'bbq']:
            if dataset_name not in model_results:
                continue
                
            variation_stats = model_results[dataset_name]
            dataset_display = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
            
            avg_score = variation_stats['mean_score'].mean()
            std_score = variation_stats['mean_score'].std()
            
            comparison_data.append({
                'Model': model_display,
                'Dataset': dataset_display,
                'Average Score': avg_score,
                'Std Dev': std_score,
                'Variations': len(variation_stats)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        report.append("### Performance Summary")
        report.append("| Model | Dataset | Avg Score | Std Dev | Variations |")
        report.append("|-------|---------|-----------|---------|------------|")
        for _, row in comparison_df.iterrows():
            report.append(f"| {row['Model']} | {row['Dataset']} | {row['Average Score']:.1f}% | {row['Std Dev']:.1f}% | {row['Variations']} |")
        report.append("")
    
    # Key insights
    report.append("## Key Insights")
    report.append("")
    
    # Find best and worst performing models
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        best_overall = comparison_df.loc[comparison_df['Average Score'].idxmax()]
        worst_overall = comparison_df.loc[comparison_df['Average Score'].idxmin()]
        
        report.append(f"- **Best Overall Performance:** {best_overall['Model']} on {best_overall['Dataset']} ({best_overall['Average Score']:.1f}%)")
        report.append(f"- **Worst Overall Performance:** {worst_overall['Model']} on {worst_overall['Dataset']} ({worst_overall['Average Score']:.1f}%)")
        report.append(f"- **Performance Range:** {best_overall['Average Score'] - worst_overall['Average Score']:.1f}%")
        report.append("")
        
        # Find most and least consistent models
        most_consistent = comparison_df.loc[comparison_df['Std Dev'].idxmin()]
        least_consistent = comparison_df.loc[comparison_df['Std Dev'].idxmax()]
        
        report.append(f"- **Most Consistent:** {most_consistent['Model']} on {most_consistent['Dataset']} (Std Dev: {most_consistent['Std Dev']:.1f}%)")
        report.append(f"- **Least Consistent:** {least_consistent['Model']} on {least_consistent['Dataset']} (Std Dev: {least_consistent['Std Dev']:.1f}%)")
        report.append("")
    
    # Save report
    with open(output_dir / 'unified_variation_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Unified variation analysis report saved to: {output_dir / 'unified_variation_analysis_report.md'}")


def main():
    """
    Main function to run the unified variation analysis.
    """
    print("🔄 Starting Unified Variation Analysis")
    print("=" * 60)
    
    # Set up output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "output" / "unified_variation_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze all models
    all_model_results = {}
    
    for model_name in MODELS:
        model_results = analyze_model_variations(model_name)
        if model_results:
            all_model_results[model_name] = model_results
    
    if not all_model_results:
        print("❌ No data found for any model")
        return
    
    print(f"\n📊 Creating visualizations...")
    
    # Create separate box plots for each dataset
    create_dataset_boxplots(all_model_results, output_dir)
    
    print(f"\n✅ Unified variation analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main() 