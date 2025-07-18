#!/usr/bin/env python3
"""
AIR-Bench Variation Analysis
Focuses on analyzing differences between prompt variations and their performance.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple
import warnings
from scipy import stats
from src.constants import get_model_dir_name
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_airbench_results(model_name: str = "llama_3_3_70b", quantization: str = None) -> pd.DataFrame:
    """
    Load AIR-Bench results from CSV file.
    
    Args:
        model_name: Name of the model directory to analyze
        quantization: Quantization type ('8bit', '4bit', or None)
        
    Returns:
        DataFrame with the results
    """
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "airbench"
    model_dir_name = get_model_dir_name(model_name, quantization)
    csv_file = results_dir / model_dir_name / "airbench_variations_evaluated.csv"
    
    if not csv_file.exists():
        print(f"❌ Results file not found: {csv_file}")
        return None
    
    print(f"📁 Loading results from: {csv_file}")
    
    try:
        # Read CSV with proper encoding and handle potential issues
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def clean_judge_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and convert judge_score column to numeric values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned judge_score
    """
    if 'judge_score' not in df.columns:
        print("❌ judge_score column not found")
        return df
    
    # Convert to string first to handle any encoding issues
    df['judge_score'] = df['judge_score'].astype(str)
    
    # Extract numeric values from judge_score
    def extract_score(score_str):
        if pd.isna(score_str) or score_str == 'nan':
            return np.nan
        try:
            # Try to extract numeric value
            import re
            numbers = re.findall(r'\d+\.?\d*', str(score_str))
            if numbers:
                return float(numbers[0])
            return np.nan
        except:
            return np.nan
    
    df['judge_score_clean'] = df['judge_score'].apply(extract_score)
    
    # Print score distribution
    print(f"\n📊 Judge Score Distribution:")
    score_counts = df['judge_score_clean'].value_counts().sort_index()
    for score, count in score_counts.items():
        print(f"   Score {score}: {count} responses ({count/len(df)*100:.1f}%)")
    
    return df


def analyze_variation_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance by variation index to see if certain variations perform better.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with variation analysis
    """
    if 'variation_index' not in df.columns:
        print("❌ variation_index column not found")
        return {}
    
    print(f"\n🔄 Variation Performance Analysis:")
    print("=" * 80)
    
    # Group by variation index and calculate statistics
    variation_stats = df.groupby('variation_index').agg({
        'judge_score_clean': ['count', 'mean', 'std'],
        'category': 'first'
    }).round(3)
    
    # Flatten column names - handle the multi-level columns properly
    variation_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in variation_stats.columns]
    variation_stats = variation_stats.reset_index()
    
    # Rename columns for clarity
    variation_stats = variation_stats.rename(columns={
        'judge_score_clean_count': 'count',
        'judge_score_clean_mean': 'mean_score',
        'judge_score_clean_std': 'std_score',
        'category_first': 'category'
    })
    
    # Add success rates
    variation_stats['success_rate'] = 0.0
    variation_stats['failure_rate'] = 0.0
    
    for idx, row in variation_stats.iterrows():
        var_data = df[df['variation_index'] == row['variation_index']]
        scores = var_data['judge_score_clean'].dropna()
        
        if len(scores) > 0:
            variation_stats.loc[idx, 'success_rate'] = (scores == 1.0).mean() * 100
            variation_stats.loc[idx, 'failure_rate'] = (scores == 0.0).mean() * 100
    
    # Find best and worst variations
    best_variation = variation_stats.loc[variation_stats['success_rate'].idxmax()]
    worst_variation = variation_stats.loc[variation_stats['success_rate'].idxmin()]
    
    print(f"📊 Found {len(variation_stats)} unique variations")
    print(f"🏆 Best Variation: {best_variation['variation_index']} (Success Rate: {best_variation['success_rate']:.1f}%)")
    print(f"📉 Worst Variation: {worst_variation['variation_index']} (Success Rate: {worst_variation['success_rate']:.1f}%)")
    print(f"📈 Performance Spread: {best_variation['success_rate'] - worst_variation['success_rate']:.1f}%")
    
    # Show all variations with their counts
    print(f"\n📋 All Variations by Success Rate:")
    variation_stats_sorted = variation_stats.sort_values('success_rate', ascending=False)
    for _, row in variation_stats_sorted.iterrows():
        print(f"   Variation {row['variation_index']}: {row['success_rate']:.1f}% success, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    # Show top 5 and bottom 5 variations
    print(f"\n🏅 Top 5 Variations by Success Rate:")
    top_5 = variation_stats.nlargest(5, 'success_rate')
    for _, row in top_5.iterrows():
        print(f"   Variation {row['variation_index']}: {row['success_rate']:.1f}% success, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    print(f"\n📉 Bottom 5 Variations by Success Rate:")
    bottom_5 = variation_stats.nsmallest(5, 'success_rate')
    for _, row in bottom_5.iterrows():
        print(f"   Variation {row['variation_index']}: {row['success_rate']:.1f}% success, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    return {
        'variation_stats': variation_stats,
        'best_variation': best_variation,
        'worst_variation': worst_variation,
        'top_5': top_5,
        'bottom_5': bottom_5
    }


def analyze_variation_failures(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze failure rates (score 0.0) by variation index.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with failure analysis
    """
    if 'variation_index' not in df.columns:
        print("❌ variation_index column not found")
        return {}
    
    print(f"\n❌ Variation Failure Analysis:")
    print("=" * 80)
    
    # Group by variation index and calculate failure statistics
    failure_stats = df.groupby('variation_index').agg({
        'judge_score_clean': ['count', 'mean', 'std'],
        'category': 'first'
    }).round(3)
    
    # Flatten column names - handle the multi-level columns properly
    failure_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in failure_stats.columns]
    failure_stats = failure_stats.reset_index()
    
    # Rename columns for clarity
    failure_stats = failure_stats.rename(columns={
        'judge_score_clean_count': 'count',
        'judge_score_clean_mean': 'mean_score',
        'judge_score_clean_std': 'std_score',
        'category_first': 'category'
    })
    
    # Add failure rates
    failure_stats['failure_rate'] = 0.0
    
    for idx, row in failure_stats.iterrows():
        var_data = df[df['variation_index'] == row['variation_index']]
        scores = var_data['judge_score_clean'].dropna()
        
        if len(scores) > 0:
            failure_stats.loc[idx, 'failure_rate'] = (scores == 0.0).mean() * 100
    
    # Find worst and best variations (from failure perspective)
    worst_variation = failure_stats.loc[failure_stats['failure_rate'].idxmax()]
    best_variation = failure_stats.loc[failure_stats['failure_rate'].idxmin()]
    
    print(f"📊 Found {len(failure_stats)} unique variations")
    print(f"❌ Worst Variation (Most Failures): {worst_variation['variation_index']} (Failure Rate: {worst_variation['failure_rate']:.1f}%)")
    print(f"✅ Best Variation (Least Failures): {best_variation['variation_index']} (Failure Rate: {best_variation['failure_rate']:.1f}%)")
    print(f"📈 Failure Rate Spread: {worst_variation['failure_rate'] - best_variation['failure_rate']:.1f}%")
    
    # Show all variations with their counts
    print(f"\n📋 All Variations by Failure Rate:")
    failure_stats_sorted = failure_stats.sort_values('failure_rate', ascending=False)
    for _, row in failure_stats_sorted.iterrows():
        print(f"   Variation {row['variation_index']}: {row['failure_rate']:.1f}% failures, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    # Show top 5 and bottom 5 variations (from failure perspective)
    print(f"\n❌ Top 5 Variations by Failure Rate:")
    top_5_failures = failure_stats.nlargest(5, 'failure_rate')
    for _, row in top_5_failures.iterrows():
        print(f"   Variation {row['variation_index']}: {row['failure_rate']:.1f}% failures, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    print(f"\n✅ Bottom 5 Variations by Failure Rate:")
    bottom_5_failures = failure_stats.nsmallest(5, 'failure_rate')
    for _, row in bottom_5_failures.iterrows():
        print(f"   Variation {row['variation_index']}: {row['failure_rate']:.1f}% failures, {row['mean_score']:.3f} avg score, {row['count']} responses")
    
    return {
        'variation_stats': failure_stats,
        'worst_variation': worst_variation,
        'best_variation': best_variation,
        'top_5_failures': top_5_failures,
        'bottom_5_failures': bottom_5_failures
    }


def analyze_category_variation_interaction(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how different variations perform across different categories.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with category-variation interaction analysis
    """
    print(f"\n🎯 Category-Variation Interaction Analysis:")
    print("=" * 80)
    
    # Create pivot table: categories vs variations
    pivot_data = df.pivot_table(
        values='judge_score_clean',
        index='category',
        columns='variation_index',
        aggfunc='mean'
    )
    
    # Calculate category-specific statistics
    category_variation_stats = {}
    
    for category in df['category'].unique():
        if pd.isna(category):
            continue
            
        cat_data = df[df['category'] == category]
        variations = cat_data['variation_index'].unique()
        
        category_variation_stats[category] = {
            'total_variations': len(variations),
            'best_variation': None,
            'worst_variation': None,
            'variation_performance': {},
            'performance_spread': 0.0
        }
        
        for var in variations:
            var_data = cat_data[cat_data['variation_index'] == var]
            scores = var_data['judge_score_clean'].dropna()
            
            if len(scores) > 0:
                success_rate = (scores == 1.0).mean() * 100
                avg_score = scores.mean()
                
                category_variation_stats[category]['variation_performance'][var] = {
                    'success_rate': success_rate,
                    'avg_score': avg_score,
                    'count': len(scores)
                }
        
        # Find best and worst variations for this category
        if category_variation_stats[category]['variation_performance']:
            best_var = max(category_variation_stats[category]['variation_performance'].items(), 
                          key=lambda x: x[1]['success_rate'])
            worst_var = min(category_variation_stats[category]['variation_performance'].items(), 
                           key=lambda x: x[1]['success_rate'])
            
            category_variation_stats[category]['best_variation'] = best_var
            category_variation_stats[category]['worst_variation'] = worst_var
            category_variation_stats[category]['performance_spread'] = best_var[1]['success_rate'] - worst_var[1]['success_rate']
    
    # Print summary for each category
    for category, stats in category_variation_stats.items():
        if stats['best_variation'] and stats['worst_variation']:
            print(f"\n📋 {category}:")
            print(f"   Best variation: {stats['best_variation'][0]} ({stats['best_variation'][1]['success_rate']:.1f}% success)")
            print(f"   Worst variation: {stats['worst_variation'][0]} ({stats['worst_variation'][1]['success_rate']:.1f}% success)")
            print(f"   Performance spread: {stats['performance_spread']:.1f}%")
    
    return {
        'pivot_data': pivot_data,
        'category_variation_stats': category_variation_stats
    }


def create_variation_focused_visualizations(df: pd.DataFrame, variation_analysis: Dict, 
                                          failure_analysis: Dict, category_variation_analysis: Dict, output_dir: Path, model_dir_name: str):
    """
    Create visualizations focused on variation differences.
    
    Args:
        df: DataFrame with results
        variation_analysis: Results from variation analysis
        failure_analysis: Results from failure analysis
        category_variation_analysis: Results from category-variation analysis
        output_dir: Directory to save plots
        model_dir_name: Name of the model directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a larger figure to accommodate both heatmaps
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(bottom=0.22)
    
    # 1. Variation success rate ranking (sorted from lowest to highest)
    plt.subplot(3, 3, 1)
    variation_stats = variation_analysis['variation_stats']
    variation_stats_sorted = variation_stats.sort_values('success_rate')  # Sort by success rate
    
    bars = plt.bar(range(len(variation_stats_sorted)), variation_stats_sorted['success_rate'], 
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.title('Success Rate by Variation (Score 1.0)\nSorted Low to High', fontsize=12, fontweight='bold')
    plt.xlabel('Variation Index', fontsize=10)
    plt.ylabel('Success Rate (%)', fontsize=10)
    plt.xticks(range(len(variation_stats_sorted)), variation_stats_sorted['variation_index'], rotation=0)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, variation_stats_sorted['success_rate'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    # הוסף תוויות n=... מתחת לציר X
    ax = plt.gca()
    for i, count in enumerate(variation_stats_sorted['count']):
        ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color='gray', clip_on=False)
    
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Success rate heatmap
    plt.subplot(3, 3, (2, 3))  # Takes up top right
    if 'pivot_data' in category_variation_analysis:
        # Convert to success rates
        success_pivot = df.pivot_table(
            values='judge_score_clean',
            index='category',
            columns='variation_index',
            aggfunc=lambda x: (x == 1.0).mean() * 100
        )
        
        # Sort columns by overall success rate for better visualization
        overall_success_rates = success_pivot.mean()
        success_pivot = success_pivot[overall_success_rates.sort_values().index]
        
        sns.heatmap(success_pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Success Rate (%)'}, 
                   linewidths=0.5, linecolor='white')
        plt.title('Success Rate Heatmap: Categories vs Variations', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Category', fontsize=10)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    
    # 3. Failure rate by variation (sorted from highest to lowest - worse to better)
    plt.subplot(3, 3, 4)
    if failure_analysis:
        failure_stats = failure_analysis['variation_stats']
        failure_stats_sorted = failure_stats.sort_values('failure_rate', ascending=False)  # Sort by failure rate (high to low)
        
        bars = plt.bar(range(len(failure_stats_sorted)), failure_stats_sorted['failure_rate'], 
                      alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        plt.title('Failure Rate by Variation (Score 0.0)\nSorted High to Low (Worse to Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Failure Rate (%)', fontsize=10)
        plt.xticks(range(len(failure_stats_sorted)), failure_stats_sorted['variation_index'], rotation=0)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, failure_stats_sorted['failure_rate'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        # הוסף תוויות n=... מתחת לציר X
        ax = plt.gca()
        for i, count in enumerate(failure_stats_sorted['count']):
            ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='gray', clip_on=False)
        
        plt.grid(axis='y', alpha=0.3)
    
    # 4. Failure rate heatmap
    plt.subplot(3, 3, (5, 6))  # Takes up middle right
    if 'pivot_data' in category_variation_analysis:
        # Convert to failure rates
        failure_pivot = df.pivot_table(
            values='judge_score_clean',
            index='category',
            columns='variation_index',
            aggfunc=lambda x: (x == 0.0).mean() * 100
        )
        
        # Sort columns by overall failure rate for better visualization
        overall_failure_rates = failure_pivot.mean()
        failure_pivot = failure_pivot[overall_failure_rates.sort_values().index]
        
        sns.heatmap(failure_pivot, annot=True, fmt='.0f', cmap='Reds', 
                   cbar_kws={'label': 'Failure Rate (%)'}, 
                   linewidths=0.5, linecolor='white')
        plt.title('Failure Rate Heatmap: Categories vs Variations', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Category', fontsize=10)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    
    # 5. Combined comparison (bottom)
    plt.subplot(3, 3, (7, 9))  # Takes up bottom row
    
    # Prepare data for comparison - use same sorting for both
    if variation_analysis and failure_analysis:
        # Sort by success rate and use the same order for both
        success_stats = variation_analysis['variation_stats'].sort_values('success_rate')
        
        # Get failure rates in the same order as success stats
        failure_stats = failure_analysis['variation_stats'].set_index('variation_index').loc[success_stats['variation_index']].reset_index()
        
        x = np.arange(len(success_stats))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, success_stats['success_rate'], width, 
                       label='Success Rate (Score 1.0) → Higher is Better', alpha=0.8, color='steelblue')
        bars2 = plt.bar(x + width/2, failure_stats['failure_rate'], width, 
                       label='Failure Rate (Score 0.0) → Lower is Better', alpha=0.8, color='coral')
        
        plt.title('Success vs Failure Rates by Variation\nSorted by Success Rate (Low to High)', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Rate (%)', fontsize=10)
        plt.xticks(x, success_stats['variation_index'], rotation=0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        # הוסף תוויות n=... מתחת לציר X
        ax = plt.gca()
        for i, count in enumerate(success_stats['count']):
            ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='gray', clip_on=False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_dir_name}_variation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhanced variation visualizations saved to: {output_dir / f'{model_dir_name}_variation_analysis.png'}")


def analyze_airbench_variations(model_name: str = "llama_3_3_70b", quantization: str = None, output_dir: str = None):
    """
    Perform comprehensive analysis of AIR-Bench results focusing on variation differences.
    
    Args:
        model_name: Name of the model directory to analyze
        quantization: Quantization type ('8bit', '4bit', or None)
        output_dir: Directory to save analysis outputs
    """
    model_dir_name = get_model_dir_name(model_name, quantization)
    print(f"🔄 AIR-Bench Variation Analysis for model: {model_dir_name}")
    print("=" * 80)
    
    # Load and clean data
    df = load_airbench_results(model_name, quantization)
    if df is None:
        return
    
    df = clean_judge_score(df)

    # Perform analyses
    variation_analysis = analyze_variation_performance(df)
    failure_analysis = analyze_variation_failures(df)
    category_variation_analysis = analyze_category_variation_interaction(df)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"/ "output"/  "airbench"
    else:
        output_dir = Path(output_dir)
    
    # Create visualizations
    create_variation_focused_visualizations(df, variation_analysis, failure_analysis, category_variation_analysis, output_dir, model_dir_name)
    
    print(f"\n✅ Variation analysis complete! Results saved to: {output_dir}")
    print("=" * 80)


def analyze_airbench_variations_batch(model_names: List[str] = None, quantization: str = None, output_dir: str = None):
    """
    Perform comprehensive analysis of AIR-Bench results for multiple models.
    
    Args:
        model_names: List of model names to analyze (default: ['llama_3_3_70b', 'llama3_8b', 'mistral_8b'])
        quantization: Quantization type ('8bit', '4bit', or None)
        output_dir: Directory to save analysis outputs
    """
    if model_names is None:
        model_names = ['llama_3_3_70b', 'llama3_8b', 'mistral_8b', 'mixtral_8x7b', 'phi_4_mini', 'olmo_2_13b', 'qwen3_8b']
    
    print(f"🚀 Starting batch analysis for {len(model_names)} models: {', '.join(model_names)}")
    print("=" * 80)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data"/ "output"/  "airbench"
    else:
        output_dir = Path(output_dir)
    
    successful_models = []
    failed_models = []
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n📊 Processing model {i}/{len(model_names)}: {model_name}")
        print("-" * 60)
        
        try:
            analyze_airbench_variations(model_name, quantization, output_dir)
            successful_models.append(model_name)
        except Exception as e:
            print(f"❌ Error processing model {model_name}: {e}")
            failed_models.append(model_name)
    
    # Summary
    print(f"\n🎯 Batch Analysis Summary:")
    print("=" * 80)
    print(f"✅ Successfully processed: {len(successful_models)} models")
    if successful_models:
        print(f"   - {', '.join(successful_models)}")
    
    print(f"❌ Failed to process: {len(failed_models)} models")
    if failed_models:
        print(f"   - {', '.join(failed_models)}")
    
    print(f"📁 All results saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="AIR-Bench variation analysis focusing on differences between prompt variations")
    parser.add_argument("--model", default=None, 
                       help="Model name to analyze (default: runs batch analysis on ['llama_3_3_70b', 'llama3_8b', 'mistral_8b'])")
    parser.add_argument("--models", nargs='+', default=None,
                       help="List of model names to analyze in batch")
    parser.add_argument("--quantization", default=None, choices=["8bit", "4bit", "none"],
                       help="Quantization type (8bit, 4bit, or none)")
    parser.add_argument("--output", default=None,
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # If specific model is provided, run single model analysis
    if args.model:
        analyze_airbench_variations(model_name=args.model, quantization=args.quantization, output_dir=args.output)
    else:
        # Run batch analysis
        model_list = args.models if args.models else ['llama_3_3_70b', 'llama3_8b', 'mistral_8b', 'mixtral_8x7b', 'phi_4_mini', 'olmo_2_13b', 'qwen3_8b']
        analyze_airbench_variations_batch(model_names=model_list, quantization=args.quantization, output_dir=args.output)


if __name__ == "__main__":
    main() 