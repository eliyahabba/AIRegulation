#!/usr/bin/env python3
"""
BBQ Variation Analysis
Focuses on analyzing differences between prompt variations and their performance on BBQ dataset.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
from typing import Dict, Any, List, Tuple
import warnings
from scipy import stats
from src.constants import get_model_dir_name
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def load_bbq_results(model_name: str = "llama_3_3_70b", quantization: str = None) -> pd.DataFrame:
    """
    Load BBQ results from CSV file.
    
    Args:
        model_name: Name of the model directory to analyze
        quantization: Quantization type ('8bit', '4bit', or None)
        
    Returns:
        DataFrame with the results
    """
    results_dir = Path(__file__).parent.parent.parent / "data" / "results" / "bbq"
    model_dir_name = get_model_dir_name(model_name, quantization)
    csv_file = results_dir / model_dir_name / "bbq_variations.csv"
    
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


def load_bbq_categories(model_name: str = "llama_3_3_70b") -> Dict[int, str]:
    """
    Load category information from the original BBQ data.
    
    Args:
        model_name: Name of the model directory
        
    Returns:
        Dictionary mapping original_row_index to category
    """
    data_file = Path(__file__).parent.parent.parent / "data" / "generated_data" / "bbq" / "bbq_variations.json"
    
    if not data_file.exists():
        print(f"❌ Original data file not found: {data_file}")
        return {}
    
    print(f"📁 Loading categories from: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        categories = {}
        for item in data:
            if 'original_row_index' in item and 'original_row_data' in item:
                if 'category' in item['original_row_data']:
                    categories[item['original_row_index']] = item['original_row_data']['category']
        
        print(f"✅ Loaded categories for {len(categories)} rows")
        return categories
    except Exception as e:
        print(f"❌ Error loading categories: {e}")
        return {}


def merge_bbq_with_categories(df: pd.DataFrame, categories: Dict[int, str]) -> pd.DataFrame:
    """
    Merge BBQ results with category information.
    
    Args:
        df: DataFrame with BBQ results
        categories: Dictionary mapping original_row_index to category
        
    Returns:
        DataFrame with category column added
    """
    if df is None or not categories:
        return df
    
    # Add category column
    df['category'] = df['original_row_index'].map(categories)
    
    # Report on missing categories
    missing_categories = df['category'].isna().sum()
    if missing_categories > 0:
        print(f"⚠️ Warning: {missing_categories} rows missing category information")
    
    print(f"📊 Categories distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"   {category}: {count} responses")
    
    return df


def analyze_variation_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze performance by variation index.
    
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
    agg_dict = {
        'is_correct': ['count', 'mean', 'std']
    }
    if 'category' in df.columns:
        agg_dict['category'] = 'first'
    
    variation_stats = df.groupby('variation_index').agg(agg_dict).round(3)
    
    # Flatten column names
    variation_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in variation_stats.columns]
    variation_stats = variation_stats.reset_index()
    
    # Rename columns for clarity
    variation_stats = variation_stats.rename(columns={
        'is_correct_count': 'count',
        'is_correct_mean': 'accuracy',
        'is_correct_std': 'std_accuracy',
        'category_first': 'category'
    })
    
    # Convert accuracy to percentage
    variation_stats['accuracy_pct'] = variation_stats['accuracy'] * 100
    
    # Find best and worst variations
    best_variation = variation_stats.loc[variation_stats['accuracy_pct'].idxmax()]
    worst_variation = variation_stats.loc[variation_stats['accuracy_pct'].idxmin()]
    
    print(f"📊 Found {len(variation_stats)} unique variations")
    print(f"🏆 Best Variation: {best_variation['variation_index']} (Accuracy: {best_variation['accuracy_pct']:.1f}%)")
    print(f"📉 Worst Variation: {worst_variation['variation_index']} (Accuracy: {worst_variation['accuracy_pct']:.1f}%)")
    print(f"📈 Performance Spread: {best_variation['accuracy_pct'] - worst_variation['accuracy_pct']:.1f}%")
    
    # Show all variations with their counts
    print(f"\n📋 All Variations by Accuracy:")
    variation_stats_sorted = variation_stats.sort_values('accuracy_pct', ascending=False)
    for _, row in variation_stats_sorted.iterrows():
        print(f"   Variation {row['variation_index']}: {row['accuracy_pct']:.1f}% accuracy, {row['count']} responses")
    
    # Show top 5 and bottom 5 variations
    print(f"\n🏅 Top 5 Variations by Accuracy:")
    top_5 = variation_stats.nlargest(5, 'accuracy_pct')
    for _, row in top_5.iterrows():
        print(f"   Variation {row['variation_index']}: {row['accuracy_pct']:.1f}% accuracy, {row['count']} responses")
    
    print(f"\n📉 Bottom 5 Variations by Accuracy:")
    bottom_5 = variation_stats.nsmallest(5, 'accuracy_pct')
    for _, row in bottom_5.iterrows():
        print(f"   Variation {row['variation_index']}: {row['accuracy_pct']:.1f}% accuracy, {row['count']} responses")
    
    return {
        'variation_stats': variation_stats,
        'best_variation': best_variation,
        'worst_variation': worst_variation,
        'top_5': top_5,
        'bottom_5': bottom_5
    }


def analyze_variation_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze error rates (is_correct=False) by variation index.
    
    Args:
        df: DataFrame with results
        
    Returns:
        Dictionary with error analysis
    """
    if 'variation_index' not in df.columns:
        print("❌ variation_index column not found")
        return {}
    
    print(f"\n❌ Variation Error Analysis:")
    print("=" * 80)
    
    # Group by variation index and calculate error statistics
    agg_dict = {
        'is_correct': ['count', 'mean']
    }
    if 'category' in df.columns:
        agg_dict['category'] = 'first'
    
    error_stats = df.groupby('variation_index').agg(agg_dict).round(3)
    
    # Flatten column names
    error_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in error_stats.columns]
    error_stats = error_stats.reset_index()
    
    # Rename columns for clarity
    error_stats = error_stats.rename(columns={
        'is_correct_count': 'count',
        'is_correct_mean': 'accuracy',
        'category_first': 'category'
    })
    
    # Add error rates
    error_stats['error_rate'] = (1 - error_stats['accuracy']) * 100
    
    # Find worst and best variations (from error perspective)
    worst_variation = error_stats.loc[error_stats['error_rate'].idxmax()]
    best_variation = error_stats.loc[error_stats['error_rate'].idxmin()]
    
    print(f"📊 Found {len(error_stats)} unique variations")
    print(f"❌ Worst Variation (Most Errors): {worst_variation['variation_index']} (Error Rate: {worst_variation['error_rate']:.1f}%)")
    print(f"✅ Best Variation (Least Errors): {best_variation['variation_index']} (Error Rate: {best_variation['error_rate']:.1f}%)")
    print(f"📈 Error Rate Spread: {worst_variation['error_rate'] - best_variation['error_rate']:.1f}%")
    
    # Show all variations with their counts
    print(f"\n📋 All Variations by Error Rate:")
    error_stats_sorted = error_stats.sort_values('error_rate', ascending=False)
    for _, row in error_stats_sorted.iterrows():
        print(f"   Variation {row['variation_index']}: {row['error_rate']:.1f}% errors, {row['count']} responses")
    
    # Show top 5 and bottom 5 variations (from error perspective)
    print(f"\n❌ Top 5 Variations by Error Rate:")
    top_5_errors = error_stats.nlargest(5, 'error_rate')
    for _, row in top_5_errors.iterrows():
        print(f"   Variation {row['variation_index']}: {row['error_rate']:.1f}% errors, {row['count']} responses")
    
    print(f"\n✅ Bottom 5 Variations by Error Rate:")
    bottom_5_errors = error_stats.nsmallest(5, 'error_rate')
    for _, row in bottom_5_errors.iterrows():
        print(f"   Variation {row['variation_index']}: {row['error_rate']:.1f}% errors, {row['count']} responses")
    
    return {
        'variation_stats': error_stats,
        'worst_variation': worst_variation,
        'best_variation': best_variation,
        'top_5_errors': top_5_errors,
        'bottom_5_errors': bottom_5_errors
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
        values='is_correct',
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
            accuracy = var_data['is_correct'].mean() * 100
            count = len(var_data)
            
            category_variation_stats[category]['variation_performance'][var] = {
                'accuracy': accuracy,
                'count': count
            }
        
        # Find best and worst variations for this category
        if category_variation_stats[category]['variation_performance']:
            best_var = max(category_variation_stats[category]['variation_performance'].items(), 
                          key=lambda x: x[1]['accuracy'])
            worst_var = min(category_variation_stats[category]['variation_performance'].items(), 
                           key=lambda x: x[1]['accuracy'])
            
            category_variation_stats[category]['best_variation'] = best_var
            category_variation_stats[category]['worst_variation'] = worst_var
            category_variation_stats[category]['performance_spread'] = best_var[1]['accuracy'] - worst_var[1]['accuracy']
    
    # Print summary for each category
    for category, stats in category_variation_stats.items():
        if stats['best_variation'] and stats['worst_variation']:
            print(f"\n📋 {category}:")
            print(f"   Best variation: {stats['best_variation'][0]} ({stats['best_variation'][1]['accuracy']:.1f}% accuracy)")
            print(f"   Worst variation: {stats['worst_variation'][0]} ({stats['worst_variation'][1]['accuracy']:.1f}% accuracy)")
            print(f"   Performance spread: {stats['performance_spread']:.1f}%")
    
    return {
        'pivot_data': pivot_data,
        'category_variation_stats': category_variation_stats
    }


def create_variation_focused_visualizations(df: pd.DataFrame, variation_analysis: Dict, 
                                          error_analysis: Dict, category_variation_analysis: Dict, output_dir: Path, model_dir_name: str):
    """
    Create visualizations focused on variation differences.
    
    Args:
        df: DataFrame with results
        variation_analysis: Results from variation analysis
        error_analysis: Results from error analysis
        category_variation_analysis: Results from category-variation analysis
        output_dir: Directory to save plots
        model_dir_name: Name of the model directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a larger figure to accommodate visualizations
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(bottom=0.22)
    
    # 1. Variation accuracy ranking (sorted from lowest to highest)
    plt.subplot(3, 3, 1)
    variation_stats = variation_analysis['variation_stats']
    variation_stats_sorted = variation_stats.sort_values('accuracy_pct')
    
    bars = plt.bar(range(len(variation_stats_sorted)), variation_stats_sorted['accuracy_pct'], 
                   alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    plt.title('Accuracy by Variation\nSorted Low to High', fontsize=12, fontweight='bold')
    plt.xlabel('Variation Index', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.xticks(range(len(variation_stats_sorted)), variation_stats_sorted['variation_index'], rotation=0)
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, variation_stats_sorted['accuracy_pct'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
    # Add count labels
    ax = plt.gca()
    for i, count in enumerate(variation_stats_sorted['count']):
        ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                    ha='center', va='top', fontsize=7, color='gray', clip_on=False)
    
    plt.grid(axis='y', alpha=0.3)
    
    # 2. Accuracy heatmap
    plt.subplot(3, 3, (2, 3))  # Takes up top right
    if 'pivot_data' in category_variation_analysis:
        # Convert to accuracy percentages
        accuracy_pivot = category_variation_analysis['pivot_data'] * 100
        
        # Sort columns by overall accuracy for better visualization
        overall_accuracy_rates = accuracy_pivot.mean()
        accuracy_pivot = accuracy_pivot[overall_accuracy_rates.sort_values().index]
        
        sns.heatmap(accuracy_pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy (%)'}, 
                   linewidths=0.5, linecolor='white')
        plt.title('Accuracy Heatmap: Categories vs Variations', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Category', fontsize=10)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    
    # 3. Error rate by variation (sorted from highest to lowest - worse to better)
    plt.subplot(3, 3, 4)
    if error_analysis:
        error_stats = error_analysis['variation_stats']
        error_stats_sorted = error_stats.sort_values('error_rate', ascending=False)
        
        bars = plt.bar(range(len(error_stats_sorted)), error_stats_sorted['error_rate'], 
                      alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        plt.title('Error Rate by Variation\nSorted High to Low (Worse to Better)', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Error Rate (%)', fontsize=10)
        plt.xticks(range(len(error_stats_sorted)), error_stats_sorted['variation_index'], rotation=0)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, error_stats_sorted['error_rate'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        # Add count labels
        ax = plt.gca()
        for i, count in enumerate(error_stats_sorted['count']):
            ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='gray', clip_on=False)
        
        plt.grid(axis='y', alpha=0.3)
    
    # 4. Error rate heatmap
    plt.subplot(3, 3, (5, 6))  # Takes up middle right
    if 'pivot_data' in category_variation_analysis:
        # Convert to error rates
        error_pivot = (1 - category_variation_analysis['pivot_data']) * 100
        
        # Sort columns by overall error rate for better visualization
        overall_error_rates = error_pivot.mean()
        error_pivot = error_pivot[overall_error_rates.sort_values().index]
        
        sns.heatmap(error_pivot, annot=True, fmt='.0f', cmap='Reds', 
                   cbar_kws={'label': 'Error Rate (%)'}, 
                   linewidths=0.5, linecolor='white')
        plt.title('Error Rate Heatmap: Categories vs Variations', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Category', fontsize=10)
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
    
    # 5. Combined comparison (bottom)
    plt.subplot(3, 3, (7, 9))  # Takes up bottom row
    
    # Prepare data for comparison - use same sorting for both
    if variation_analysis and error_analysis:
        # Sort by accuracy and use the same order for both
        accuracy_stats = variation_analysis['variation_stats'].sort_values('accuracy_pct')
        
        # Get error rates in the same order as accuracy stats
        error_stats = error_analysis['variation_stats'].set_index('variation_index').loc[accuracy_stats['variation_index']].reset_index()
        
        x = np.arange(len(accuracy_stats))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, accuracy_stats['accuracy_pct'], width, 
                       label='Accuracy → Higher is Better', alpha=0.8, color='steelblue')
        bars2 = plt.bar(x + width/2, error_stats['error_rate'], width, 
                       label='Error Rate → Lower is Better', alpha=0.8, color='coral')
        
        plt.title('Accuracy vs Error Rates by Variation\nSorted by Accuracy (Low to High)', fontsize=12, fontweight='bold')
        plt.xlabel('Variation Index', fontsize=10)
        plt.ylabel('Rate (%)', fontsize=10)
        plt.xticks(x, accuracy_stats['variation_index'], rotation=0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        # Add count labels
        ax = plt.gca()
        for i, count in enumerate(accuracy_stats['count']):
            ax.annotate(f'n={int(count)}', xy=(i, 0), xytext=(0, -28), textcoords='offset points',
                        ha='center', va='top', fontsize=7, color='gray', clip_on=False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_dir_name}_bbq_variation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Enhanced BBQ variation visualizations saved to: {output_dir / f'{model_dir_name}_bbq_variation_analysis.png'}")


def analyze_bbq_variations(model_name: str = "llama_3_3_70b", quantization: str = None, output_dir: str = None):
    """
    Perform comprehensive analysis of BBQ results focusing on variation differences.
    
    Args:
        model_name: Name of the model directory to analyze
        quantization: Quantization type ('8bit', '4bit', or None)
        output_dir: Directory to save analysis outputs
    """
    model_dir_name = get_model_dir_name(model_name, quantization)
    print(f"🔄 BBQ Variation Analysis for model: {model_dir_name}")
    print("=" * 80)
    
    # Load and clean data
    df = load_bbq_results(model_name, quantization)
    if df is None:
        return
    
    # Load categories and merge
    categories = load_bbq_categories(model_name)
    df = merge_bbq_with_categories(df, categories)
    
    # Perform analyses
    variation_analysis = analyze_variation_performance(df)
    error_analysis = analyze_variation_errors(df)
    category_variation_analysis = analyze_category_variation_interaction(df)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "output" / "bbq"
    else:
        output_dir = Path(output_dir)
    
    # Create visualizations
    create_variation_focused_visualizations(df, variation_analysis, error_analysis, category_variation_analysis, output_dir, model_dir_name)
    
    print(f"\n✅ BBQ variation analysis complete! Results saved to: {output_dir}")
    print("=" * 80)


def analyze_bbq_variations_batch(model_names: List[str] = None, quantization: str = None, output_dir: str = None):
    """
    Perform comprehensive analysis of BBQ results for multiple models.
    
    Args:
        model_names: List of model names to analyze (default: ['llama_3_3_70b', 'llama3_8b', 'mistral_8b'])
        quantization: Quantization type ('8bit', '4bit', or None)
        output_dir: Directory to save analysis outputs
    """
    if model_names is None:
        model_names = ['llama_3_3_70b', 'llama3_8b', 'mistral_8b']
    
    print(f"🚀 Starting batch analysis for {len(model_names)} models: {', '.join(model_names)}")
    print("=" * 80)
    
    # Create output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "output" / "bbq"
    else:
        output_dir = Path(output_dir)
    
    successful_models = []
    failed_models = []
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n📊 Processing model {i}/{len(model_names)}: {model_name}")
        print("-" * 60)
        
        try:
            analyze_bbq_variations(model_name, quantization, output_dir)
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
    parser = argparse.ArgumentParser(description="BBQ variation analysis focusing on differences between prompt variations")
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
        analyze_bbq_variations(model_name=args.model, quantization=args.quantization, output_dir=args.output)
    else:
        # Run batch analysis
        model_list = args.models if args.models else ['llama_3_3_70b', 'llama3_8b', 'mistral_8b']
        analyze_bbq_variations_batch(model_names=model_list, quantization=args.quantization, output_dir=args.output)


if __name__ == "__main__":
    main() 