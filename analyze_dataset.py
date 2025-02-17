import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from loguru import logger
import os
from typing import Dict, List
import numpy as np

def load_dataset(filepath: str) -> List[Dict]:
    """Load the prepared dataset from JSON file"""
    logger.info(f"Loading dataset from {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_tag_distribution(dataset: List[Dict]) -> Dict[str, int]:
    """Analyze the distribution of thinks across different similarity tags"""
    tag_counts = Counter()
    for entry in dataset:
        for tag, thinks in entry['thinks'].items():
            tag_counts[tag] += len(thinks)
    return dict(tag_counts)

def analyze_prompt_lengths(dataset: List[Dict]) -> List[int]:
    """Analyze the distribution of prompt lengths"""
    return [len(entry['prompt'].split()) for entry in dataset]

def analyze_thinks_lengths(dataset: List[Dict]) -> List[int]:
    """Analyze the distribution of thinks lengths for each tag"""
    thinks_lengths = {
        tag: [] for tag in dataset[0]['thinks'].keys()
    }
    
    for entry in dataset:
        for tag, thinks in entry['thinks'].items():
            thinks_lengths[tag].extend([len(think.split()) for think in thinks])
    
    return thinks_lengths

def plot_tag_distribution(tag_counts: Dict[str, int], output_dir: str):
    """Plot the distribution of similarity tags"""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(tag_counts.keys()), y=list(tag_counts.values()))
    plt.title('Distribution of Similarity Tags')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tag_distribution.png'))
    plt.close()

def plot_length_distributions(lengths: List[int], title: str, output_path: str):
    """Plot length distribution histograms"""
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=50)
    plt.title(f'{title} Length Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_thinks_length_boxplot(thinks_lengths: Dict[str, List[int]], output_dir: str):
    """Plot box plot of thinks lengths by tag"""
    plt.figure(figsize=(12, 6))
    data = []
    labels = []
    for tag, lengths in thinks_lengths.items():
        data.extend(lengths)
        labels.extend([tag] * len(lengths))
    
    df = pd.DataFrame({'Tag': labels, 'Length': data})
    sns.boxplot(x='Tag', y='Length', data=df)
    plt.title('Distribution of Thinks Lengths by Tag')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thinks_lengths_by_tag.png'))
    plt.close()

def main():
    # Create output directory for plots
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset("data_output/prepared_ds.json")
    logger.info(f"Loaded dataset with {len(dataset)} entries")
    
    # Analyze tag distribution
    tag_counts = analyze_tag_distribution(dataset)
    logger.info("Tag distribution:")
    for tag, count in tag_counts.items():
        logger.info(f"{tag}: {count}")
    
    # Analyze lengths
    prompt_lengths = analyze_prompt_lengths(dataset)
    thinks_lengths = analyze_thinks_lengths(dataset)
    
    # Calculate statistics
    logger.info("\nPrompt length statistics:")
    logger.info(f"Mean: {np.mean(prompt_lengths):.2f} words")
    logger.info(f"Median: {np.median(prompt_lengths):.2f} words")
    logger.info(f"Min: {min(prompt_lengths)} words")
    logger.info(f"Max: {max(prompt_lengths)} words")
    
    logger.info("\nThinks length statistics by tag:")
    for tag, lengths in thinks_lengths.items():
        logger.info(f"\n{tag}:")
        logger.info(f"Mean: {np.mean(lengths):.2f} words")
        logger.info(f"Median: {np.median(lengths):.2f} words")
        logger.info(f"Min: {min(lengths)} words")
        logger.info(f"Max: {max(lengths)} words")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    plot_tag_distribution(tag_counts, output_dir)
    plot_length_distributions(
        prompt_lengths,
        "Prompt",
        os.path.join(output_dir, 'prompt_lengths.png')
    )
    plot_thinks_length_boxplot(thinks_lengths, output_dir)
    
    logger.success("Analysis completed! Check analysis_output directory for plots.")

if __name__ == "__main__":
    main() 