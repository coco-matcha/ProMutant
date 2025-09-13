#!/usr/bin/env python3
"""
Compilable Mutant Sampler

This script filters compilable mutants from baseline output files and samples them
for manual analysis. It extracts full function context with diff markers for
each valid mutant.

Usage:
    python sample_compilable.py [options]
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List


def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON or JSONL file safely."""
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return []


def validate_mutant(mutant: Dict) -> bool:
    """
    Validate that a mutant meets basic requirements.
    
    Args:
        mutant: Mutant dictionary to validate
        
    Returns:
        bool: True if mutant is valid, False otherwise
    """
    instance_id = mutant.get('instance_id', 'Unknown')
    
    # Check for error entry
    if 'error' in mutant:
        print(f"Skipping error mutant: {instance_id}")
        return False
    
    # Check for required fields
    generated_code = mutant.get('generated_code', '').strip()
    original_code = mutant.get('original_code', '').strip()
    
    if not generated_code:
        print(f"No generated code for {instance_id}")
        return False
    
    if not original_code:
        print(f"No original code for {instance_id}")
        return False
    
    # Check if codes are identical
    if generated_code == original_code:
        print(f"Generated code same as original for {instance_id}")
        return False
    
    # Check for multi-line violations
    if '\n' in generated_code:
        print(f"Multi-line generated code for {instance_id}")
        return False
    
    return True


def find_original_item(instance_id: str, dataset: List[Dict]) -> Dict:
    """
    Find the original dataset item for a given instance ID.
    
    Args:
        instance_id: Instance ID to search for
        dataset: List of dataset items
        
    Returns:
        Original dataset item or None if not found
    """
    matching_items = [item for item in dataset if item.get('instance_id') == instance_id]
    
    if len(matching_items) != 1:
        print(f"Expected 1 original item for {instance_id}, found {len(matching_items)}")
        return None
    
    return matching_items[0]


def find_fixed_code_in_patch(mutant: Dict, original_item: Dict) -> bool:
    """
    Check if the mutant's original code can be found in the patch.
    
    Args:
        mutant: Mutant dictionary
        original_item: Original dataset item
        
    Returns:
        bool: True if original code found in patch, False otherwise
    """
    patch_lines = original_item.get('patch', '').split('\n')
    original_code = mutant['original_code']
    
    for line in patch_lines:
        if line.startswith('+') and original_code in line:
            return True
    
    print(f"Cannot find original code '{original_code}' in patch for {mutant['instance_id']}")
    return False


def extract_full_function_with_diff(mutant: Dict) -> str:
    """
    Extract full function with diff markers from mutant prompt.
    
    Args:
        mutant: Mutant dictionary containing prompt
        
    Returns:
        Full function with diff markers, or empty string if extraction fails
    """
    prompt = mutant.get('prompt', '')
    
    # Extract function content from prompt
    parts = prompt.split('Above is the original code', 1)
    if len(parts) < 2:
        print(f"Cannot parse prompt for {mutant['instance_id']}")
        return ""
    
    full_function = parts[0].strip()
    original_code = mutant['original_code']
    generated_code = mutant['generated_code']
    
    # Find the line containing the original code
    function_lines = full_function.split('\n')
    original_line = None
    
    for line in function_lines:
        if line.strip() == original_code.strip():
            original_line = line
            break
    
    if not original_line:
        print(f"Cannot find original code in function for {mutant['instance_id']}")
        return ""
    
    # Create diff-style replacement
    replacement_line = '\n+' + original_line.replace(original_code, generated_code)
    full_function_with_diff = full_function.replace(original_line, '-' + original_line + replacement_line)
    
    return full_function_with_diff


def filter_compilable_mutants(dataset_file: str, mutants_file: str) -> List[Dict]:
    """
    Filter mutants to find those that are compilable and valid.
    
    Args:
        dataset_file: Path to original SWE-Bench dataset file
        mutants_file: Path to mutants JSONL file
        
    Returns:
        List of valid compilable mutants
    """
    # Load data
    print(f"Loading dataset from {dataset_file}")
    dataset = load_json_file(dataset_file)
    if not dataset:
        return []
    
    print(f"Loading mutants from {mutants_file}")
    mutants = load_json_file(mutants_file)
    if not mutants:
        return []
    
    print(f"Processing {len(mutants)} mutants...")
    
    valid_mutants = []
    error_count = 0
    
    for mutant in mutants:
        instance_id = mutant.get('instance_id', 'Unknown')
        
        # Basic validation
        if not validate_mutant(mutant):
            error_count += 1
            continue
        
        # Find original dataset item
        original_item = find_original_item(instance_id, dataset)
        if not original_item:
            error_count += 1
            continue
        
        # Check if original code exists in patch
        if not find_fixed_code_in_patch(mutant, original_item):
            error_count += 1
            continue
        
        # Extract full function with diff markers
        full_function = extract_full_function_with_diff(mutant)
        if not full_function:
            error_count += 1
            continue
        
        # Create valid mutant entry
        valid_mutants.append({
            "mutant_id": 2000 + len(valid_mutants),
            "instance_id": instance_id,
            "original_code": mutant['original_code'],
            "generated_code": mutant['generated_code'],
            "full_function": full_function
        })
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total mutants processed: {len(mutants)}")
    print(f"  Invalid mutants: {error_count}")
    print(f"  Valid compilable mutants: {len(valid_mutants)}")
    
    return valid_mutants


def sample_mutants(mutants: List[Dict], sample_size: int, random_seed: int = 42) -> List[Dict]:
    """
    Sample mutants randomly from the valid set.
    
    Args:
        mutants: List of valid mutants
        sample_size: Number of mutants to sample
        random_seed: Random seed for reproducibility
        
    Returns:
        List of sampled mutants
    """
    if sample_size >= len(mutants):
        print(f"Sample size ({sample_size}) >= available mutants ({len(mutants)}), returning all")
        return mutants
    
    random.seed(random_seed)
    sampled = random.sample(mutants, sample_size)
    
    print(f"Sampled {len(sampled)} mutants from {len(mutants)} available")
    return sampled


def save_sampled_mutants(mutants: List[Dict], output_file: str, show_preview: int = 5) -> None:
    """
    Save sampled mutants to file and show preview.
    
    Args:
        mutants: List of sampled mutants
        output_file: Output file path
        show_preview: Number of mutants to preview (0 to skip preview)
    """
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(mutants, f, indent=2)
        print(f"Saved {len(mutants)} sampled mutants to {output_file}")
    except IOError as e:
        print(f"Error saving to {output_file}: {e}")
        return
    
    # Show preview
    if show_preview > 0 and mutants:
        print(f"\nPreview of first {min(show_preview, len(mutants))} mutants:")
        print("=" * 60)
        
        for i, mutant in enumerate(mutants[:show_preview]):
            print(f"Mutant {i+1} (ID: {mutant['mutant_id']}):")
            print(mutant.get('full_function', 'No function content'))
            print("=" * 60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter and sample compilable mutants for manual analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-file', '-d',
        default='data/swe_bench_lite/swe_bench_lite.json',
        help='Path to SWE-Bench dataset file'
    )
    
    parser.add_argument(
        '--mutants-file', '-m',
        default='data/llm_baseline/baseline_gpt-4o_limited_output.jsonl',
        help='Path to mutants JSONL file'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        default='data/human_study/baseline_sampled_mutants.json',
        help='Output file for sampled mutants'
    )
    
    parser.add_argument(
        '--sample-size', '-s',
        type=int,
        default=150,
        help='Number of mutants to sample'
    )
    
    parser.add_argument(
        '--random-seed', '-r',
        type=int,
        default=42,
        help='Random seed for reproducible sampling'
    )
    
    parser.add_argument(
        '--preview', '-p',
        type=int,
        default=5,
        help='Number of mutants to preview (0 to skip preview)'
    )
    
    args = parser.parse_args()
    
    try:
        # Filter compilable mutants
        valid_mutants = filter_compilable_mutants(args.dataset_file, args.mutants_file)
        
        if not valid_mutants:
            print("No valid compilable mutants found")
            sys.exit(1)
        
        # Sample mutants
        sampled_mutants = sample_mutants(valid_mutants, args.sample_size, args.random_seed)
        
        # Save and preview results
        save_sampled_mutants(sampled_mutants, args.output_file, args.preview)
        
        print("Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()