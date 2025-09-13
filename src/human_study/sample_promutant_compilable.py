#!/usr/bin/env python3
"""
Back Translation Mutant Sampler

This script filters compilable mutants from back translation output files and samples them
for manual analysis. It creates diff visualizations and extracts full function context.

Usage:
    python sample_compilable.py [options]
"""

import argparse
import difflib
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


def create_diff(code1: str, code2: str) -> str:
    """
    Create a simple +/- diff between two code strings.
    
    Args:
        code1: First code section (original)
        code2: Second code section (modified)
    
    Returns:
        String with diff in +/- format
    """
    lines1 = code1.splitlines()
    lines2 = code2.splitlines()
    
    result = []
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Lines are identical
            for line in lines1[i1:i2]:
                result.append(f" {line}")
        
        elif tag == 'delete':
            # Lines only in first code
            for line in lines1[i1:i2]:
                result.append(f"-{line}")
        
        elif tag == 'insert':
            # Lines only in second code
            for line in lines2[j1:j2]:
                result.append(f"+{line}")
        
        elif tag == 'replace':
            # Lines changed
            for line in lines1[i1:i2]:
                result.append(f"-{line}")
            for line in lines2[j1:j2]:
                result.append(f"+{line}")
    
    return '\n'.join(result)


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
    fixed_code = mutant.get('fixed_code', '').strip()
    if not fixed_code:
        print(f"No fixed code for {instance_id}")
        return False
    
    generated_code = mutant.get('generated_code', '').strip()
    if not generated_code:
        print(f"No generated code for {instance_id}")
        return False
    
    buggy_code = mutant.get('buggy_code', '').strip()
    
    # Check if generated code is same as existing codes
    if generated_code == buggy_code:
        print(f"Generated code same as buggy code for {instance_id}")
        return False
    
    if generated_code == fixed_code:
        print(f"Generated code same as fixed code for {instance_id}")
        return False
    
    return True


def find_matching_context(mutant: Dict, contexts: List[Dict]) -> Optional[Dict]:
    """
    Find the matching context for a mutant based on instance_id and code matching.
    
    Args:
        mutant: Mutant dictionary
        contexts: List of context dictionaries
        
    Returns:
        Matching context or None if not found
    """
    instance_id = mutant['instance_id']
    fixed_code = mutant['fixed_code']
    buggy_code = mutant['buggy_code']
    
    for context in contexts:
        if (context['instance_id'] == instance_id and 
            context.get('fixed_code', '') == fixed_code and 
            context.get('buggy_code', '') == buggy_code):
            return context
    
    print(f"Cannot find corresponding context for mutant {instance_id}")
    return None


def create_function_with_diff(context: Dict, fixed_code: str, generated_code: str) -> Optional[str]:
    """
    Create full function with diff markers showing the mutation.
    
    Args:
        context: Context dictionary containing function information
        fixed_code: Original fixed code
        generated_code: Generated mutant code
        
    Returns:
        Full function with diff markers, or None if creation fails
    """
    full_function = context.get('patched_function', '')
    
    if not full_function:
        print(f"No patched function found for {context['instance_id']}")
        return None
    
    if fixed_code not in full_function:
        print(f"Cannot find fixed code in function for {context['instance_id']}")
        return None
    
    # Create diff and replace in function
    diff = create_diff(fixed_code, generated_code)
    modified_function = full_function.replace(fixed_code, diff)
    
    return modified_function


def filter_compilable_mutants(context_file: str, mutants_file: str) -> List[Dict]:
    """
    Filter mutants to find those that are compilable and valid.
    
    Args:
        context_file: Path to buggy code contexts JSON file
        mutants_file: Path to mutants JSONL file
        
    Returns:
        List of valid compilable mutants
    """
    # Load data
    print(f"Loading contexts from {context_file}")
    contexts = load_json_file(context_file)
    if not contexts:
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
        
        # Find matching context
        context = find_matching_context(mutant, contexts)
        if not context:
            error_count += 1
            continue
        
        # Create function with diff
        full_function = create_function_with_diff(
            context, 
            mutant['fixed_code'], 
            mutant['generated_code']
        )
        
        if not full_function:
            error_count += 1
            continue
        
        # Create valid mutant entry
        valid_mutants.append({
            "mutant_id": 1000 + len(valid_mutants),
            "instance_id": instance_id,
            "original_code": mutant['fixed_code'],
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


def save_sampled_mutants(mutants: List[Dict], output_file: str, 
                        show_preview: int = 5, preview_start: int = 0) -> None:
    """
    Save sampled mutants to file and show preview.
    
    Args:
        mutants: List of sampled mutants
        output_file: Output file path
        show_preview: Number of mutants to preview (0 to skip preview)
        preview_start: Starting index for preview
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
        preview_end = min(preview_start + show_preview, len(mutants))
        if preview_start >= len(mutants):
            print(f"Preview start index ({preview_start}) >= number of mutants ({len(mutants)})")
            return
        
        print(f"\nPreview of mutants {preview_start+1}-{preview_end}:")
        print("=" * 60)
        
        for i in range(preview_start, preview_end):
            mutant = mutants[i]
            print(f"Mutant {i+1} (ID: {mutant['mutant_id']}):")
            print(mutant.get('full_function', 'No function content'))
            print("=" * 60)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter and sample compilable mutants from back translation output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--context-file', '-c',
        default='data/swe_bench_lite/buggy_code_contexts.json',
        help='Path to buggy code contexts JSON file'
    )
    
    parser.add_argument(
        '--mutants-file', '-m',
        default='data/promutant/baseline_config_gpt-4o_output.jsonl',
        help='Path to mutants JSONL file'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        default='data/human_study/promutant_sampled_mutants.json',
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
    
    parser.add_argument(
        '--preview-start',
        type=int,
        default=0,
        help='Starting index for preview (useful for checking different ranges)'
    )
    
    args = parser.parse_args()
    
    try:
        # Filter compilable mutants
        valid_mutants = filter_compilable_mutants(args.context_file, args.mutants_file)
        
        if not valid_mutants:
            print("No valid compilable mutants found")
            sys.exit(1)
        
        # Sample mutants
        sampled_mutants = sample_mutants(valid_mutants, args.sample_size, args.random_seed)
        
        # Save and preview results
        save_sampled_mutants(
            sampled_mutants, 
            args.output_file, 
            args.preview, 
            args.preview_start
        )
        
        print("Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()