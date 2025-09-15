import json
import random
from typing import List, Dict
from collections import defaultdict
import difflib

def create_diff(code1, code2):
    """
    Create a simple +/- diff between two code strings.
    
    Args:
        code1: First code section (string)
        code2: Second code section (string)
    
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

def filter_compilable(context_file_path: str, outpur_file_path: str, model_name: str = "gpt-4") -> List[Dict]:
    """
    Create the final prediction format with patches.
    
    Args:
        results: Results from OpenAI API
        original_data: Original JSON data with patches
        model_name: Name of the model used
        
    Returns:
        List of predictions in required format
    """
    
    with open(context_file_path, 'r') as f:
        original_data = json.load(f)
        f.close()
    with open(output_file_path, 'r') as f:
        results = [json.loads(line) for line in f]
        f.close()

    predictions = []

    # For summarizing
    error_mutant = 0

    for mutant in results:
        instance_id = mutant['instance_id']
        fixed_code = mutant['fixed_code']
        if fixed_code == "":
            error_mutant += 1
            print("No fixed code")
            continue
        buggy_code = mutant['buggy_code']

        if 'error' in mutant:
            error_mutant += 1
            print("Encountered error mutant with ID ",instance_id)
            continue
            
        # Get original item to get function content
        original_item = None
        for instance in original_data:
            i_fixed_code = instance.get('fixed_code', '')
            i_buggy_code = instance.get('buggy_code', '')
            if instance['instance_id'] == instance_id and i_fixed_code == fixed_code and i_buggy_code == buggy_code:
                original_item = instance
                break
        if not original_item:
            error_mutant += 1
            print("Cannot find corresponding context for mutant ", instance_id)

        if mutant['generated_code'] == "":
            error_mutant += 1
            continue
        # If generated code is the same as original code, skip
        if mutant['generated_code'].strip() == mutant['buggy_code'].strip():
            error_mutant += 1
            print(f"Generated code for {instance_id} the same as context code, skipping")
            continue
        if mutant['generated_code'].strip() == mutant['fixed_code'].strip():
            error_mutant += 1
            print(f"Generated code for {instance_id} the same as fixed code, skipping")
            continue

        # At this point we're certain that the instance is one of those compilable in the baseline code

        diff = create_diff(fixed_code, mutant['generated_code'])

        # Get full function content
        full_function = original_item.get('patched_function', '')

        if fixed_code not in full_function:
            error_mutant += 1
            print("Cannot find fixed code for instance ", instance_id)
            continue

        full_function = full_function.replace(fixed_code, diff)
        
        # Create prediction object
        predictions.append({
            "mutant_id": 1000 + len(predictions),
            "instance_id": instance_id,
            "original_code": mutant['fixed_code'],
            "generated_code": mutant['generated_code'],
            "full_function": full_function
        })

    # Summary
    print(f"📊 Summary:")
    print(f"  Total mutants: {len(results)}")
    print(f"  Error in mutants: {error_mutant}")
    print(f"  Total mutants: {len(predictions)}")
    
    return predictions


if __name__ == "__main__":
    context_file_path = 'data/swe_bench_lite/extracted_code_contexts.json'
    output_file_path = 'data/promutant/baseline_config_gpt-4o_output.jsonl'

    predictions = filter_compilable(context_file_path, output_file_path)

    random.seed(42)
    sampled_mutants = random.sample(predictions, 150) 

    for i in sampled_mutants:
        print(i.get('full_function', ''))
        print("="*60)

    with open('data/human_study/promutant_sampled_mutants.json', 'w') as f:
        json.dump(sampled_mutants, f, indent=2)