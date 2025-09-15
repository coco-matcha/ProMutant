import json
import random
from typing import List, Dict
from collections import defaultdict

def filter_compilable(dataset_file_path: str, output_file_path: str, model_name: str = "gpt-4") -> List[Dict]:
    """
    Create the final prediction format with patches.
    
    Args:
        results: Results from OpenAI API
        original_data: Original JSON data with patches
        model_name: Name of the model used
        
    Returns:
        List of predictions in required format
    """
    
    with open(dataset_file_path, 'r') as f:
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

        if 'error' in mutant:
            error_mutant += 1
            print("Encountered error mutant with ID ",instance_id)
            continue
            
        # Get original item to get patch content
        original_item = [instance for instance in original_data if instance.get('instance_id')==instance_id]
        assert len(original_item)==1, f"Expected to find 1 item matching ID with mutant but found {len(original_item)}"
        original_item = original_item[0]

        # If there is no replacement code, skip
        if mutant['generated_code'] == "":
            error_mutant += 1
            print("No generated code found")
            continue
        # If there is no original code, skip
        if mutant['original_code'] == "":
            error_mutant += 1
            print("No original code found")
            continue
        # If generated code is the same as original code, skip
        if mutant['generated_code'].strip() == mutant['original_code'].strip():
            error_mutant += 1
            print(f"Generated code for {instance_id} the same as fixed code, skipping")
            continue
        # If the mutant is multi-line, this violates the condition of the test
        if '\n' in mutant['generated_code']:
            error_mutant += 1
            print(f"Generated code for {instance_id} in the wrong format, spanning multiple lines.")
            continue
    
        patch_lines = original_item.get('patch').split('\n')
        fixed_code = None
        for l in patch_lines:
            # Find fixed code line in original fixed code, if not exist then must be error
            if l.startswith('+') and mutant['original_code'] in l:
                fixed_code = l
                break
        if not fixed_code:
            error_mutant += 1
            print(f"Cannot find original code {mutant['original_code']} in patch for {instance_id}")
            continue

        # At this point we're certain that the instance is one of those compilable in the baseline code

        # Get full function content
        full_function = mutant['prompt'].split('Above is the original code', 1)
        full_function = full_function[0].strip()
        
        replace_line = None
        original_line = None
        function_lines = full_function.split('\n')
        for line in function_lines:
            if line.strip() == mutant['original_code'].strip():
                original_line = line
                replace_line = '\n+' + line.replace(mutant['original_code'], mutant['generated_code']) 
                break
        if not replace_line:
            error_mutant += 1
            print("Cannot find original code in function for instance", instance_id)
            continue
        
        full_function = full_function.replace(original_line, '-' + original_line + replace_line)
        
        # Create prediction object
        predictions.append({
            "mutant_id": 2000 + len(predictions),
            "instance_id": instance_id,
            "original_code": mutant['original_code'],
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
    dataset_file_path = 'data/swe_bench_lite/swe_bench_lite.json'
    output_file_path = 'data/llm_baseline/baseline_gpt-4o_limited_output.jsonl'

    predictions = filter_compilable(dataset_file_path, output_file_path)

    random.seed(42)
    sampled_mutants = random.sample(predictions, 150) 

    for i in sampled_mutants:
        print(i.get('full_function', ''))
        print("="*60)

    with open('data/human_study/baseline_sampled_mutants.json', 'w') as f:
        json.dump(sampled_mutants, f, indent=2)