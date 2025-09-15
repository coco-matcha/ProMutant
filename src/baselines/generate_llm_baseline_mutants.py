#!/usr/bin/env python3
"""
LLM Baseline Mutant Generator

This script generates code mutants using LLM from pre-extracted buggy code contexts,
creates predictions in SWE-Bench format, and splits them for evaluation.

Usage:
    python generate_llm_baseline_mutants.py [options]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import openai


class MutantGenerator:
    """Generates code mutants using LLM based on buggy code contexts."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 3):
        self.llm = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
    
    def generate_mutants_from_contexts(self, contexts: List[Dict], output_file: str) -> Dict:
        """
        Generate mutants from all contexts and save to file.
        
        Args:
            contexts: List of buggy code contexts
            output_file: Path to output JSONL file
            
        Returns:
            Summary dictionary with results
        """
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        all_mutants = []
        processed_count = 0
        failed_count = 0
        
        print(f"Processing {len(contexts)} contexts...")
        
        for i, context in enumerate(contexts, 1):
            instance_id = context['instance_id']
            print(f"Processing context {i}/{len(contexts)}: {instance_id}")
            
            try:
                mutants = self._generate_mutants_for_instance(context)
                
                if mutants:
                    all_mutants.extend(mutants)
                    processed_count += 1
                    print(f"  Generated {len(mutants)} mutants")
                else:
                    failed_count += 1
                    print(f"  Failed to generate mutants")
                    
            except Exception as e:
                failed_count += 1
                print(f"  Error processing {instance_id}: {e}")
                # Add error entry
                all_mutants.append({
                    'instance_id': instance_id,
                    'error': f"Processing failed: {str(e)}"
                })
        
        # Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for mutant in all_mutants:
                    f.write(json.dumps(mutant) + '\n')
            
            print(f"\nSaved {len(all_mutants)} total mutants to: {output_file}")
            
            return {
                'success': True,
                'total_contexts': len(contexts),
                'processed_contexts': processed_count,
                'failed_contexts': failed_count,
                'total_mutants': len(all_mutants),
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"Failed to save output file: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_mutants': len(all_mutants)
            }
    
    def _generate_mutants_for_instance(self, context: Dict) -> List[Dict]:
        """Generate mutants for a single instance."""
        instance_id = context['instance_id']
        file_path = context['file_path']
        fixed_code = context['fixed_code']
        full_function = context['patched_function']
        
        # Determine number of mutants based on code complexity
        num_mutants = min(2, max(1, len(fixed_code.split('\n'))))
        
        # Create prompt
        prompt = self._create_mutant_prompt(
            full_function=full_function,
            target_code=fixed_code,
            num_mutants=num_mutants,
            file_path=file_path
        )
        
        # Generate mutants with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in code mutation for software testing."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                generated_text = response.choices[0].message.content
                mutants = self._parse_mutants_response(generated_text, instance_id, prompt)
                
                if mutants:
                    return mutants
                    
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
        
        # Return error entry if all attempts failed
        return [{
            'instance_id': instance_id,
            'error': f"Failed to generate mutants after {self.max_retries} attempts"
        }]
    
    def _create_mutant_prompt(self, full_function: str, target_code: str, 
                             num_mutants: int, file_path: str) -> str:
        """Create the LLM prompt for mutant generation."""
        return f"""
{full_function}

Above is the original code. Your task is to generate {num_mutants} mutants in the original code (notice: mutant refers to mutant in software engineering, i.e. making subtle alterations to the original code).

Focus particularly on this line:
{target_code}

Here are some examples of mutants which you can refer to:
{{
    "precode": "n = (n & (n - 1));",
    "aftercode": "n = (n ^ (n - 1));"
}},
{{
    "precode": "while (!queue.isEmpty()) {{",
    "aftercode": "while (true) {{"
}},
{{
    "precode": "return depth==0;",
    "aftercode": "return true;"
}},
{{
    "precode": "c = bin_op.apply(b,a);",
    "aftercode": "c = bin_op.apply(a,b);"
}},
{{
    "precode": "while (Math.abs(x-approx*approx) > epsilon) {{",
    "aftercode": "while (Math.abs(x-approx) > epsilon) {{"
}}

Requirements:
1. Provide generated mutants directly
2. A mutation can only occur on one line
3. Your output must be like:
[
    {{
        "id": 1,
        "precode": "original line of code",
        "aftercode": "mutated line of code"
    }}
]

Where:
- "id": mutant serial number
- "precode": line of code before mutation (cannot be empty)
- "aftercode": line of code after mutation

4. Prohibit generating the exact same mutants
5. Output as valid JSON array
"""
    
    def _parse_mutants_response(self, response: str, instance_id: str, prompt: str) -> List[Dict]:
        """Parse LLM response to extract mutants."""
        # Extract JSON from response
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        if not json_matches:
            return []
        
        try:
            parsed_mutants = json.loads(json_matches[0])
            
            mutant_entries = []
            for mutant in parsed_mutants:
                if 'precode' in mutant and 'aftercode' in mutant:
                    mutant_entries.append({
                        "instance_id": instance_id,
                        "original_code": mutant['precode'],
                        "generated_code": mutant['aftercode'],
                        "prompt": prompt
                    })
            
            return mutant_entries
            
        except json.JSONDecodeError as e:
            print(f"  JSON parsing failed: {e}")
            return []


class PredictionCreator:
    """Creates predictions in SWE-Bench format from mutant data."""
    
    def create_predictions(self, mutants_file: str, dataset_file: str, 
                          model_name: str = "gpt-4o") -> List[Dict]:
        """
        Create predictions in SWE-Bench format.
        
        Args:
            mutants_file: Path to mutants JSONL file
            dataset_file: Path to original SWE-Bench dataset
            model_name: Model name for predictions
            
        Returns:
            List of predictions
        """
        # Load data
        try:
            with open(dataset_file, 'r') as f:
                original_data = json.load(f)
            
            with open(mutants_file, 'r') as f:
                mutants = [json.loads(line) for line in f]
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading data: {e}")
            return []
        
        predictions = []
        error_count = 0
        
        for mutant in mutants:
            instance_id = mutant['instance_id']
            
            # Skip error entries
            if 'error' in mutant:
                error_count += 1
                print(f"Skipping error mutant: {instance_id}")
                continue
            
            # Validate mutant data
            if not self._validate_mutant(mutant):
                error_count += 1
                continue
            
            # Find original item
            original_items = [item for item in original_data 
                            if item.get('instance_id') == instance_id]
            
            if len(original_items) != 1:
                error_count += 1
                print(f"Expected 1 original item for {instance_id}, found {len(original_items)}")
                continue
            
            original_item = original_items[0]
            
            # Create patch
            try:
                updated_patch = self._create_updated_patch(mutant, original_item)
                if updated_patch:
                    predictions.append({
                        "instance_id": instance_id,
                        "model_name_or_path": model_name,
                        "model_patch": updated_patch
                    })
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"Error creating patch for {instance_id}: {e}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Total mutants: {len(mutants)}")
        print(f"  Error mutants: {error_count}")
        print(f"  Valid predictions: {len(predictions)}")
        
        return predictions
    
    def _validate_mutant(self, mutant: Dict) -> bool:
        """Validate mutant data."""
        instance_id = mutant['instance_id']
        
        # Check for required fields
        if not mutant.get('generated_code', '').strip():
            print(f"No generated code for {instance_id}")
            return False
        
        if not mutant.get('original_code', '').strip():
            print(f"No original code for {instance_id}")
            return False
        
        # Check if codes are identical
        if mutant['generated_code'].strip() == mutant['original_code'].strip():
            print(f"Generated code same as original for {instance_id}")
            return False
        
        # Check for multi-line violations
        if '\n' in mutant['generated_code']:
            print(f"Multi-line generated code for {instance_id}")
            return False
        
        return True
    
    def _create_updated_patch(self, mutant: Dict, original_item: Dict) -> Optional[str]:
        """Create updated patch by replacing original code with generated code."""
        patch_lines = original_item.get('patch', '').split('\n')
        original_code = mutant['original_code']
        
        # Find the fixed code line in the patch
        fixed_code_line = None
        for line in patch_lines:
            if line.startswith('+') and original_code in line:
                fixed_code_line = line
                break
        
        if not fixed_code_line:
            print(f"Cannot find original code '{original_code}' in patch for {mutant['instance_id']}")
            return None
        
        # Replace original code with generated mutant
        patch_content = original_item.get('patch', '')
        updated_patch = patch_content.replace(original_code, mutant['generated_code'])
        
        return updated_patch


class FileSplitter:
    """Splits JSONL files into multiple variants for evaluation."""
    
    @staticmethod
    def split_predictions(input_file: str, output_dir: str = "split_predictions", 
                         output_prefix: str = "predictions") -> None:
        """
        Split JSONL file into multiple files with one patch per instance.
        
        Args:
            input_file: Input JSONL file path
            output_dir: Directory to save split files
            output_prefix: Prefix for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Group patches by instance_id
        instance_patches = defaultdict(list)
        
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    pred = json.loads(line.strip())
                    instance_id = pred['instance_id']
                    instance_patches[instance_id].append(pred)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading input file: {e}")
            return
        
        if not instance_patches:
            print("No patches found in input file")
            return
        
        # Find max number of variants
        max_variants = max(len(patches) for patches in instance_patches.values())
        
        # Create split files
        for variant_idx in range(max_variants):
            output_file = os.path.join(output_dir, f"{output_prefix}_{variant_idx + 1}.jsonl")
            
            # Collect predictions for this variant
            predictions = []
            for instance_id, patches in instance_patches.items():
                if variant_idx < len(patches):
                    predictions.append(patches[variant_idx])
            
            # Write file
            try:
                with open(output_file, 'w') as f:
                    for pred in predictions:
                        f.write(json.dumps(pred) + '\n')
                print(f"Created {output_file} with {len(predictions)} predictions")
            except IOError as e:
                print(f"Error creating {output_file}: {e}")


class LLMBaselinePipeline:
    """Main pipeline for LLM baseline mutant generation."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.mutant_generator = MutantGenerator(
            api_key=config['api_key'],
            model=config['model'],
            max_retries=config['max_retries']
        )
        self.prediction_creator = PredictionCreator()
        self.file_splitter = FileSplitter()
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        print("Starting LLM Baseline Pipeline")
        print(f"Configuration: {self.config['experiment_name']}")
        
        try:
            # Load contexts
            contexts = self._load_contexts()
            if not contexts:
                return False
            
            # Generate mutants
            if not self._generate_mutants(contexts):
                return False
            
            # Create predictions
            if not self._create_predictions():
                return False
            
            # Split predictions
            self._split_predictions()
            
            print("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return False
    
    def _load_contexts(self) -> List[Dict]:
        """Load buggy code contexts."""
        try:
            with open(self.config['context_file'], 'r', encoding='utf-8') as f:
                contexts = json.load(f)
            
            if self.config.get('limit_contexts'):
                contexts = contexts[:self.config['limit_contexts']]
                print(f"Limited to {len(contexts)} contexts")
            
            return contexts
            
        except Exception as e:
            print(f"Error loading contexts: {e}")
            return []
    
    def _generate_mutants(self, contexts: List[Dict]) -> bool:
        """Generate mutants from contexts."""
        print("\nGenerating mutants...")
        result = self.mutant_generator.generate_mutants_from_contexts(
            contexts, self.config['mutants_file']
        )
        return result.get('success', False)
    
    def _create_predictions(self) -> bool:
        """Create predictions from mutants."""
        print("\nCreating predictions...")
        predictions = self.prediction_creator.create_predictions(
            self.config['mutants_file'],
            self.config['dataset_file'],
            self.config['model']
        )
        
        if not predictions:
            print("No predictions created")
            return False
        
        # Save predictions
        try:
            with open(self.config['predictions_file'], 'w') as f:
                for prediction in predictions:
                    json.dump(prediction, f)
                    f.write('\n')
            
            print(f"Saved {len(predictions)} predictions to {self.config['predictions_file']}")
            return True
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return False
    
    def _split_predictions(self) -> None:
        """Split predictions into variant files."""
        print("\nSplitting predictions...")
        self.file_splitter.split_predictions(
            self.config['predictions_file'],
            self.config['split_output_dir'],
            self.config['split_output_prefix']
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LLM baseline mutants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--model', default='gpt-4o', help='OpenAI model to use')
    parser.add_argument('--experiment-name', default='llm_baseline', help='Experiment name')
    parser.add_argument('--context-file', default='data/swe_bench_lite/extracted_code_contexts.json',
                       help='Path to buggy code contexts JSON file')
    parser.add_argument('--dataset-file', default='data/swe_bench_lite/swe_bench_lite.json',
                       help='Path to SWE-Bench dataset file')
    parser.add_argument('--output-dir', default='llm_baseline', help='Output directory')
    parser.add_argument('--max-retries', type=int, default=3, help='Max API retry attempts')
    parser.add_argument('--limit-contexts', type=int, help='Limit number of contexts to process')
    
    return parser.parse_args()


def create_config(args) -> Dict:
    """Create configuration from arguments."""
    output_dir = Path(args.output_dir)
    base_name = f"llm_{args.model.replace('-', '_')}"
    
    return {
        'api_key': args.api_key,
        'model': args.model,
        'experiment_name': args.experiment_name,
        'context_file': args.context_file,
        'dataset_file': args.dataset_file,
        'max_retries': args.max_retries,
        'limit_contexts': args.limit_contexts,
        
        'mutants_file': str(output_dir / f"{base_name}_output.jsonl"),
        'predictions_file': str(output_dir / f"{base_name}_predictions.jsonl"),
        'split_output_dir': str(output_dir / "split_predictions"),
        'split_output_prefix': f"llm_{args.model.replace('-', '_')}_predictions"
    }


def main():
    """Main entry point."""
    args = parse_arguments()
    config = create_config(args)
    
    pipeline = LLMBaselinePipeline(config)
    
    if not pipeline.run_pipeline():
        sys.exit(1)


if __name__ == "__main__":
    main()