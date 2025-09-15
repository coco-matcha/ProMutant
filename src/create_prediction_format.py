import json
import os
import subprocess
import argparse
import tempfile
from collections import defaultdict
from typing import List, Dict, Optional


class PatchProcessor:
    """Handles patch processing and formatting for SWE-Bench predictions."""
    
    @staticmethod
    def get_indentation(code: str) -> str:
        """Extract indentation from the first line of code."""
        if not code:
            return ""
        
        first_line = code.split('\n')[0]
        indentation = first_line[:len(first_line) - len(first_line.lstrip())]
        return indentation.strip('\n')
    
    @staticmethod
    def format_diff_lines(code: str, prefix: str, indentation: str = "") -> str:
        """Format code lines with diff prefixes (+/-)."""
        if not code:
            return prefix
        
        # Apply indentation if code doesn't already have it
        if code and len(code) - len(code.lstrip()) == 0 and indentation:
            code = indentation + code
        
        # Add prefix to each line
        lines = code.split('\n')
        formatted_lines = [prefix + line for line in lines]
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def rediff_patch(patch_content: str) -> Optional[str]:
        """
        Fix patch content using rediff command.
        
        Args:
            patch_content: String containing the patch content
            
        Returns:
            Fixed patch content, or None if failed
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as temp_file:
                temp_file.write(patch_content)
                temp_path = temp_file.name
            
            try:
                # Run rediff on the temporary file
                result = subprocess.run(
                    ['rediff', temp_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return result.stdout
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
        
        except subprocess.CalledProcessError as e:
            print(f"rediff failed: {e}")
            print(f"stderr: {e.stderr}")
            return None
        except FileNotFoundError:
            print("rediff command not found. Please install patchutils.")
            return None
    
    def replace_in_git_patch(self, patch_content: str, fixed_code: str, 
                           buggy_code: str, replacement_code: str) -> str:
        """
        Replace buggy code in git patch format, handling diff markers properly.
        
        Args:
            patch_content: Original patch content
            fixed_code: Original fixed code
            buggy_code: Original buggy code  
            replacement_code: New code to replace with
            
        Returns:
            Updated patch content
        """
        if not patch_content:
            return patch_content

        # Get indentation level
        indentation = ""
        if fixed_code:
            indentation = self.get_indentation(fixed_code)
        elif buggy_code:
            indentation = self.get_indentation(buggy_code)

        # Format codes with diff prefixes
        formatted_buggy = self.format_diff_lines(buggy_code, '-')
        formatted_fixed = self.format_diff_lines(fixed_code, '+')
        formatted_replacement = self.format_diff_lines(replacement_code, '+', indentation)
        
        # Debug output
        if formatted_fixed == "+":
            print("Original Buggy Code:\n", formatted_buggy)
        else:
            print("Original Fixed Code:\n", formatted_fixed)
        print("Replacement Code:\n", formatted_replacement)

        # Replace in patch content
        if formatted_fixed == "+":
            # Replace first occurrence of buggy code, add replacement after
            patch_content = patch_content.replace(
                formatted_buggy, 
                formatted_buggy + '\n' + formatted_replacement, 
                1
            )
        else:
            # Replace first occurrence of fixed code with replacement
            patch_content = patch_content.replace(formatted_fixed, formatted_replacement, 1)

        # Fix inconsistent tab/space usage
        patch_content = patch_content.expandtabs(4)

        # Use rediff for other inconsistencies
        rediffed_patch = self.rediff_patch(patch_content)
        return rediffed_patch if rediffed_patch else patch_content


class PredictionGenerator:
    """Generates predictions in SWE-Bench format from back-translation results."""
    
    def __init__(self, dataset_file: str = "swe_bench_lite/swe_bench_lite.json"):
        self.dataset_file = dataset_file
        self.patch_processor = PatchProcessor()
        self.dataset_data = self._load_dataset()
    
    def _load_dataset(self) -> List[Dict]:
        """Load the SWE-Bench dataset."""
        try:
            with open(self.dataset_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Dataset file not found: {self.dataset_file}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in dataset file: {self.dataset_file}")
            return []
    
    def _get_original_patch(self, instance_id: str) -> Optional[str]:
        """Get the original patch for a given instance ID."""
        matching_items = [x for x in self.dataset_data if x['instance_id'] == instance_id]
        return matching_items[0].get('patch') if matching_items else None
    
    def _is_valid_result(self, result: Dict) -> bool:
        """Check if a result is valid for processing."""
        if 'error' in result:
            return False
        
        if not result.get('generated_code', '').strip():
            return False
        
        generated_code = result['generated_code'].strip()
        buggy_code = result.get('buggy_code', '').strip()
        fixed_code = result.get('fixed_code', '').strip()
        
        # Skip if generated code is same as original codes
        if generated_code == buggy_code:
            print(f"Generated code for {result['instance_id']} same as buggy code, skipping")
            return False
        
        if generated_code == fixed_code:
            print(f"Generated code for {result['instance_id']} same as fixed code, skipping")
            return False
        
        return True
    
    def create_predictions(self, results: List[Dict], model_name: str = "gpt-4o") -> List[Dict]:
        """
        Create predictions in SWE-Bench format.
        
        Args:
            results: Results from back-translation process
            model_name: Name of the model used
            
        Returns:
            List of predictions in required format
        """
        predictions = []
        
        for result in results:
            if not self._is_valid_result(result):
                continue
            
            instance_id = result['instance_id']
            original_patch = self._get_original_patch(instance_id)
            
            if not original_patch:
                print(f"No original patch found for {instance_id}, skipping")
                continue
            
            # Create updated patch
            updated_patch = self.patch_processor.replace_in_git_patch(
                original_patch,
                result['fixed_code'],
                result['buggy_code'],
                result['generated_code']
            )
            
            # Create prediction object
            prediction = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": updated_patch
            }
            
            predictions.append(prediction)
        
        return predictions


class FileManager:
    """Handles file I/O operations for predictions."""
    
    @staticmethod
    def save_predictions(predictions: List[Dict], output_file: str) -> None:
        """Save predictions to JSONL file."""
        try:
            with open(output_file, 'a') as f:
                for prediction in predictions:
                    json.dump(prediction, f)
                    f.write('\n')
            print(f"Saved {len(predictions)} predictions to {output_file}")
        except IOError as e:
            print(f"Error saving predictions: {e}")
    
    @staticmethod
    def load_results(input_file: str) -> List[Dict]:
        """Load results from JSONL file."""
        try:
            with open(input_file, 'r') as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            print(f"Input file not found: {input_file}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON in input file: {input_file}")
            return []
    
    @staticmethod
    def split_predictions_by_variants(input_file: str, output_dir: str = "split_predictions", 
                                    output_prefix: str = "predictions") -> None:
        """
        Split JSONL file into multiple files where each file has one patch per instance.
        
        Args:
            input_file: Input JSONL file path
            output_dir: Directory to save split files
            output_prefix: Prefix for output files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Group patches by instance_id
        instance_patches = defaultdict(list)
        
        # Read and group by instance_id
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    pred = json.loads(line.strip())
                    instance_id = pred['instance_id']
                    instance_patches[instance_id].append(pred)
        except FileNotFoundError:
            print(f"Input file not found: {input_file}")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in input file: {input_file}")
            return
        
        # Find max number of variants
        if not instance_patches:
            print("No patches found in input file")
            return
        
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


def process_back_translation_output(raw_output_file: str, dataset_file: str = "swe_bench_lite/swe_bench_lite.json", 
                                  model_name: str = "gpt-4o") -> List[Dict]:
    """
    Process back-translation output and create predictions.
    
    Args:
        raw_output_file: Path to raw output file
        dataset_file: Path to SWE-Bench dataset file
        model_name: Name of the model used
        
    Returns:
        List of predictions
    """
    # Load results
    results = FileManager.load_results(raw_output_file)
    if not results:
        return []
    
    # Generate predictions
    generator = PredictionGenerator(dataset_file)
    predictions = generator.create_predictions(results, model_name)
    
    return predictions

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create SWE-Bench predictions from mutant output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mutants-file', required=True, 
                       help='Path to mutants JSONL file')
    parser.add_argument('--dataset-file', default='data/swe_bench_lite/swe_bench_lite.json',
                       help='Path to SWE-Bench dataset file')
    parser.add_argument('--output-dir', default='data/promutant',
                       help='Output directory for predictions files')
    parser.add_argument('--model-name', default='gpt-4o',
                       help='Model name for predictions')
    parser.add_argument('--split-prefix', 
                       help='Prefix for split files (default: derived from model)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    split_dir = f'{args.output_dir}/split_predictions'
    os.makedirs(split_dir, exist_ok=True)
    
    # Set file path, default split prefix if not provided
    predictions_file = f'{args.output_dir}/predictions.jsonl'
    split_prefix = args.split_prefix or f"{args.model_name}_predictions"
    
    # Process output file
    print(f"Processing {args.mutants_file}...")
    predictions = process_back_translation_output(
        raw_output_file=args.mutants_file,
        dataset_file=args.dataset_file,
        model_name=args.model_name
    )
    
    if not predictions:
        print("No valid predictions generated")
        return
    
    # Save predictions
    FileManager.save_predictions(predictions, predictions_file)
    print(f"Processed {len(predictions)} items")
    
    # Split into multiple files for evaluation
    print(f"Splitting predictions into variants...")
    FileManager.split_predictions_by_variants(
        input_file=predictions_file,
        output_dir=split_dir,
        output_prefix=split_prefix
    )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()