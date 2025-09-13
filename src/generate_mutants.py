import openai
import re
import json
from typing import List, Dict, Optional


class BackTranslationGenerator:
    """Generates code mutations using back-translation between problem statements and code."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.llm = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract code from LLM response, handling markdown code blocks."""
        pattern = r'```(?:\w+)?\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1).strip() if match else response.strip()
    
    def extract_statement_from_response(self, response: str) -> str:
        """Extract the first sentence from LLM response as problem statement."""
        lines = response.strip().split('\n')
        first_line = lines[0] if lines else ""
        # Get first sentence
        sentences = first_line.split('. ')
        return sentences[0].strip() if sentences else first_line.strip()
    
    def create_patch(self, context_before: str, buggy_code: str, fixed_code: str, context_after: str) -> str:
        """Create a git-style patch from code changes."""
        buggy_lines = [f"-{line}" for line in buggy_code.split('\n')]
        fixed_lines = [f"+{line}" for line in fixed_code.split('\n')]
        
        return '\n'.join([
            context_before,
            '\n'.join(buggy_lines),
            '\n'.join(fixed_lines),
            context_after
        ])
    
    def generate_buggy_code(self, fixed_code: str, context_before: str, context_after: str, 
                           problem_statement: str) -> str:
        """Generate buggy code from fixed code and problem description."""
        system_prompt = "You are an expert software engineer. Your task is to reverse-engineer buggy code from a fixed version and problem description."
        user_prompt = f"""**Fixed Code (Reference):**
```python
{fixed_code}
```
**Context Before**
{context_before}

**Context After**
{context_after}

**Problem Statement:**
{problem_statement}

**Your Task:**
Based on the problem statement, reconstruct what the ORIGINAL BUGGY CODE likely looked like before it was fixed. The fixed code above shows the correct implementation - you need to introduce the specific bug described in the problem statement.

**Constraints:**
- ONLY modify the code within the fixed code section to introduce the bug
- The context is provided for understanding only - DO NOT modify it  
- Return ONLY the buggy version of the code section
- Do not include any context lines in your response
- Your output should be the original buggy code that would need the fix described

**Output Format:**
Return only the buggy Python code without any explanations, comments, or markdown formatting."""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def generate_problem_statement(self, patch: str) -> str:
        """Generate problem statement from code patch."""
        system_prompt = "You are an expert software engineer. Your task is to analyze a code patch and generate a clear technical bug description about the bug that was fixed."
        user_prompt = f"""**Your Task:** 
Analyze the differences between the buggy and fixed code in the patch to understand what issue was resolved. Write a clear, concise problem statement that describes the bug and what needs to be fixed.
**Patch**
{patch}
**Output Format:**
Return only the bug description without any explanations, comments, or markdown formatting."""
        
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def run_back_translation(self, code_context: Dict, num_loops: int = 3, 
                           max_retries: int = 3) -> List[Dict]:
        """
        Run the back-translation process to generate code mutations.
        
        Args:
            code_context: Dictionary containing code context information
            num_loops: Number of mutation loops to run
            max_retries: Maximum retry attempts for generating unique mutants
            
        Returns:
            List of generated mutations with metadata
        """
        results = []
        
        # Extract context information
        instance_id = code_context.get('instance_id')
        original_statement = code_context.get('problem_statement', '').split('\n')[0]
        instance_buggy = code_context.get('buggy_code', '')
        instance_fixed = code_context.get('fixed_code', '')
        context_before = '\n'.join(code_context.get('context_before', []))
        context_after = '\n'.join(code_context.get('context_after', []))
        
        # Track generated content to avoid duplicates
        generated_statements = {self._normalize_text(original_statement)}
        generated_codes = {
            self._normalize_text(instance_fixed),
            self._normalize_text(instance_buggy)
        }
        
        # Initialize with original values
        current_code = instance_fixed
        current_statement = original_statement
        
        print(f"Starting Back Translation for {num_loops} loops...")
        print(f"Instance ID: {instance_id}")
        print(f"Original statement: {original_statement}")
        print("-" * 80)
        
        for loop in range(num_loops):
            print(f"\n=== Loop {loop + 1} ===")
            
            # Handle empty fixed code case for first loop
            if loop == 0 and not instance_fixed.strip():
                current_patch = self.create_patch(context_before, instance_buggy, 
                                                instance_fixed, context_after)
                current_statement = self._generate_unique_statement(
                    current_patch, original_statement, generated_statements, max_retries
                )
            
            # Generate mutant code from problem statement
            print("Generating mutant code...")
            current_code = self._generate_unique_code(
                instance_fixed, context_before, context_after, current_statement,
                generated_codes, max_retries
            )
            
            if current_code:
                results.append({
                    'generated_code': current_code,
                    'instance_id': instance_id,
                    'buggy_code': instance_buggy,
                    'fixed_code': instance_fixed,
                    'problem_statement': current_statement
                })
                print(f"Generated code: {current_code}")
            
            # Generate problem statement for next loop (skip on last iteration)
            if loop < num_loops - 1:
                print("Generating mutant problem statement...")
                current_patch = self.create_patch(context_before, current_code, 
                                                instance_fixed, context_after)
                current_statement = self._generate_unique_statement(
                    current_patch, original_statement, generated_statements, max_retries
                )
        
        print(f"\nBack Translation completed: {len(results)} mutations generated")
        return results
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for duplicate detection."""
        return text.strip().replace(' ', '') if text else ''
    
    def _generate_unique_code(self, fixed_code: str, context_before: str, context_after: str,
                             statement: str, generated_codes: set, max_retries: int) -> Optional[str]:
        """Generate unique code with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.generate_buggy_code(fixed_code, context_before, 
                                                  context_after, statement)
                code = self.extract_code_from_response(response)
                normalized_code = self._normalize_text(code)
                
                if code and normalized_code not in generated_codes:
                    generated_codes.add(normalized_code)
                    return code
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        print(f"Failed to generate unique code after {max_retries} attempts")
        return None
    
    def _generate_unique_statement(self, patch: str, original_statement: str,
                                  generated_statements: set, max_retries: int) -> str:
        """Generate unique problem statement with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.generate_problem_statement(patch)
                statement = self.extract_statement_from_response(response)
                normalized_statement = self._normalize_text(statement)
                
                if len(statement) > 1 and normalized_statement not in generated_statements:
                    generated_statements.add(normalized_statement)
                    print(f"Generated statement: {statement}")
                    return statement
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        print(f"Failed to generate unique statement after {max_retries} attempts")
        return original_statement


def main():
    """Main execution function."""
    # Configuration
    API_KEY = "your-api-key-here"  # Replace with actual API key
    MODEL = "gpt-4o"
    CONTEXT_FILE = 'data/swe_bench_lite/extracted_code_contexts.json'
    OUTPUT_FILE = f'data/promutant/promutant_{MODEL}_output.jsonl'

    # Load and prepare data
    try:
        with open(CONTEXT_FILE, 'r') as f:
            context_list = json.load(f)
    except FileNotFoundError:
        print(f"Context file not found: {CONTEXT_FILE}")
        return
    
    # Initialize generator
    generator = BackTranslationGenerator(API_KEY, MODEL)
    
    # Process each context item
    for i, code_context_item in enumerate(context_list, 1):
        print(f"\nProcessing item {i}/{len(context_list)}")
        
        try:
            results = generator.run_back_translation(
                code_context_item, 
                max_retries=3, 
                num_loops=2
            )
            
            # Append results to output file
            with open(OUTPUT_FILE, 'a') as f:
                for result in results:
                    json.dump(result, f)
                    f.write('\n')
                    
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    
    print(f"\nProcessing complete. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()