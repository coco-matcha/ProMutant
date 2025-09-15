#!/usr/bin/env python3
"""
Mutmut-Based Mutant Generator

Uses Mutmut's actual mutation operators to generate mutants without requiring test execution.
Adapted from Mutmut's file_mutation.py and node_mutation.py.
"""

import json
import argparse
import sys
import os
import re
from typing import Any, Union, cast, Set, List, Dict
from collections.abc import Callable, Iterable, Sequence
from collections import defaultdict
from dataclasses import dataclass

import libcst as cst
import libcst.matchers as m
from libcst.metadata import PositionProvider

# Mutation operators from Mutmut
OPERATORS_TYPE = Sequence[
    tuple[
        type[cst.CSTNode],
        Callable[[Any], Iterable[cst.CSTNode]],
    ]
]

# Pattern to match (nearly) all chars in a string that are not part of an escape sequence
NON_ESCAPE_SEQUENCE = re.compile(r"((?<!\\)[^\\]+)")

@dataclass
class Mutation:
    original_node: cst.CSTNode
    mutated_node: cst.CSTNode
    line_number: int

# === MUTMUT'S MUTATION OPERATORS ===

def operator_number(node: cst.BaseNumber) -> Iterable[cst.BaseNumber]:
    """Mutate numbers: increment by 1"""
    if isinstance(node, (cst.Integer, cst.Float)):
        yield node.with_changes(value=repr(node.evaluated_value + 1))
    elif isinstance(node, cst.Imaginary):
        yield node.with_changes(value=repr(node.evaluated_value + 1j))

def operator_string(node: cst.BaseString) -> Iterable[cst.BaseString]:
    """Mutate strings: add XX prefix/suffix, change case"""
    if isinstance(node, cst.SimpleString):
        value = node.value
        prefix = value[
            : min([x for x in [value.find('"'), value.find("'")] if x != -1])
        ]
        value = value[len(prefix) :]

        if value.startswith('"""') or value.startswith("'''"):
            # Skip triple-quoted strings (docs)
            return

        supported_str_mutations = [
            lambda x: "XX" + x + "XX",
            lambda x: NON_ESCAPE_SEQUENCE.sub(lambda match: match.group(1).lower(), x),
            lambda x: NON_ESCAPE_SEQUENCE.sub(lambda match: match.group(1).upper(), x),
        ]

        for mut_func in supported_str_mutations:
            new_value = f"{prefix}{value[0]}{mut_func(value[1:-1])}{value[-1]}"
            if new_value != value:
                yield node.with_changes(value=new_value)

def operator_lambda(node: cst.Lambda) -> Iterable[cst.Lambda]:
    """Mutate lambda expressions"""
    if m.matches(node, m.Lambda(body=m.Name("None"))):
        yield node.with_changes(body=cst.Integer("0"))
    else:
        yield node.with_changes(body=cst.Name("None"))

def operator_remove_unary_ops(node: cst.UnaryOperation) -> Iterable[cst.BaseExpression]:
    """Remove unary operators like 'not' and '~'"""
    if isinstance(node.operator, (cst.Not, cst.BitInvert)):
        yield node.expression

def operator_name(node: cst.Name) -> Iterable[cst.CSTNode]:
    """Mutate name bindings: True<->False, deepcopy<->copy"""
    name_mappings = {
        "True": "False",
        "False": "True",
        "deepcopy": "copy",
    }
    if node.value in name_mappings:
        yield node.with_changes(value=name_mappings[node.value])

def operator_assignment(node: Union[cst.Assign, cst.AnnAssign]) -> Iterable[cst.CSTNode]:
    """Mutate assignments: a = b -> a = None, a = None -> a = ''"""
    if not node.value:
        return
    if m.matches(node.value, m.Name("None")):
        mutated_value = cst.SimpleString('""')
    else:
        mutated_value = cst.Name("None")
    yield node.with_changes(value=mutated_value)

def operator_augmented_assignment(node: cst.AugAssign) -> Iterable[cst.Assign]:
    """Mutate augmented assignments: += -> ="""
    yield cst.Assign([cst.AssignTarget(node.target)], node.value, node.semicolon)

def operator_dict_arguments(node: cst.Call) -> Iterable[cst.Call]:
    """mutate dict(a=b, c=d) to dict(aXX=b, c=d) and dict(a=b, cXX=d)"""
    if not m.matches(node.func, m.Name(value="dict")):
        return

    for i, arg in enumerate(node.args):
        if not arg.keyword:
            return
        keyword = arg.keyword
        mutated_keyword = keyword.with_changes(value=keyword.value + "XX")
        mutated_args = [
            *node.args[:i],
            node.args[i].with_changes(keyword=mutated_keyword),
            *node.args[i+1:],
        ]
        yield node.with_changes(args=mutated_args)

def operator_arg_removal(node: cst.Call) -> Iterable[cst.Call]:
    """try to drop each arg in a function call, e.g. foo(a, b) -> foo(b), foo(a)"""
    for i, arg in enumerate(node.args):
        # replace with None
        if arg.star == '' and not m.matches(arg.value, m.Name("None")):
            mutated_arg = arg.with_changes(value=cst.Name("None"))
            yield node.with_changes(args=[*node.args[:i], mutated_arg, *node.args[i + 1 :]])

    if len(node.args) > 1:
        for i in range(len(node.args)):
            yield node.with_changes(args=[*node.args[:i], *node.args[i + 1 :]])

def operator_string_methods_swap(node: cst.Call) -> Iterable[cst.Call]:
    """try to swap string method to opposite e.g. a.lower() -> a.upper()"""
    supported_swaps = [
        ("lower", "upper"), ("upper", "lower"),
        ("lstrip", "rstrip"), ("rstrip", "lstrip"),
        ("find", "rfind"), ("rfind", "find"),
        ("ljust", "rjust"), ("rjust", "ljust"),
        ("index", "rindex"), ("rindex", "index"),
        ("removeprefix", "removesuffix"), ("removesuffix", "removeprefix"),
        ("partition", "rpartition"), ("rpartition", "partition"),
        ("split", "rsplit"), ("rsplit", "split")
    ]

    for old_call, new_call in supported_swaps:
        if m.matches(node.func, m.Attribute(value=m.DoNotCare(), attr=m.Name(value=old_call))):
            func_name = cast(cst.Attribute, node.func).attr
            yield node.with_deep_changes(func_name, value=new_call)

def operator_match(node: cst.Match) -> Iterable[cst.CSTNode]:
    """Drop the case statements in a match."""
    if len(node.cases) > 1:
        for i in range(len(node.cases)):
            yield node.with_changes(cases=[*node.cases[:i], *node.cases[i+1:]])

# Keyword mappings
_keyword_mapping: dict[type[cst.CSTNode], type[cst.CSTNode]] = {
    cst.Is: cst.IsNot,
    cst.IsNot: cst.Is,
    cst.In: cst.NotIn,
    cst.NotIn: cst.In,
    cst.Break: cst.Return,
    cst.Continue: cst.Break,
}

def operator_keywords(node: cst.CSTNode) -> Iterable[cst.CSTNode]:
    """Mutate keywords: is<->is not, in<->not in, break<->return"""
    yield from _simple_mutation_mapping(node, _keyword_mapping)

# Operator mappings
_operator_mapping: dict[type[cst.CSTNode], type[cst.CSTNode]] = {
    cst.Add: cst.Subtract,
    cst.Subtract: cst.Add,
    cst.Multiply: cst.Divide,
    cst.Divide: cst.Multiply,
    cst.FloorDivide: cst.Divide,
    cst.Modulo: cst.Divide,
    cst.LeftShift: cst.RightShift,
    cst.RightShift: cst.LeftShift,
    cst.BitAnd: cst.BitOr,
    cst.BitOr: cst.BitAnd,
    cst.BitXor: cst.BitAnd,
    cst.Power: cst.Multiply,
    cst.LessThan: cst.LessThanEqual,
    cst.LessThanEqual: cst.LessThan,
    cst.GreaterThan: cst.GreaterThanEqual,
    cst.GreaterThanEqual: cst.GreaterThan,
    cst.Equal: cst.NotEqual,
    cst.NotEqual: cst.Equal,
    cst.And: cst.Or,
    cst.Or: cst.And,
}

def operator_swap_op(node: cst.CSTNode) -> Iterable[cst.CSTNode]:
    """Swap operators: +<->-, *<->/,  ==<->!=, and<->or, etc."""
    if m.matches(node, m.BinaryOperation() | m.UnaryOperation() | m.BooleanOperation() | m.ComparisonTarget() | m.AugAssign()):
        typed_node = cast(Union[cst.BinaryOperation, cst.UnaryOperation, cst.BooleanOperation, cst.ComparisonTarget, cst.AugAssign], node)
        operator = typed_node.operator
        for new_operator in _simple_mutation_mapping(operator, _operator_mapping):
            yield node.with_changes(operator=new_operator)

def _simple_mutation_mapping(
    node: cst.CSTNode, mapping: dict[type[cst.CSTNode], type[cst.CSTNode]]
) -> Iterable[cst.CSTNode]:
    """Yield mutations from the node class mapping"""
    mutated_node_type = mapping.get(type(node))
    if mutated_node_type:
        yield mutated_node_type()

# All mutation operators from Mutmut
mutation_operators: OPERATORS_TYPE = [
    (cst.BaseNumber, operator_number),
    (cst.BaseString, operator_string),
    (cst.Name, operator_name),
    (cst.Assign, operator_assignment),
    (cst.AnnAssign, operator_assignment),
    (cst.AugAssign, operator_augmented_assignment),
    (cst.UnaryOperation, operator_remove_unary_ops),
    (cst.Lambda, operator_lambda),
    (cst.Call, operator_dict_arguments),
    (cst.Call, operator_arg_removal),
    (cst.Call, operator_string_methods_swap),
    (cst.Match, operator_match),
    # Be more specific instead of using generic CSTNode
    (cst.Is, operator_keywords),
    (cst.IsNot, operator_keywords),
    (cst.In, operator_keywords),
    (cst.NotIn, operator_keywords),
    (cst.Break, operator_keywords),
    (cst.Continue, operator_keywords),
    (cst.BinaryOperation, operator_swap_op),
    (cst.UnaryOperation, operator_swap_op),
    (cst.BooleanOperation, operator_swap_op),
    (cst.ComparisonTarget, operator_swap_op),
    (cst.AugAssign, operator_swap_op),
]

# === MUTATION VISITOR ===

class MutationVisitor(cst.CSTVisitor):
    """Create mutations for all nodes, optionally filtered by target lines"""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, operators: OPERATORS_TYPE, target_lines: Set[int] = None, debug: bool = False):
        self.mutations: List[Mutation] = []
        self._operators = operators
        self._target_lines = target_lines or set()
        self._debug = debug

    def on_visit(self, node):
        if self._should_mutate_node(node):
            self._create_mutations(node)
        return True

    def _create_mutations(self, node: cst.CSTNode):
        """Create all possible mutations for this node"""
        position = self.get_metadata(PositionProvider, node, None)
        line_number = position.start.line if position else 0
        
        if self._debug:
            print(f"    DEBUG: Checking node {type(node).__name__} at line {line_number}")
        
        mutations_created = 0
        for node_type, operator in self._operators:
            if isinstance(node, node_type):
                if self._debug:
                    print(f"    DEBUG: Node matches operator for {node_type.__name__}")
                try:
                    for mutated_node in operator(node):
                        # Special handling for assignment mutations - replace the value, not the whole assignment
                        if isinstance(node, (cst.Assign, cst.AnnAssign)) and node.value:
                            # For assignments, we want to replace just the value part
                            mutation = Mutation(
                                original_node=node.value,  # Replace the value, not the whole assignment
                                mutated_node=mutated_node.value if hasattr(mutated_node, 'value') else mutated_node,
                                line_number=line_number
                            )
                        else:
                            # For other mutations, replace the whole node
                            mutation = Mutation(
                                original_node=node,
                                mutated_node=mutated_node,
                                line_number=line_number
                            )
                        
                        self.mutations.append(mutation)
                        mutations_created += 1
                        if self._debug:
                            print(f"    DEBUG: Created mutation {mutations_created} for {node_type.__name__}")
                except Exception as e:
                    if self._debug:
                        print(f"    DEBUG: Error creating mutation for {node_type.__name__}: {e}")
        
        if self._debug and mutations_created == 0:
            print(f"    DEBUG: No mutations created for {type(node).__name__} at line {line_number}")

    def _should_mutate_node(self, node: cst.CSTNode) -> bool:
        """Check if this node should be mutated"""
        position = self.get_metadata(PositionProvider, node, None)
        if not position:
            if self._debug:
                print(f"    DEBUG: No position metadata for {type(node).__name__}")
            return True
        
        # If target lines specified, only mutate those lines
        if self._target_lines and position.start.line not in self._target_lines:
            if self._debug:
                print(f"    DEBUG: Skipping {type(node).__name__} at line {position.start.line} (not in target lines)")
            return False
        
        return True

# === MUTANT GENERATOR ===

class MutmutBasedGenerator:
    """Generate mutants using Mutmut's operators"""
    
    def generate_mutants(self, code: str, fixed_code: str, target_lines: Set[int] = None, debug: bool = False) -> List[str]:
        """Generate all mutants for the given code"""
        try:
            module = cst.parse_module(code)
        except Exception as e:
            print(f"    Parse error: {e}")
            return []
        
        if debug:
            print(f"    DEBUG: Successfully parsed module")
        
        # Collect mutations WITHOUT MetadataWrapper to preserve node identity
        mutants = []
        
        # For each mutation operator, find and apply mutations directly
        for node_type, operator_func in mutation_operators:
            try:
                # Use a simple visitor to collect nodes of this type
                collector = SimpleNodeCollector(node_type, target_lines, code, debug)
                module.visit(collector)
                
                if debug and collector.nodes:
                    print(f"    DEBUG: Found {len(collector.nodes)} {node_type.__name__} nodes")
                
                # For each collected node, generate mutations
                for original_node in collector.nodes:
                    try:
                        mutated_nodes = list(operator_func(original_node))
                        
                        for mutated_node in mutated_nodes:
                            # Handle assignments specially
                            if isinstance(original_node, (cst.Assign, cst.AnnAssign)):
                                if hasattr(original_node, 'value') and original_node.value:
                                    # Replace just the value part
                                    if hasattr(mutated_node, 'value'):
                                        mutant_module = module.deep_replace(original_node.value, mutated_node.value)
                                    else:
                                        continue
                                else:
                                    continue
                            else:
                                # Replace the whole node
                                mutant_module = module.deep_replace(original_node, mutated_node)
                            
                            mutant_code = mutant_module.code
                            if mutant_code != code:
                                # Extract only the mutated part corresponding to fixed_code
                                mutated_fragment = self._extract_mutated_fragment(
                                    mutant_code, target_lines, debug
                                )
                                if mutated_fragment != fixed_code:
                                    mutants.append(mutated_fragment)
                                    if debug:
                                        print(f"    DEBUG: Generated mutant fragment: {mutated_fragment[:50]}...")
                                elif debug:
                                    print(f"    DEBUG: Generated mutant for {node_type.__name__}")
                            else:
                                print(f"    DEBUG: Mutant identical to original")
                                
                    except Exception as e:
                        if debug:
                            print(f"    DEBUG: Error with {node_type.__name__} mutation: {e}")
                        continue
                        
            except Exception as e:
                if debug:
                    print(f"    DEBUG: Error processing {node_type.__name__}: {e}")
                continue
        
        # Remove duplicates
        unique_mutants = []
        seen = set()
        for mutant in mutants:
            if mutant not in seen:
                unique_mutants.append(mutant)
                seen.add(mutant)
        
        if debug:
            print(f"    DEBUG: {len(unique_mutants)} unique mutants after deduplication")
        
        return unique_mutants
    
    def _extract_mutated_fragment(self, mutated_code: str, target_lines: Set[int], debug: bool = False) -> str:
        """Extract only the lines that were actually mutated"""
        try:
            mutated_lines = mutated_code.split('\n')
            generated_code = [mutated_lines[x] for x in target_lines]
            return '\n'.join(generated_code)
                
        except Exception as e:
            if debug:
                print(f"    DEBUG: Error extracting fragment: {e}")
            return mutated_code  # Fallback to full code


class SimpleNodeCollector(cst.CSTVisitor):
    """Collect nodes without MetadataWrapper to preserve identity"""
    
    def __init__(self, target_node_type, target_lines: Set[int], original_code: str, debug: bool):
        self.target_node_type = target_node_type
        self.target_lines = target_lines or set()
        self.original_code = original_code
        self.debug = debug
        self.nodes = []
        self.current_line = 1
        
        # Pre-calculate line positions if we need to filter by lines
        if self.target_lines:
            self.code_lines = original_code.split('\n')
    
    def on_visit(self, node: cst.CSTNode) -> bool:
        # Update current line estimate based on newlines in the code
        if hasattr(node, 'leading_lines'):
            for line in getattr(node, 'leading_lines', []):
                if hasattr(line, 'newline'):
                    self.current_line += 1
        
        # Check if this is the type we're looking for
        if isinstance(node, self.target_node_type):
            # If we need to filter by lines, do rough line checking
            should_include = True
            if self.target_lines:
                # For assignment nodes, check if the line content matches our target
                if isinstance(node, (cst.Assign, cst.AnnAssign)):
                    should_include = self._is_target_assignment(node)
                else:
                    # For other nodes, include if we're in the right general area
                    should_include = True # We'll be less strict for now
            
            if should_include:
                self.nodes.append(node)
                if self.debug:
                    print(f"    DEBUG: Collected {self.target_node_type.__name__} node")
        
        return True
    
    def _is_target_assignment(self, node: cst.Assign) -> bool:
        """Check if this assignment matches our target lines"""
        if not self.target_lines:
            return True
            
        # This is a rough heuristic - check if any target line contains assignment-like content
        for line_num in self.target_lines:
            if line_num <= len(self.code_lines):
                line_content = self.code_lines[line_num - 1].strip()
                if '=' in line_content and not line_content.startswith('#'):
                    return True
        return False

# === TARGET LINE FINDER ===

def find_target_lines(patched_function: str, fixed_code: str, debug: bool = False) -> Set[int]:
    """Find line numbers in patched_function that correspond to fixed_code"""
    if not fixed_code.strip():
        return set()
    
    if fixed_code not in patched_function:
        return set()

    fixed_lines = [line.strip() for line in fixed_code.strip().split('\n')]
    patched_lines = [line.strip() for line in patched_function.split('\n')]
    
    if debug:
        print(f"    DEBUG: Fixed lines: {fixed_lines}")
        print(f"    DEBUG: Patched lines ({len(patched_lines)} total):")
        for i, line in enumerate(patched_lines):
            print(f"    DEBUG:   {i}: {line}")
    
    target_line_numbers = set()
    
    # Method 1: Direct substring matching
    for i, patched_line in enumerate(patched_lines):
        if fixed_lines[0] == patched_line and fixed_lines[-1] == patched_lines[i+len(fixed_lines)-1]:
            target_line_numbers.update(range(i,i+len(fixed_lines)))
            if debug:
                print(f"    DEBUG: Found match at line {i}: {repr(patched_line)}")

    if debug:
        print(f"    DEBUG: Target lines found: {sorted(target_line_numbers)}")
    
    return sorted(target_line_numbers)

# === MAIN PROCESSING ===

def process_code_section(instance_id: str, patched_function: str, fixed_code: str, debug: bool = False) -> List[Dict[str, str]]:
    """Process code section with Mutmut-based mutations"""
    print(f"Processing instance: {instance_id}")
    
    # Find target lines
    target_lines = find_target_lines(patched_function, fixed_code, debug)
    if target_lines:
        print(f"    Target lines: {sorted(target_lines)}")
    else:
        print(f"    No target lines found, mutating all lines")
    
    # Generate mutants
    generator = MutmutBasedGenerator()
    mutants = generator.generate_mutants(patched_function, fixed_code, target_lines, debug)
    
    # Format results
    results = []
    for i, mutant_code in enumerate(mutants):
        results.append({
            'instance_id': f"{instance_id}_mutant_{i+1}",
            'original_code': fixed_code,
            'generated_code': mutant_code
        })
    
    print(f"    Generated {len(results)} mutants")
    return results

def process_json_file(input_file: str, output_file: str):
    """Process JSON file with Mutmut-based mutant generation"""
    
    print(f"Reading from: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} code sections...")
    
    # Clear output file at start
    with open(output_file, 'w') as f:
        pass
    
    total_mutants = 0
    successful_instances = 0
    parse_error = 0
    
    for i, section in enumerate(data, 1):
        instance_id = section.get('instance_id', f'section_{i}')
        patched_function = section.get('patched_function', '')
        fixed_code = section.get('fixed_code', '')

        try:
            module = cst.parse_module(patched_function)
        except Exception as e:
            parse_error += 1
            continue
        
        if not patched_function:
            print(f"[{i}/{len(data)}] Skipping {instance_id} - no patched_function")
            continue
        
        if not fixed_code:
            print(f"[{i}/{len(data)}] Processing {instance_id} - no fixed_code")
            continue
        
        print(f"[{i}/{len(data)}] Processing: {instance_id}")
        
        try:
            results = process_code_section(instance_id, patched_function, fixed_code, debug=False)
            
            if results:
                # Save results immediately after each instance
                with open(output_file, 'a') as f:
                    for result in results:
                        json.dump(result, f)
                        f.write('\n')
                
                total_mutants += len(results)
                successful_instances += 1
                print(f"    Saved {len(results)} mutants to {output_file}")
            else:
                print(f"    No mutants generated for {instance_id}")
            
        except Exception as e:
            print(f"    Error processing {instance_id}: {e}")
            continue

    print(f"\nProcessing complete!")
    print(f"Summary:")
    print(f"  - Total instances processed: {len(data)}")
    print(f"  - Error parsing function: {parse_error}")
    print(f"  - Successful instances: {successful_instances}")
    print(f"  - Total mutants generated: {total_mutants}")
    print(f"  - Output saved to: {output_file}")


def create_prediction_format(dataset_file_path: str, output_file_path: str, model_name: str = "mutmut") -> List[Dict]:
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
        instance_id = mutant['instance_id'].split('_mutant_')[0]

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

        original_code = mutant['original_code'].replace('\n', '\n+')
        generated_code = mutant['generated_code'].replace('\n', '\n+')
            
        # Replace original code with generated mutant
        patch_content = original_item.get('patch')
        if original_code in patch_content:
            updated_patch = patch_content.replace(original_code, generated_code)
        
            # Create prediction object
            predictions.append({
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": updated_patch
            })
        else:
            error_mutant += 1


    # Summary
    print(f"📊 Summary:")
    print(f"  Total mutants: {len(results)}")
    print(f"  Error in mutants: {error_mutant}")
    print(f"  Total mutants: {len(predictions)}")
    
    return predictions

def split_jsonl_by_variants(input_file, output_dir="split_predictions", output_prefix="mutmut_predictions"):
    """
    Split JSONL file into multiple files where each file has one patch per instance
    
    Args:
        input_file: Input JSONL file path
        output_dir: Directory to save split files (default: "split_predictions")
        output_prefix: Prefix for output files (default: "predictions")
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group patches by instance_id
    instance_patches = defaultdict(list)
    
    # Read and group by instance_id
    with open(input_file, 'r') as f:
        for line in f:
            pred = json.loads(line.strip())
            instance_id = pred['instance_id']
            instance_patches[instance_id].append(pred)
    
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
        with open(output_file, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Generate Mutmut-based mutants and create SWE-Bench predictions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-file', '-i',
        default='data/swe_bench_lite/extracted_code_contexts.json',
        help='Input file with code contexts'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='data/mutmut_baseline',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--dataset-file', '-d',
        default='data/swe_bench_lite/swe_bench_lite.json',
        help='SWE-Bench dataset file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.dataset_file):
        print(f"Error: Dataset file '{args.dataset_file}' not found")
        sys.exit(1)
    
    output_file = f"{args.output_dir}/mutmut_output.jsonl"
    predictions_file = f"{args.output_dir}/mutmut_predictions.jsonl"
    process_json_file(args.input_file, output_file)

    predictions = create_prediction_format(args.dataset_file, output_file)

    # Save predictions to a file
    print("Saving prediction format to ", predictions_file)
    with open(predictions_file, 'w') as f:
        for line in predictions:
            json.dump(line, f)
            f.write('\n')

    split_jsonl_by_variants(predictions_file, args.output_dir)

if __name__ == "__main__":
    main()