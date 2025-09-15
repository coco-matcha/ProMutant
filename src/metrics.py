#!/usr/bin/env python3
"""
Simple SWE-Bench Coupling Metrics Calculator

Analyzes evaluation results to calculate coupling metrics for mutants.
"""

import json
import os
import sys
from pathlib import Path


def calculate_coupling_rate(directory_path):
    """Calculate coupling metrics for a single directory."""
    directory = Path(directory_path)
    
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory '{directory_path}' does not exist or is not a directory")
    
    # Initialize counters
    total_folders = 0
    weak_coupled = 0
    middle_coupled = 0
    strong_coupled = 0
    broken_instances = 0
    would_be_coupled = 0
    unkilled_mutants = 0
    total_footprint = 0
    
    # Process each instance folder
    for item in directory.iterdir():
        if not item.is_dir():
            continue
            
        total_folders += 1
        report_file = item / "report.json"
        
        if not report_file.exists():
            continue
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Analyze this instance
            instance_metrics = analyze_instance(report_data)
            
            weak_coupled += instance_metrics['weak_coupled']
            middle_coupled += instance_metrics['middle_coupled']
            strong_coupled += instance_metrics['strong_coupled']
            broken_instances += instance_metrics['broken']
            would_be_coupled += instance_metrics['would_be_coupled']
            unkilled_mutants += instance_metrics['unkilled']
            total_footprint += instance_metrics['footprint']
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error reading report.json in '{item.name}': {e}")
    
    return {
        'weak_coupled': weak_coupled,
        'middle_coupled': middle_coupled,
        'strong_coupled': strong_coupled,
        'total_folders': total_folders,
        'broken': broken_instances,
        'would_be_coupled': would_be_coupled,
        'unkilled': unkilled_mutants,
        'footprint': total_footprint
    }


def analyze_instance(report_data):
    """Analyze a single instance's report data."""
    # Initialize flags for this instance
    is_weak_coupled = True
    is_middle_coupled = True
    is_strong_coupled = True
    is_broken = False
    is_would_be_coupled = False
    is_unkilled = True
    instance_footprint = 0
    
    # Process each test result in the report
    for test_key, test_data in report_data.items():
        if not isinstance(test_data, dict) or 'tests_status' not in test_data:
            continue
        
        tests_status = test_data['tests_status']
        
        # Extract test results
        fail_to_pass_success = tests_status.get('FAIL_TO_PASS', {}).get('success', [])
        fail_to_pass_failure = tests_status.get('FAIL_TO_PASS', {}).get('failure', [])
        pass_to_pass_failure = tests_status.get('PASS_TO_PASS', {}).get('failure', [])
        pass_to_pass_success = tests_status.get('PASS_TO_PASS', {}).get('success', [])
        
        # Calculate semantic footprint
        instance_footprint += len(pass_to_pass_failure) + len(fail_to_pass_failure)
        
        # Check if instance was broken by the mutant
        if pass_to_pass_failure:
            is_broken = True
            # If also failing the same tests as the bug, would be coupled
            if not fail_to_pass_success:
                is_would_be_coupled = True
        else:
            # Check if mutant was unkillable (no failures in either category)
            if not fail_to_pass_failure:
                is_unkilled = True
            else:
                is_unkilled = False
        
        # Determine coupling criteria
        # Strong coupling: both fail_to_pass_success and pass_to_pass_failure are empty
        if fail_to_pass_success or pass_to_pass_failure:
            is_strong_coupled = False
        
        # Weak coupling: fail_to_pass_failure is not empty
        if not fail_to_pass_failure:
            is_weak_coupled = False
        
        # Middle coupling: fail_to_pass_failure not empty AND pass_to_pass_failure empty
        if not fail_to_pass_failure or pass_to_pass_failure:
            is_middle_coupled = False
    
    return {
        'weak_coupled': 1 if is_weak_coupled else 0,
        'middle_coupled': 1 if is_middle_coupled else 0,
        'strong_coupled': 1 if is_strong_coupled else 0,
        'broken': 1 if is_broken else 0,
        'would_be_coupled': 1 if is_would_be_coupled else 0,
        'unkilled': 1 if is_unkilled else 0,
        'footprint': instance_footprint
    }


def parse_run_specification(run_spec):
    """Parse run specification like 'test-run.1-30' into name and numbers."""
    if '.' not in run_spec:
        raise ValueError(f"Invalid run specification: {run_spec}")
    
    run_name, run_part = run_spec.split('.', 1)
    
    if '-' in run_part:
        # Range specification like "1-30"
        start_str, end_str = run_part.split('-', 1)
        start_num = int(start_str)
        end_num = int(end_str)
        run_numbers = list(range(start_num, end_num + 1))
    else:
        # Single run number
        run_numbers = [int(run_part)]
    
    return run_name, run_numbers


def aggregate_metrics(run_name, run_numbers, model_name):
    """Aggregate metrics across multiple runs."""
    total_metrics = {
        'weak_coupled': 0,
        'middle_coupled': 0,
        'strong_coupled': 0,
        'total_folders': 0,
        'broken': 0,
        'would_be_coupled': 0,
        'unkilled': 0,
        'footprint': 0
    }
    
    max_footprint = 0
    min_footprint = float('inf')
    successful_runs = 0
    
    for run_num in run_numbers:
        directory_path = f"logs/run_evaluation/{run_name}.{run_num}/{model_name}"
        
        try:
            run_metrics = calculate_coupling_rate(directory_path)
            
            # Aggregate totals
            for key in total_metrics:
                total_metrics[key] += run_metrics[key]
            
            # Track footprint extremes
            if run_metrics['footprint'] > max_footprint:
                max_footprint = run_metrics['footprint']
            if run_metrics['footprint'] < min_footprint and run_metrics['footprint'] > 0:
                min_footprint = run_metrics['footprint']
            
            successful_runs += 1
            print(f"Processed run {run_num}: {run_metrics['total_folders']} instances")
            
        except ValueError as e:
            print(f"Warning: Skipping run {run_num}: {e}")
    
    if successful_runs == 0:
        raise ValueError(f"No valid runs found for {run_name} with model {model_name}")
    
    # Handle special case for min_footprint
    if min_footprint == float('inf'):
        min_footprint = 0
    
    total_metrics['max_footprint'] = max_footprint
    total_metrics['min_footprint'] = min_footprint
    
    return total_metrics


def print_results(metrics, run_description):
    """Print the analysis results."""
    total = metrics['total_folders']
    
    print("\n" + "=" * 50)
    print(f"ANALYSIS RESULTS for {run_description}:")
    print("=" * 50)
    print(f"Total mutants count: {total}")
    print(f"Instances broken by mutants: {metrics['broken']}")
    print(f"Instances that would have been coupled if not broken: {metrics['would_be_coupled']}")
    print(f"Mutants coupled with bugs: {metrics['weak_coupled']}")
    print(f"Mutants middle-coupled with bugs: {metrics['middle_coupled']}")
    print(f"Mutant strong coupled with bugs: {metrics['strong_coupled']}")
    
    if total > 0:
        print(f"Weak coupling rate: {(metrics['weak_coupled']/total*100):.1f}%")
        print(f"Middle coupling rate: {(metrics['middle_coupled']/total*100):.1f}%")
        print(f"Strong coupling rate: {(metrics['strong_coupled']/total*100):.1f}%")
        print(f"Mutation Score: {1 - (metrics['unkilled']/total):.2f}")
        
        print(f"Semantic Footprint:")
        print(f"Average: {(metrics['footprint']/total):.2f}")
        print(f"Max: {metrics['max_footprint']}")
        print(f"Min: {metrics['min_footprint']}")
    else:
        print("Coupling rates: N/A (no mutants found)")
    
    print(f"Unkillable mutants: {metrics['unkilled']}")
    
    print("\nCriteria for strong coupling: Both FAIL_TO_PASS.success and PASS_TO_PASS.failure lists are empty")
    print("Criteria for coupling: FAIL_TO_PASS.failure not empty")


def main():
    """Main function."""
    try:
        # Get user input
        model_name = input("Model name: ").strip()
        if not model_name:
            print("Error: Model name is required")
            sys.exit(1)
        
        run_spec = input("Run ID (for example test-run.1-30): ").strip()
        if not run_spec:
            print("Error: Run specification is required")
            sys.exit(1)
        
        # Parse run specification
        run_name, run_numbers = parse_run_specification(run_spec)
        
        # Aggregate metrics across runs
        print(f"Processing {len(run_numbers)} runs...")
        metrics = aggregate_metrics(run_name, run_numbers, model_name)
        
        # Create run description
        if len(run_numbers) == 1:
            run_desc = f"{run_spec}"
        else:
            run_desc = f"{run_name}.{min(run_numbers)}-{max(run_numbers)}"
        
        # Print results
        print_results(metrics, run_desc)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()