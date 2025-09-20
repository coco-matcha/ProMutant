import json
import os
import random
import requests
from colorama import init, Fore, Style

# CHANGE FILENAME HERE
FILENAME = "test_sampled_mutants.json"
# SKIP THE FIRST 0 SAMPLES
SKIP = 0


def clear_screen():
    # For Windows
    if os.name == 'nt':
        os.system('cls')
    # For macOS and Linux
    else:
        os.system('clear')

def get_commit_url(instance_id: str):
    """Get the url of related commit"""
    repo_author = instance_id.split('__', 1)[0]
    repo = instance_id.split('__', 1)[1]
    pr = repo.split('-')[-1]
    repo_name = '-'.join(repo.split('-')[:-1])
    pr_response = requests.get(f"https://api.github.com/repos/{repo_author}/{repo_name}/pulls/{pr}")
    commit_sha = pr_response.json().get('merge_commit_sha')
    if commit_sha:
        commit_url = f"https://github.com/{repo_author}/{repo_name}/commit/{commit_sha}"
    else: 
        commit_response = requests.get(f"https://api.github.com/repos/{repo_author}/{repo_name}/pulls/{pr}/commits")
        commit_sha = commit_response.json()[-1]['sha']
        commit_url = f"https://github.com/{repo_author}/{repo_name}/commit/{commit_sha}"

    return commit_url

def load_json_file(filename,  sample: int = 236):
    """Load and parse JSON file"""
    try:
        if filename.endswith('.json'):
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
        
        sampled_mutants = data
        
        return sampled_mutants
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{filename}': {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def display_object(obj, index, total):
    """Display a single object with formatting"""
    commit_url = get_commit_url(obj.get('instance_id'))
    print("\n" + "="*50)
    print(f"Object {index + 1} of {total}")
    print(f"Mutant ID: {obj.get('mutant_id')}")
    print(f"Commit URL: {commit_url}")
    print("="*50)
    print("Code for Evaluation:")
    for line in obj.get('full_function').splitlines():
        if line.startswith("-"):
            print(Fore.RED + line)
        elif line.startswith("+"):
            print(Fore.GREEN + line)
        else:
            print(Fore.WHITE + line)
    print("="*50)

def browse_json_objects(filename, sample: int = 236):
    """Main function to browse JSON objects"""
    data = load_json_file(filename, sample)
    
    if data is None:
        return
    
    # Handle different JSON structures
    if isinstance(data, list):
        objects = data
    elif isinstance(data, dict):
        # If it's a dict, treat each key-value pair as an object
        objects = [{"key": k, "value": v} for k, v in data.items()]
    else:
        # If it's a single value, wrap it in a list
        objects = [data]
    
    if not objects:
        print("No objects found in the JSON file.")
        return
    
    print(f"Found {len(objects)} objects in '{filename}'")
    index = 0 + SKIP
    total = len(objects)

    label_results = []
    acceptable_inputs = ["0", "1", "2", "3", "4", "5"]

    while index < total:
        display_object(objects[index], index, total)

        current_object = {
            'mutant_id': objects[index].get('mutant_id'),
            'natural': None,
            'equivalent': None
        }
        
        # Wait for user input
        print("Check for equivalent: Does the changed program behaves identically to the original program for ALL possible inputs?")
        user_input = input("Press 0 for NO, 1 for YES, 2 for UNSURE, or 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("Goodbye!")
            break
        elif user_input in acceptable_inputs:
            current_object["equivalent"] = user_input

        # Wait for user input
        print("\nCheck for natural: Does the change represent a REALISTIC change that a developer might make?")
        user_input = input("Evaluate on a scale of 1(Strongly Unnatural) to 5(Strongly Natural), or press 'q' to quit: ").strip().lower()
        
        if user_input == 'q':
            print("Goodbye!")
            break
        elif user_input in acceptable_inputs:
            current_object["natural"] = user_input

        if current_object["equivalent"] or current_object["natural"]:
            label_results.append(current_object)
            with open(f"label_{FILENAME[:-5]}.jsonl", 'a') as f:
                json.dump(current_object, f)
                f.write('\n')

        index += 1
        
        # Clear screen for better readability (optional)
        clear_screen()
    
    if index >= total:
        print("\nReached the end of all objects!")
    
    return label_results

def main():
    """Main entry point"""
    print("JSON Object Browser")
    print("-" * 20)
    
    if not FILENAME:
        print("No filename provided.")
        return
    
    # Change sample size here
    clear_screen()
    init(autoreset=True)
    label_results = browse_json_objects(FILENAME)
    print(f"Finished labeling {len(label_results)} samples! 🥳")


if __name__ == "__main__":
    main()
