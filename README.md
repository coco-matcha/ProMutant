# ProMutant

This is the replication package for the paper "Back Translation as Mutation: Prompt-Code Round-Trip Perturbations with LLMs". This package contains all code, data, and tools necessary to reproduce the experiments and results presented in the paper.

## Overview

ProMutant is a novel approach that uses Large Language Models (LLMs) to generate semantically meaningful mutants for analyzing the coupling between bugs and test failures in software projects. This package includes implementations of ProMutant and baseline methods, along with evaluation tools and human study data.

## Repository Structure

```
.
├── src/                         # Source code and scripts
│   ├── baselines/               # Baseline method implementations
│   │   ├── generate_mutmut_mutants.py       # Traditional baseline
│   │   └── generate_llm_baseline_mutants.py # LLM baseline
│   ├── generate_mutants.py                 # Generate mutants for ProMutant
│   ├── create_prediction_format.py         # Create patch format for SWE-bench evaluation
│   ├── metrics.py                          # Calculate coupling rate from SWE-bench evaluation logs
│   ├── run_promutant.sh                   # Script to run ProMutant
│   └── human_study/             # Human evaluation code
│       ├── sample_baseline_mutants.py      # Sample from baseline output
│       └── sample_promutant_mutants.py     # Sample from ProMutant output
├── data/                        # Data files and results
│   ├── swe_bench_lite/          # SWE-Bench Lite dataset
│   │   ├── swe_bench_lite.json                 # Original dataset
│   │   └── extracted_code_contexts.json        # Extracted code contexts
│   ├── promutant/                  # ProMutant generated mutants
│   │   └── ablation/                   # Ablation study data
│   ├── llm_baselines/              # Baseline generated mutants
│   ├── mutmut_baselines/           # Traditional baseline generated mutants
│   └── human_study/                # Human evaluation data, with sampled mutants and annotations
│        └──labels/                     # Human annotations
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM-based methods)
- [SWE-Bench evaluation environment](https://github.com/SWE-bench/SWE-bench)
- [Patchutils](https://github.com/twaugh/patchutils)
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd promutant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Quick Start

#### Running ProMutant

Generate mutants using ProMutant with default settings:
```bash
python src/generate_mutants.py --api-key $OPENAI_API_KEY
```

#### Running Baselines

Generate mutants using the LLM baseline:
```bash
python src/baselines/generate_llm_baseline_mutants.py --api-key $OPENAI_API_KEY
```

Generate mutants using traditional baseline:
```bash
python src/baselines/generate_mutmut_mutants.py
```

#### Running ProMutant + Evaluating Results

Run ProMutant, SWE-Bench evaluation and calculate metrics:
```bash
./src/evaluation/run_promutant.sh --api-key $OPENAI_API_KEY
```

Calculate coupling metrics from evaluation logs:
```bash
python src/evaluation/metrics.py
```

## Detailed Usage

### ProMutant Pipeline
```bash
./src/run_promutant.sh
```

The ProMutant pipeline consists of several stages:

1. **Mutant Generation**: Use back-translation to generate semantically meaningful mutants
-> See src/generate_mutants.py
2. **Format Conversion**: Convert output to SWE-Bench evaluation format
-> See src/create_prediction_formats.py
3. **Evaluation**: Run evaluation using SWE-Bench harness
-> An evaluation logs directory will be created in current working directory
4. **Metric Evaluation**: Calculate coupling metrics
-> See src/metrics.py

### Evaluation and Metrics

#### Running SWE-Bench Evaluation
To evaluate results using SWE-bench harness, we run the following:
```bash
python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path "prediction-format-file" \
    --max_workers 8 \
    --run_id "run-id.file-number"
```

#### Calculating Coupling Metrics
After evaluation, calculate coupling metrics:
```bash
python src/metrics.py
# Follow interactive prompts
```

Metrics calculated:
- **Weak Coupling Rate**: Percentage of mutants that fail at least one test that detected the original bug
- **Middle Coupling Rate**: Percentage of mutants that fail bug-revealing tests but not regression tests
- **Strong Coupling Rate**: Percentage of mutants that fail only bug-revealing tests
- **Mutation Score**: Percentage of mutants killed by tests
- **Semantic Footprint**: Average number of test failures per mutant

### Human Study

#### Sampling Mutants for Human Evaluation
```bash
# Sample 150 from ProMutant output
python src/sampling/sample_promutant_mutants.py

# Sample 150 from baseline output
python src/sampling/sample_baseline_mutants.py
```

## Data Description

### SWE-Bench Lite Dataset
- **swe_bench_lite.json**: Original dataset with 300 instances
- **buggy_code_contexts.json**: Extracted code contexts for each instance

### Generated Mutants
- **ProMutant mutants**: Located in `data/promutant/generated_mutants/`
    - **baseline_config_gpt-4o_output.jsonl**: Mutants produced by GPT-4o, number of loops depend on number of lines in buggy code
    - **2_loops_gpt-4o_output.jsonl**:  Mutants produced by GPT-4o, number of loops equal 2 for each code hunk
    - **3_loops_gpt-4o_output.jsonl**:  Mutants produced by GPT-4o, number of loops equal 2 for each code hunk
    - **promutant_gpt-4.1_output.jsonl**: Mutants produced by GPT-4.1, number of loops equal 3 for each code hunk
- **Traditional Baseline mutants**: Located in `data/mutmut_baselines/mutmut_output.jsonl`
- **LLM Baseline mutants**: Located in `data/llm_baselines`
    - **baseline_config_gpt-4o_output.jsonl**: Mutants produced by GPT-4o, number of loops depend on number of lines in buggy code
    - **baseline_gpt-4.1_output.jsonl**: Mutants produced by GPT-4.1, number of loops equal 3 for each code hunk
    - **baseline_gpt-4o__limited_output.jsonl**: Mutants produced by GPT-4o, number of loops equal 1 for single-line hunk and 2 otherwise


### Human Study Data
- **Sampled mutants**: 150 randomly selected mutants from each method in `data/human_study`
- **Human labels**: Annotations by human raters, as well as the tie-breaker labels in `data/human_study/labels`
