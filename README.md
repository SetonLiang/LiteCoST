# LiteSEA
An RL-enhanced framework designed to automatically finetune lightweight SLMs for structured knowledge extraction and analytics.

## Overview
LiteSEA first constructs high-quality Chain-of-Structured-Thought (CoST) data through structure analysis, data distillation, LLM-based automatic verification, and iterative refinement. 
then trains the model via Supervised Fine-tuning (SFT) for initialization and format adaptation, followed by Group Relative Policy Optimization (GRPO) with dual-level rewards to enhance structured extraction capabilities.

## Method
### Chain-of-Structured-Thought (CoST) Data Generation
1. Structure Analysis
2. Data Distillation
3. Data Verification
4. Data Refinement


### Two-step Model Training
1. Supervised Fine-Tuning (SFT)
2. Group Relative Poilcy Optimization (GRPO)


## Architecture
The core execution of **LiteSEA** is implemented in the src directory:
```text
src
├── convert_func.py              # Conversion function module
├── data_refinement.py           # Data refinement module
├── data_verification.py         # Data verification module
├── extract/                     # Extraction module
│   ├── graph.py                 # Graph class
│   ├── main.py                  # Main program
│   ├── table.py                 # Table class
│   ├── to_desc.py               # Convert to description
│   ├── to_graph.py              # Convert to graph 
│   └── to_table.py              # Convert to table
├── grpo.py                      # GRPO module
├── sft.py                       # SFT module
├── prompt.py                    # Prompt template module
├── reasoner.py                  # Reasoning module
├── reward.py                    # Reward module
├── structure_analysis/          # Structure analysis module
│   ├── query2schema.py          # Schema construction
│   └── structure_decision.py    # Structure decision
└── utils.py                     # Utility functions module
```

## Usage
Construct the CoST Data:
```
python main.py --model gpt-4o --dataset Loong --structured --document
```
