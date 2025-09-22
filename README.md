# LiteCoST
An RL-enhanced framework designed to automatically finetune lightweight SLMs for structured knowledge extraction and analytics.

## Overview
LiteSEA first constructs high-quality Chain-of-Structured-Thought (CoST) data through structure analysis, data distillation, LLM-based automatic verification, and iterative refinement. 
then trains the model via Supervised Fine-tuning (SFT) for initialization and format adaptation, followed by Group Relative Policy Optimization (GRPO) with dual-level rewards to enhance structured extraction capabilities.

## Method
### CoST: Structure-First Reasoning and Trace Generation
1. Structure Analysis
2. Data Distillation
3. Data Verification
4. Data Refinement


### SLM Fine-Tuning: SFT → GRPO
1. Supervised Fine-Tuning (SFT)
2. Group Relative Poilcy Optimization (GRPO)


## Architecture
The core execution of **LiteCoST** is implemented in the src directory:
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
1. Generate the Serialized Structured Output 
```python
python main.py --model gpt-4o --dataset Loong --structured --document

cd src
python data_verification.py
python data_refinement.py
```

2. Conduct SFT Training
```python
python sft.py
```

3. Conduct GRPO Optimization
```python
python grpo.py
```
