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


## Usage
Construct the CoST Data:
```
python main.py --model gpt-4o --dataset Loong --structured --document
```
