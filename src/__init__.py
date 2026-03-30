"""
RLHF Pipeline: Reward Modeling & Policy Optimization.

Three-stage pipeline implemented on GPT-2 / Anthropic hh-rlhf:
  1. SFT   — supervised fine-tuning on chosen responses
  2. RM    — reward model trained via Bradley-Terry preference loss
  3. PPO   — proximal policy optimization using RM signal
  3. DPO   — direct preference optimization (no explicit RM at train time)
"""
