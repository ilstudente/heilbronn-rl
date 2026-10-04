# Heilbronn RL

An agentic RL environment for the Heilbronn triangle problem, and GRPO training of Qwen3.8-27B in it.

**Task.** Place points in a region so the smallest triangle they form is as large as possible. The model works in a
tool loop (`bash`, `write_file`), writes and runs its own search code, and submits candidate configurations with
`python3 attempt.py FILE.json`. Every attempt is graded exactly by the environment, never by the agent.

**Files**

- `hull_heilbronn.py`: the environment: random instances, exact grading, episode rewards and group-relative advantages
- `hull_grpo.py`: GRPO with LoRA on one H200 on Modal. vLLM samples rollouts (token in / token out), then sleeps while
  the trainer takes a truncated-importance-weighted policy-gradient step; the run tracks its own spend and stops at a
  dollar cap
- `hull_lab.py`, `hull_lab.html`: a local lab for running Claude or Qwen agents on instances and watching them live
- `qwen_serve.py`, `qwen_agent.py`: an OpenAI-compatible vLLM endpoint for Qwen3.8-27B on Modal, and the agent loop
- `grpo_dash.py`, `grpo_dash.html`: dashboard for training runs
- `grpo_runs/`: configs, metrics and rollouts from the training runs

**Results.** Evaluation is the classic Heilbronn square with 16 points (best known: 7/341), 16 rollouts per
evaluation, scored as a percentage of the best known result. Five evaluations of the base model averaged about 13%
(range 7.8–17.4%). The final adapter scored 29.9% and 30.0% in two separate evaluations. Samples are small and
noisy: one earlier checkpoint scored 33.0% and then 12.7% on re-evaluation.

**Run**

    modal run --detach hull_grpo.py            # full run (budget-capped)
    modal run --detach hull_grpo.py --pilot    # one training step, no evals
