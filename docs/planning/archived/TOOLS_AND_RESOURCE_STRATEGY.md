# Tooling and Resource Strategy

## Recommended role allocation

The objective is to maximise reliable implementation time while preserving the user's limited cognitive bandwidth.

### 1. Primary coding agent — Claude Code OR Codex

Use **one primary coding agent**, not three competing agents.

Recommended workflow:

- give the agent the FSD and implementation brief;
- let it own the repository implementation;
- require tests and commits;
- review major design decisions yourself.

Use the second coding agent as a **reviewer/debugger**, not as a parallel implementation stream.

### 2. Cursor

Useful if you prefer an IDE-centric workflow.

Do not use Cursor + Claude Code + Codex simultaneously for the same codebase during the POC. This increases merge/conflict and cognitive overhead.

### 3. Perplexity

Use the free accounts for targeted research questions:

- verify data sources;
- compare historical-data providers;
- investigate dataset licensing/limitations;
- find documentation.

Do not delegate implementation planning to Perplexity if the coding agent already has the FSD.

### 4. Google Antigravity

Treat as an optional experimentation environment. Do not make the POC dependent on an unfamiliar platform.

---

# Recommended Operating Model

## One builder + one reviewer

```text
You
 |
 | research decisions / approval
 v
Primary coding agent
 |
 | implementation + experiments
 v
GitHub repository
 |
 +--> Reviewer agent
       |
       +--> leakage audit
       +--> code review
       +--> test review
       +--> result sanity check
```

This is preferable to asking three agents to build overlapping implementations.

---

# Token/Usage Strategy

Because the POC window is only 1.5 days:

### Spend agent context on

- implementation;
- debugging;
- tests;
- experiment execution;
- documentation.

### Avoid spending agent context on

- repeatedly discussing architecture;
- building elaborate abstractions;
- speculative product design;
- unnecessary cloud infrastructure;
- premature dashboard polish.

The FSD should act as the persistent source of truth so that agent context is not repeatedly consumed re-explaining the project.

---

# Suggested Agent Prompts

## Initial builder prompt

> Read FSD.md, AI_AGENT_IMPLEMENTATION_BRIEF.md and EXPERIMENT_PLAN.md. Implement the POC incrementally. Do not add deferred features until all mandatory acceptance criteria pass. Start by inspecting the repository and validating the data source. Before major architectural changes, explain the decision and stop for approval. For routine implementation, proceed autonomously. Every completed stage must include tests and a short implementation-log entry.

## Reviewer prompt

> Review the current trading POC against FSD.md and AI_AGENT_IMPLEMENTATION_BRIEF.md. Focus on leakage, temporal validation, target construction, execution timing, transaction costs, portfolio accounting and reproducibility. Do not rewrite the project. Produce a prioritised list of defects and risks, with tests that would demonstrate each issue.

---

# Recommended Local-First MLOps

Do this first:

```text
Python environment
      ↓
pytest
      ↓
single-command pipeline
      ↓
runs/<run_id>/ artefacts
      ↓
MLflow
```

Then:

```text
GitHub Actions
      ↓
tests
      ↓
training
      ↓
artefact upload
```

Avoid introducing AWS/SageMaker for this POC unless there is a specific requirement. Cloud orchestration is not itself research value.

---

# Decision Rule for the Weekend

If the POC is not end-to-end by roughly the halfway point:

**cut features, not reliability.**

Specifically remove in this order:

1. RL;
2. RNN;
3. news;
4. MPT;
5. dashboard.

Do not remove:

- temporal validation;
- leakage tests;
- baseline;
- backtest;
- financial metrics;
- experiment persistence.

A smaller scientifically credible experiment is much stronger than a large but unreliable system.
