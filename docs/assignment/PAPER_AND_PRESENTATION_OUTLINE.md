# Paper and presentation outline

## Eventual paper

1. **Title and abstract** — precise task, data/split, two model families, portfolio translation and appropriately bounded conclusion.
2. **Problem and motivation** — why cross-sectional ranking is a practical ML problem and why PIT correctness/costs matter.
3. **Related concepts** — regularized linear models, boosted trees, temporal validation and portfolio evaluation.
4. **Data and point-in-time policy** — universe, Yahoo/SEC sources, identifier bridge, schemas, strict filed-date rule and limitations.
5. **Methods** — F0/F1, target, split, model objectives, fixed grids, validation-only selection, P1/P2 and T+1 accounting.
6. **System design and AI assistance** — layered local architecture, immutable artefacts, audit controls, AI actions and independent verification.
7. **Results** — generated experiment/split table; generated equity, drawdown, IC and turnover/cost evidence; no hand-entered numbers.
8. **Discussion** — relationship among forecast error, IC and portfolio outcomes; limitations, implications and failure modes.
9. **Conclusion and future work** — answer only what the evidence supports; specify new holdout requirements.
10. **Personal reflection** — student-authored evidence only, guided by [STUDENT_REFLECTION_PROMPTS.md](STUDENT_REFLECTION_PROMPTS.md).
11. **Appendix/reproducibility** — run ID, report version, hashes, code revision, commands, registry and metric definitions.

Before submission, every numeric claim should trace to a generated output, and every generated output should trace to one immutable run.

## Five-minute presentation

| Time | Slide | Purpose and evidence |
|---:|---|---|
| 0:00-0:35 | Problem | Decision, prediction target and research question |
| 0:35-1:15 | Data and leakage controls | Yahoo + SEC, strict filed-date rule, purged/embargoed timeline |
| 1:15-2:00 | Models and pipeline | Elastic Net vs histogram GBT, F0 vs F1, one architecture diagram |
| 2:00-3:05 | Evaluation | Generated comparison plus one IC/prediction observation |
| 3:05-4:00 | Portfolio behavior | Generated equity/drawdown and turnover/cost view; P1 vs P2 |
| 4:00-4:35 | Limitations | Survivorship, accounting comparability, execution and single split |
| 4:35-5:00 | Learning and next step | Student's own reflection and one gated future-work decision |

Avoid dense code screenshots and tables that cannot be read. State that test results are descriptive and were not used for selection.
