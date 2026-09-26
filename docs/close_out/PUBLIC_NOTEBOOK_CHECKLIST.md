# Public Notebook / Final Submission Checklist

The A2 specification requires a plain-text URL to a publicly accessible cloud Python notebook.

## Notebook sections

1. **Project summary**
2. **Environment setup**
3. **Configuration**
4. **Data acquisition / archived data retrieval**
5. **Security master**
6. **Market + SEC preprocessing**
7. **Point-in-time feature construction**
8. **Target construction**
9. **Temporal split / purge**
10. **Elastic Net**
11. **GBT**
12. **Predictions / IC**
13. **Portfolio construction**
14. **Backtest**
15. **Baselines**
16. **Results**
17. **Limitations**
18. **Links to full repository/run artefacts**

## Must verify

```text
[ ] Public without requesting access
[ ] Plain-text URL placed in final PDF
[ ] Fresh runtime executes
[ ] Dependency versions controlled
[ ] Data preparation included
[ ] No local D:\ paths
[ ] No API secrets
[ ] Representative data download works
[ ] Same formulas/config as submitted repo
[ ] Results are either reproducible or clearly loaded from an archived canonical run
```

## Recommended strategy

If the full local pipeline is too expensive or fragile for Colab:
- keep the implementation complete;
- run a smaller representative universe/history in the notebook;
- include/download the immutable final full-run artefacts;
- state transparently which results came from the full canonical run.

Do not silently present loaded results as though they were generated in the current notebook execution.
