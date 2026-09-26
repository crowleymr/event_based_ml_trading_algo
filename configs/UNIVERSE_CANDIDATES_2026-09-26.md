# Expanded-universe candidate enumeration

The source snapshot in `sp500_constituents_2026-09-26.csv` records the symbol,
CIK and displayed order of the [Wikipedia S&P 500 component stocks table](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), retrieved on
26 September 2026. Its 503 rows are current constituent listings, not a
historical membership file. The table says these are common stocks. The
snapshot is used solely to enumerate candidates; Yahoo remains the market
history source and SEC remains the identifier and fundamental source.

`universe_candidates_500.csv` is generated reproducibly by
`build_sp500_candidate_snapshot`: all 100 original `universe.txt` tickers
in their original order, followed by the remaining constituent symbols in
source order, truncated to 500. Dots in source symbols are converted to
hyphens using the existing SEC/Yahoo ticker convention. The source CIK is
carried into every candidate row and must exactly match the SEC mapping at
admission. `universe_candidates_500.txt` is the ordered ticker-only mirror;
neither file is an admitted trading universe.

The current-survivor limitation remains: today's membership and identifiers
cannot reconstruct historical index membership or delisted securities. The
separate admission ledger must establish the achieved count after each
security passes the frozen SEC and Yahoo checks. No candidate count is an
admitted or feature-ready count.

The source CSV SHA256 is
`07d0c5d0088c48446ad22a567a323b54a191cdb93b96842069cb1563bc9204ff`.
The candidate CSV SHA256 is
`1d3a9486a9966b2389bc6a71279cfd83deef7ed90cbe99714f6acc27f06a445a`.
The original 100-name universe SHA256 is
`9fe40543977929d862af55b1f550d4fddee9e5caae6302e3f4836921d5db58ef3`.

To regenerate and verify the candidate CSV without replacing it:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -c "from trading_pipeline.data.universe_admission import build_sp500_candidate_snapshot; build_sp500_candidate_snapshot('configs/sp500_constituents_2026-09-26.csv', 'configs/universe.txt', 'configs/universe_candidates_500.csv', as_of='2026-09-26')"
```

The live admission command, once Yahoo/SEC network access is available, is:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m trading_pipeline.data.universe_admission `
  --candidates configs\universe_candidates_500.csv `
  --output data\expanded_2026-09-26\universe_admissions `
  --snapshot-id sp500-candidates-2026-09-26-v1 `
  --start 2015-01-01 --end 2026-09-25 --min-sessions 260 `
  --sec-mapping-file data\raw\sec\company_tickers_exchange.json `
  --source-snapshot configs\sp500_constituents_2026-09-26.csv
```

Set `SEC_USER_AGENT` to a real contact before uncached SEC Company Facts
requests. The command writes into a new expanded data root. It resumes
per-security checks and never replaces a successful raw Yahoo/SEC cache.
The old 100-name universe, raw caches and completed runs remain untouched.
