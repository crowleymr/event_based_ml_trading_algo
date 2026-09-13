# Decisions and limitations

Implementation choices (not human research decisions): sklearn HistGradientBoostingRegressor;
filesystem tracking; deterministic fixed universe with accepted survivorship bias;
offline synthetic fixtures are software checks only, never research evidence.
SEC contact must be supplied through SEC_USER_AGENT; no fabricated contact.

2026-09-12: Corrected MMC to MRSH for the same issuer (CIK 0000062709).
Official SEC mapping and the issuer announcement confirm the 14 January 2026 ticker rename:
https://www.marsh.com/en/corp/about/news/marsh-mclennan-to-change-nyse-symbol-to-mrsh.html
This preserves the intended company universe; it is not an issuer substitution.

Execution detail: weekly signal on first observed session of each ISO week, then T+1 close.
The FSD specifies next-session execution but does not specify open/close. Close fills
avoid assuming access to adjusted open data. Old positions earn the return into the
execution close, new positions earn returns only after that close. Positions drift
between rebalances. Costs apply to actual absolute traded dollars, including initial
entry; terminal positions are marked to market without forced liquidation.

Split implementation: common eligible calendar after 20-session feature warmup,
60/20/20 boundaries, remove labels reaching the next boundary and embargo the first
five sessions after each boundary. Test tail is retained for valuation and excluded
from label metrics where its five-session label is unavailable. Latest fundamental
values are selected by filed date, latest fiscal end, shortest duration, accession,
then value; only USD and USD/shares enter models. Fiscal growth features deferred.

2026-09-13 documentation decision: retain a layered local research application with
`run.py` as the application-service orchestrator. Package boundaries improve review
and testing but are not microservices. Reporting and notebooks are read-only derivatives
of one specified immutable run, with outputs stored separately.

Experiment identifiers E0-E5 and B0 are immutable machine IDs. Human-readable semantic
labels are joined at presentation time from the registry. Future IDs are append-only;
completed run data is never retrospectively renamed.

Assignment-support material remains Markdown until generated evidence and the student's
own interpretation are ready. AI-action records remain factual. First-person reflection,
understanding, critique and knowledge gaps are student-authored only.
