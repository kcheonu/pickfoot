# AGENTS.md --- Football Match Prediction

## 1. Project purpose

Build a maintainable football match prediction application that
estimates the probabilities of:

-   Home win
-   Draw
-   Away win

The project should combine historical match results, team form, starting
lineups, player appearances, and player-combination features. The goal
is a usable prediction service, not merely a one-off notebook.

The initial target competitions are the European top five domestic
leagues:

-   Premier League (England)
-   La Liga (Spain)
-   Bundesliga (Germany)
-   Serie A (Italy)
-   Ligue 1 (France)

**Important data limitation:** StatsBomb Open Data is free and useful
for prototyping, but it covers only selected competitions and seasons.
Never claim that it supplies complete historical coverage for all five
leagues. Inspect the actual competition and season metadata first,
record coverage, and represent unavailable data explicitly. Keep
data-source integrations replaceable so a broader-coverage API provider
can be added later without rewriting the prediction domain.

## 2. Data-source policy

### Initial source: StatsBomb Open Data

Use the official public dataset: - Repository:
https://github.com/statsbomb/open-data - Data format: JSON - Typical
data groups: competitions, matches, events, lineups, and (where
available) 360 data.

Use only competitions and seasons actually present in the repository. Do
not invent fixtures, lineups, player records, or statistics to fill
gaps. Preserve source identifiers and attribution information.

If a future provider such as API-Football is added to achieve broader
top-five-league coverage: - Implement it as a separate provider
adapter. - Check current documentation, coverage, plan limits, terms,
and rate limits before using it. - Keep provider-specific IDs separate
from internal database IDs. - Never scrape or bypass access controls
when an official API or permitted dataset is available.

### Data quality requirements

-   Store the provider/source name and source identifier for imported
    records.
-   Make ingestion idempotent: re-running a sync must not create
    duplicate matches or players.
-   Validate payloads and report malformed or incomplete records.
-   Preserve the difference between a genuine zero and a missing value.
-   Store timestamps consistently in UTC; retain the original kickoff
    timezone if supplied.
-   Distinguish scheduled, in-progress, completed, postponed, and
    cancelled matches when the source provides status.
-   Never assume that every match has lineup, player-stat, or event
    data.
-   Log sync status, counts, errors, and source coverage without
    exposing secrets.

## 3. Recommended technology

Prefer a simple modular monolith until real requirements justify added
infrastructure.

-   Python 3.12+
-   `httpx` for HTTP access when needed
-   `pydantic` / `pydantic-settings` for validation and configuration
-   `pandas` for data processing
-   `SQLAlchemy 2.x` and `Alembic` for PostgreSQL persistence and
    migrations
-   `scikit-learn` for baseline models and evaluation
-   `xgboost` only after a reliable baseline exists
-   `pytest` for automated tests
-   FastAPI for a prediction API
-   Streamlit or a separate frontend for an initial dashboard
-   Docker Compose for local PostgreSQL and application development when
    useful

Use the repository's existing dependencies and conventions if they are
already established. Do not add a dependency without a clear need.

## 4. Suggested project layout

Keep modules focused and responsibilities separate. Adapt this layout to
the existing repository rather than blindly replacing existing files.

``` text
football-predictor/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── configs/
│   ├── competitions.yaml
│   └── model.yaml
├── src/
│   └── football_predictor/
│       ├── config.py
│       ├── domain/
│       │   ├── match.py
│       │   └── prediction.py
│       ├── ingestion/
│       │   ├── base.py
│       │   ├── statsbomb_open_data.py
│       │   └── sync.py
│       ├── db/
│       │   ├── models.py
│       │   ├── session.py
│       │   └── repositories.py
│       ├── features/
│       │   ├── team_form.py
│       │   ├── elo.py
│       │   ├── lineup_features.py
│       │   └── build_dataset.py
│       ├── training/
│       │   ├── train.py
│       │   └── evaluate.py
│       ├── prediction/
│       │   └── predict_match.py
│       └── api/
│           └── main.py
├── scripts/
│   ├── inspect_coverage.py
│   ├── sync_data.py
│   └── train_model.py
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── artifacts/
└── frontend/
```

Keep generated data and model artifacts out of source control unless
explicitly intended. Do not commit secrets, large raw datasets, or
downloaded data files without checking repository policy and dataset
terms.

## 5. Data model and persistence

Use PostgreSQL for the application database when persistence is
introduced. Use migrations for schema changes. The initial logical
entities should cover:

-   `competitions`: provider, provider competition ID, name, country.
-   `seasons`: competition, provider season ID, season label and date
    range.
-   `teams`: provider, provider team ID, canonical name.
-   `players`: provider, provider player ID, canonical name.
-   `matches`: provider match ID, competition, season, kickoff, home
    team, away team, status, goals when available.
-   `lineup_players`: match, team, player, starter flag, position for
    that match, and minutes played when available.
-   `team_match_stats`: match/team-level statistics that the provider
    actually supplies.
-   `player_match_stats`: match/player-level statistics that the
    provider actually supplies.
-   `data_sync_runs`: source, start/end time, status, counts, and error
    summary.
-   `predictions`: match, prediction timestamp, model version,
    probabilities, feature/lineup snapshot reference.

Database rules:

-   Use internal primary keys and preserve external provider IDs
    separately.
-   Add appropriate unique constraints, such as
    `(provider, provider_match_id)` for matches.
-   Prevent duplicate player entries for the same match and team.
-   Model player appearances per match; do not assume a player's current
    team or position represents their historical state.
-   Do not force unsupported fields into the schema. Add
    provider-specific raw payload storage only when it has a clear use
    and privacy/storage implications are understood.
-   Use transactions for related writes and safe upserts for repeated
    ingestion.
-   Do not store missing statistics as zero.

## 6. Lineup and player-combination features

Build features only from information available at the prediction cutoff
time.

Start with simple, interpretable features: - Team form over the previous
5 and 10 matches. - Previous goals for and against, split by home/away
where useful. - Elo or another time-updated team-strength rating. - Rest
days and schedule congestion when reliable dates are available. -
Starting-player availability and lineup continuity when lineups are
available before kickoff. - Count of shared appearances for player
pairs. - Shared minutes and team goal difference during shared
appearances, where the data supports these measures. - Formation and
positional balance when reliably recorded.

Do not immediately generate every possible player pair or use player
names as arbitrary numeric features. Start with team-level features,
establish a baseline, then add lineup features and player-combination
features as separate experiments.

For player-combination statistics: - Require a minimum sample size or
use smoothing/shrinkage. - Account for minutes played rather than
treating every appearance equally. - Avoid interpreting raw pair win
rates as causal effects. - Recognize that opponent strength, player
quality, tactical role, and match context can confound pair
statistics. - Compare against a baseline model to measure whether each
feature group actually improves performance.

## 7. Prediction and evaluation rules

The target is three-class classification: home win, draw, away win.

Build a simple baseline first, such as Elo-derived probabilities and/or
multinomial logistic regression. Consider Random Forest or XGBoost only
after the data pipeline and baseline evaluation are reliable.

Use chronological validation: - Train on earlier matches. - Validate on
a later period. - Test on a still later period when data volume
permits. - Never randomly mix future and past matches in a way that
leaks information.

Prevent leakage: - For a historical prediction, use only features
available before that match's kickoff or the explicitly defined
prediction cutoff. - Do not use final scores, post-match statistics,
completed lineups that were not known at the simulated cutoff, or future
matches to calculate historical features. - Compute rolling statistics
using past matches only. - Fit preprocessing and calibration only on
training/validation data, not the final test set. - Store the prediction
timestamp, model version, and feature snapshot or equivalent
reproducibility metadata.

Evaluate more than accuracy. Report multiclass log loss, Brier score
where implemented correctly, calibration, and confusion matrix. Compare
baseline, team-form, lineup, and player-combination feature sets on the
same time-based test window. Never invent or claim performance
improvements that have not been measured.

Return three probabilities that are each in `[0, 1]` and sum to
approximately `1`. If the input data is missing or out of distribution,
surface that limitation rather than presenting unjustified certainty.

## 8. API, credentials, and operational safety

-   Read secrets from environment variables or a local `.env` file
    excluded by `.gitignore`.
-   Keep `.env.example` limited to placeholder values.
-   Never print API keys or include them in exceptions, URLs, committed
    files, screenshots, or test fixtures.
-   Set HTTP timeouts and bounded retry behavior.
-   Respect source terms, rate limits, and attribution requirements.
-   Handle network failures and malformed responses with actionable
    errors.
-   Make data-sync operations safe to retry.
-   Keep API handlers thin; business logic belongs in services/domain
    modules.
-   Record prediction and sync errors with enough context to debug,
    without logging credentials.
-   Do not claim a sync or test succeeded unless it actually ran and its
    result was checked.

## 9. Testing requirements

Add tests alongside each feature. At minimum:

-   Unit tests for provider JSON parsing and schema validation.
-   Tests for missing optional fields and malformed payloads.
-   Tests that repeated ingestion does not duplicate records.
-   Tests for match outcome labels and probability normalization.
-   Tests for rolling features that prove future matches do not affect
    past rows.
-   Tests for player-combination features with small samples and missing
    appearances.
-   Integration tests for database writes and migrations when a test
    database is available.
-   Use saved, anonymized or appropriately licensed fixtures; unit tests
    must not depend on a live external API.

Run relevant tests after code changes. If a command is blocked, requires
approval, or cannot run in the current environment, report that clearly
rather than claiming success.

## 10. Copilot working protocol

For every task:

1.  Read this file and inspect the relevant existing files before
    editing.
2.  Summarize the intended change and identify assumptions or
    data-source limitations.
3.  Make the smallest coherent change; do not rewrite unrelated code.
4.  Keep functions typed, focused, and documented where behavior is
    non-obvious.
5.  Add or update tests for changed behavior.
6.  Run the relevant tests, linting, or type checks when available and
    permitted.
7.  Review the diff for secrets, generated files, unrelated edits, and
    data leakage.
8.  Summarize changed files, commands executed, actual test results, and
    remaining limitations.

Do not delete files, reset Git state, overwrite user work, commit, push,
or perform destructive operations without explicit authorization. If
file modification, terminal, or Git access requires approval, request it
and wait. Do not claim to have modified files or run commands when
permission was denied.

## 11. Implementation milestones

Work in small, reviewable milestones. Do not implement all milestones in
one request.

1.  Inspect repository and create a coverage-inspection script for
    StatsBomb Open Data.
2.  Build validated models and tests for competition and match JSON.
3.  Import selected matches idempotently and report actual
    competition/season coverage.
4.  Add lineup ingestion and validate player/team relationships.
5.  Add team form and Elo baseline features.
6.  Train and evaluate a chronological baseline model.
7.  Add lineup features and compare results.
8.  Add player-combination features and compare results.
9.  Persist reproducible predictions and expose them through an API.
10. Add a simple dashboard and scheduled refresh only after the pipeline
    is reliable.

At each milestone, state what is implemented, what was tested, what data
is actually available, and what remains incomplete.
