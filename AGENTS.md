# AGENTS.md --- Football Match Prediction

## 1. Project purpose

Build a maintainable football match prediction application that
estimates the probabilities of:

-   Home win
-   Draw
-   Away win

The final service is planned to support two prediction modes:

-   `PRE_MATCH`: use team strength, historical results, and recent form
    available by the day before kickoff.
-   `POST_LINEUP`: after official starting lineups are announced, update
    the prediction using available player records and player-combination
    features.

The immediate priority is to complete and evaluate a reliable
`PRE_MATCH` baseline before starting lineup-based modeling. The goal is
a usable prediction service, not merely a one-off notebook. Predictions
are estimates that require validation; they do not guarantee match
outcomes.

The initial target competitions are the European top five domestic
leagues:

-   Premier League (England)
-   La Liga (Spain)
-   Bundesliga (Germany)
-   Serie A (Italy)
-   Ligue 1 (France)

**Important data limitation:** StatsBomb Open Data covers only selected
competitions and seasons. Never claim that it supplies complete
historical coverage for all five leagues. Coverage and match-availability
inspection are completed exploration work; preserve the existing
inspection tools and their findings where available, and represent
unavailable coverage explicitly. This exploration does not mean that
historical results have been imported or that a model has been built.
Keep data-source integrations replaceable so a broader-coverage provider
can be added later without rewriting the prediction domain.

## 2. Data-source policy

### Initial model input: authorized local match-results CSV

Build the initial model from local CSV match-results data whose
permission for the intended use has been confirmed. Public
accessibility alone does not establish permission for machine-learning
training or redistribution. If an authorized local file is not
available, do not invent, synthesize, or download match results to fill
the gap; report that model development is blocked on suitable data.
Normalize and validate CSV input before feature generation, preserving
missing values rather than silently treating them as zero.

### Supplementary source: StatsBomb Open Data

Use the official public dataset at
https://github.com/statsbomb/open-data as a possible future source for
lineup and player-combination analysis. Its JSON data includes selected
competitions and seasons only. Inspect actual metadata and match
availability before use; never assume it covers every season or all
five leagues. Do not invent fixtures, lineups, player records, or
statistics to fill gaps. Preserve source identifiers and attribution
information.

Defer external APIs such as API-Football and automated latest-season
collection to a later milestone. Before using any dataset or provider,
verify the applicable license/terms and permitted uses, attribution and
redistribution requirements, competition/season coverage, and
collection date. Record the source, source identifiers, license or
terms reference, coverage, and retrieval/collection timestamp for each
dataset or imported record as applicable. Implement each future provider
as a separate adapter, keep provider IDs separate from internal IDs,
and never scrape or bypass access controls.

### Data quality requirements

-   Store the provider/source name and source identifier for imported
    records.
-   Track each data source's license/terms, provenance, actual coverage,
    and collection or retrieval time; do not imply rights that have not
    been verified.
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
│   ├── inspect_match_availability.py
│   ├── train_model.py
│   └── predict_match.py
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
terms. Adapt this suggested layout to the actual repository; do not
assume these paths or files already exist.

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
-   `predictions`: match, prediction timestamp/cutoff, prediction mode,
    model version, H/D/A probabilities, feature snapshot reference.
-   `model_artifacts` or equivalent: model version, training/validation
    period, source-data reference, feature definition, and evaluation
    metrics needed to identify and reproduce the model state.

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

For the first `PRE_MATCH` model, prioritize features in this order:
team Elo rating; each team's previous-five-match form and goals for and
against; and home advantage. Any additional form or schedule feature
must use only reliable data available at the prediction cutoff.

Do not use the actual starting XI as a `PRE_MATCH` input. Add lineup
availability/continuity, player appearances, shared player-pair
appearances and minutes, and reliable formation/position features only
for the later `POST_LINEUP` mode, and only when those facts were
available after lineup announcement but before kickoff.

Do not immediately generate every possible player pair or use player
names as arbitrary numeric features. Start with team-level features,
establish a baseline, then add lineup features and player-combination
features as separate experiments.

For player-combination statistics, require a minimum sample size or use
smoothing/shrinkage; account for minutes played rather than treating
every appearance equally; and do not interpret raw pair win rates as
causal effects. Opponent strength, player quality, tactical role, and
match context can confound these statistics. Compare them with the
`PRE_MATCH` baseline on the same chronological test period before
claiming value.

## 7. Prediction and evaluation rules

The target is three-class classification: home win (H), draw (D), and
away win (A). Build the first `PRE_MATCH` model in this order:

1. Read, validate, and normalize authorized local match-results CSV
   input.
2. Calculate time-updated team Elo ratings.
3. Generate previous-five-match form and goals-for/goals-against
   features for each team.
4. Include home advantage.
5. Train a multinomial Logistic Regression model to predict H/D/A
   probabilities.
6. Train and validate on chronologically separated data.
7. Persist predictions and sufficient model state/metadata to identify
   and reproduce the model.

Elo is a team-strength feature; do not assume Elo alone automatically
produces a draw probability. Consider more complex models only after
this baseline and its evaluation are reliable.

### Leakage prevention

-   For each prediction, use only information available at its
    explicitly recorded cutoff. `PRE_MATCH` uses information available
    by the day before kickoff and must not include the actual starting
    XI. `POST_LINEUP` may use confirmed lineup and player information
    available after announcement and before kickoff, but no
    post-match-only information.
-   Never use a match's result, final score, or post-match statistics to
    generate that match's input features. Update Elo and rolling form
    only after the result becomes available, for later predictions.
-   Do not let results from another match on the same date leak into a
    prediction merely because an implementation processes that match
    first. Unless kickoff and cutoff timestamps establish that the
    result was already available, calculate same-date match features
    without any same-date results.
-   Split training and validation chronologically, never randomly.
    Fit preprocessing and any probability calibration using training
    data (and validation data only as appropriate); do not fit or tune
    on the final test set.
-   When using multiple seasons, preserve their actual chronological
    order. Document how teams entering through promotion or leaving
    through relegation are initialized, carried forward, or otherwise
    handled; do not invent prior results for them.
-   Store the prediction cutoff/timestamp, mode, model version, and
    feature/data snapshot or equivalent reproducibility metadata.

### Evaluation and probabilities

Report Accuracy, multiclass Log Loss, and multiclass Brier Score, and
compare against a simple class-frequency baseline on the same
chronological evaluation period. State the evaluation dates and number
of matches. Where useful, inspect and report probability calibration.
Document the multiclass Brier Score convention used. Do not overstate
performance when data is limited, and never claim unmeasured
improvements. All modes must return H, D, and A probabilities, each in
`[0, 1]`, summing to `1` within the documented numerical tolerance. If
data is missing or out of distribution, surface that limitation rather
than presenting unjustified certainty. These are estimates that require
validation, not guarantees of actual outcomes.

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

-   Test CSV normalization, required fields, malformed rows, missing
    optional values, and outcome labels.
-   Test Elo and previous-five-match features, including that a result
    cannot affect its own or an earlier prediction's features and that
    same-date results do not leak through processing order.
-   Test chronological train/validation splitting and H/D/A probability
    normalization.
-   Test the class-frequency baseline and evaluation metric
    calculations.
-   Test provider JSON parsing, schema validation, missing optional
    fields, and idempotent ingestion when those integrations are
    implemented.
-   Test player-combination features with small samples and missing
    appearances when that later milestone is implemented.
-   Add integration tests for database writes and migrations when a test
    database is available.
-   Use saved, anonymized or appropriately licensed fixtures; unit tests
    must not depend on a live external API or unauthorized dataset.

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

The StatsBomb coverage and match-availability inspection is completed
exploration work. Preserve the existing inspection tools and document
their actual findings; this does not imply complete five-league
coverage, match ingestion, or model completion.

Proceed in this priority order; do not skip ahead while the
`PRE_MATCH` baseline remains unevaluated:

1.  **M1.** Check public dataset terms and confirm permissions for
    intended use; record source, license/terms, attribution,
    competition/season coverage, and collection time. Do not treat
    public visibility as permission to train or redistribute.
2.  **M2.** Normalize authorized local match-results CSV input; if no
    eligible file exists, report the blocker without fabricating or
    downloading data.
3.  **M3.** Generate Elo and previous-five-match form and goals
    features, including home/away context where supported, with
    prediction cutoffs and leakage controls.
4.  **M4.** Train and chronologically validate the `PRE_MATCH` H/D/A
    model; compare it with the class-frequency baseline and report
    Accuracy, Log Loss, Multiclass Brier Score, evaluation period, and
    match count.
5.  **M5.** Add a CLI for predicting a specified match and persist the
    prediction plus identifiable model state and reproducibility
    metadata.
6.  **M6.** Integrate historical starting-XI data with verified
    permissions, actual coverage, and pre-kickoff availability.
7.  **M7.** Add and evaluate `POST_LINEUP` player and player-combination
    features against the same `PRE_MATCH` test window.
8.  **M8.** Expose predictions through an API and user interface.
9.  **M9.** Integrate a latest-season API source and automated updates
    only after checking current coverage, documentation, terms, plan
    limits, and rate limits.

At each milestone, state what is implemented, what was tested, what
data and permissions are actually available, the evaluation period and
sample size where applicable, and what remains incomplete.
