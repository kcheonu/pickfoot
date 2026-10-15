# PickFoot

## Local Docker backend + Vercel frontend

The current deployment target is a Next.js frontend hosted in a **new
Vercel Hobby project** and a local Docker backend on the administrator's
Windows PC:

- **Frontend:** Next.js 15, React, TypeScript, and Tailwind CSS in `web/`.
- **Backend:** FastAPI and a separate `pickfoot-worker` container.
- **Database:** PostgreSQL in Docker with the `pickfoot-postgres` named
  volume. Its port is not published to Windows or the internet.
- **Models and runtime files:** the host `models/` and
  `data/persistent/` directories are bind-mounted so model versions,
  uploaded data and state survive container replacement.
- **Public HTTPS:** Cloudflare Tunnel targets only
  `http://127.0.0.1:8000`; no router port-forwarding is required.
- **Browser API path:** browsers call same-origin `/api/v1/...` URLs on
  Vercel. A Next.js external rewrite forwards them to the server-side
  `PICKFOOT_API_ORIGIN` (the Tunnel HTTPS origin in production). The
  Tunnel URL is not exposed as a `NEXT_PUBLIC_*` variable.

Long-running training stays in the local worker, never in a Vercel
Function. The existing Streamlit application, Elo/model modules,
2025/26 CSV, and five model artifacts remain in the repository. The
Docker/Vercel/Tunnel architecture is **not connected or publicly
deployed**; no production URL, remote login, or internet E2E success is
claimed. The Vercel Hobby plan has eligibility and usage limits; check
Vercel's current plan terms and confirm that the project's use qualifies
before creating the separate project.

The same-origin proxy is intentional: direct browser-to-Tunnel calls
would require cross-origin credential/CORS configuration and expose the
Tunnel origin in frontend settings. The proxy lets the browser use a
same-site HttpOnly cookie while FastAPI still validates the exact
production `Origin` for administrator writes.

### Windows local backend setup

Use Python 3.12+, Node.js 20+, Docker Desktop with Compose, and PowerShell.
Start Docker Desktop and wait until its Linux engine reports ready before
running Compose.
Create a local `.env` from `.env.example`, replace every placeholder, and
keep real credentials out of Git. Generate an administrator password hash with
`python -m pickfoot.api.auth` (minimum 14 characters) and a session secret
with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
Set `COOKIE_SECURE=true` and `ENVIRONMENT=production` before publishing
the API through a fixed HTTPS Tunnel; configure `WEB_ORIGIN` to the exact
production Vercel origin. The development defaults are only for local
work.

Start Docker Desktop, then start PostgreSQL, apply the Alembic schema, and
start the API and worker:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f api worker
```

The API is published only on `127.0.0.1:8000`; PostgreSQL has no host
port mapping. Readiness is available at
`http://127.0.0.1:8000/api/v1/health/ready`. The database migration runs
before the API and worker start. Docker named volume
`pickfoot-postgres` persists the database; `models/` and
`data/persistent/` are host-backed, persistent directories.
To apply migrations without starting application services, start the
database and run `docker compose run --rm migrate`.

For local web development, set `web/.env.local` from `web/.env.example`
(`PICKFOOT_API_ORIGIN=http://127.0.0.1:8000`) and run:

```powershell
Set-Location web
npm.cmd install
npm.cmd run dev
```

Visit `http://localhost:3000`. Next.js rewrites the browser's same-origin
API calls to the local FastAPI origin.

To stop/restart or inspect services:

```powershell
docker compose stop
docker compose start
docker compose down
docker compose logs --tail 200 api worker db
```

`docker compose down` retains the PostgreSQL named volume. Do not add
`--volumes`/`-v` unless intentionally deleting the database.

### Import existing local SQLite records into PostgreSQL

The existing SQLite database is retained unchanged at
`data/runtime/pickfoot.sqlite3`. Importing its result uploads, source
provenance, registered teams, aliases, corrections, and prediction
records is a separate, explicit operation; starting Compose does not
import it. The importer reads SQLite in read-only mode and writes all
records in one PostgreSQL transaction. Stable IDs and conflict checks
make an identical import safe to repeat. A score conflict or malformed
legacy record aborts the transaction. The CSV file bytes themselves were
not retained by the legacy database; source filename, SHA-256, permission
reference, collection/upload time and counts are preserved in audit
records.

Back up the SQLite file first, confirm Docker is connected to the
intended local PostgreSQL volume, and then explicitly run:

```powershell
Copy-Item .\data\runtime\pickfoot.sqlite3 .\data\runtime\pickfoot.sqlite3.bak
docker compose up -d db
docker compose --profile migration run --rm migrate-sqlite `
  python -m pickfoot.api.migrate_sqlite `
  --source /app/data/runtime/pickfoot.sqlite3 --confirm-import
```

The importer does not change the 2025/26 CSV or model artifacts.
PostgreSQL migrations run before this one-off job. Check its JSON count
summary and API pages before relying on imported records. Do not run it
against a production or unrelated database.

### Cloudflare Tunnel (manual setup)

No Tunnel has been created or started. Install `cloudflared` using the
[official Cloudflare Windows instructions](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/)
and authenticate to the Cloudflare account that controls a domain. A
stable hostname needs a Cloudflare-managed zone and DNS access. Keep the
generated credentials under `%USERPROFILE%\.cloudflared`; never copy them
into this repository.

After Docker reports the API healthy, create a named Tunnel and DNS route
using the official Cloudflare CLI (substitute the real API hostname):

```powershell
cloudflared tunnel login
cloudflared tunnel create pickfoot-local
cloudflared tunnel route dns pickfoot-local api.<your-domain>
```

Configure its ingress to serve only the local API:

```yaml
tunnel: <TUNNEL-UUID>
credentials-file: C:\Users\<Windows-user>\.cloudflared\<TUNNEL-UUID>.json
ingress:
  - hostname: api.<your-domain>
    service: http://127.0.0.1:8000
  - service: http_status:404
```

Run `cloudflared tunnel run pickfoot-local` in a separate PowerShell
window; stop it with `Ctrl+C`. Do not expose PostgreSQL. A temporary
Quick Tunnel can be started with
`cloudflared tunnel --url http://127.0.0.1:8000`; its generated URL
changes and is for short-lived testing only. Update
`PICKFOOT_API_ORIGIN` and redeploy the frontend if that test URL changes.
It is not a stable production address. If the Tunnel/PC is offline the
frontend remains hosted but shows that the backend is unavailable.
Administrator writes still require FastAPI authentication and HTTPS
cookies; keep the backend host and Cloudflare account secured.

### New Vercel Hobby project

Create a **separate new** Vercel project connected to this repository;
do not import, link, modify, or overwrite `mksplayground` or any other
existing project. Set Root Directory to `web/`, allow the standard
Next.js build, and set the non-public environment variable
`PICKFOOT_API_ORIGIN=https://api.<your-domain>` to the fixed Tunnel
origin. Vercel applies this origin when building the rewrite
configuration, so redeploy after changing it. Never use
`NEXT_PUBLIC_PICKFOOT_API_URL` or put credentials in frontend variables.

Set backend `.env` values `WEB_ORIGIN=https://<your-pickfoot-vercel-domain>`,
`COOKIE_SECURE=true`, and `ENVIRONMENT=production` before public use.
The FastAPI origin check accepts only the configured web origin. Vercel
preview domains are not automatically trusted; use a deliberate separate
preview configuration rather than a wildcard production origin.
Vercel-hosted pages proxy their API requests through Next.js to the
Tunnel; browser CORS access to the Tunnel is not required. Vercel Hobby
is subject to current plan terms/quotas, and the project owner must
verify its eligibility for the intended use.

To remove a deployment later, stop the Tunnel and the local stack
(`docker compose down`), then disable/remove only the new PickFoot
project in its own Vercel dashboard after confirming its identity.

### PostgreSQL and model backups

Use container-side `pg_dump` plus `docker compose cp` so PowerShell does
not recode a binary dump:

```powershell
New-Item -ItemType Directory -Force .\backups
docker compose exec -T db sh -c "pg_dump -U pickfoot -d pickfoot -Fc -f /tmp/pickfoot.dump"
docker compose cp db:/tmp/pickfoot.dump .\backups\pickfoot.dump
docker compose exec -T db rm -f /tmp/pickfoot.dump
Copy-Item -Recurse .\models .\backups\models
Copy-Item -Recurse .\data\persistent .\backups\data-persistent
```

Restore only into a deliberately selected, compatible database. A
`pg_restore --clean` operation can replace tables; never use it on the
only copy of important data. For a confirmed restore target, stop API and
worker writes, then run:

```powershell
docker compose stop api worker
docker compose cp .\backups\pickfoot.dump db:/tmp/pickfoot.dump
docker compose exec -T db pg_restore -U pickfoot -d pickfoot --clean --if-exists /tmp/pickfoot.dump
docker compose exec -T db rm -f /tmp/pickfoot.dump
docker compose start api worker
```

Restore the matching model and persistent data directories from the same
backup point. Keep backups encrypted and outside Git. Test recovery on a
disposable database before depending on the backup procedure.

### Troubleshooting and recovery

- `docker compose ps` and `docker compose logs --tail 200 db migrate api worker`
  identify container start/migration/worker errors.
- If API readiness fails, check PostgreSQL health, `.env` values and
  host directories `models/` and `data/persistent/`; the existing
  2025/26 `models/*.joblib` are read from the host mount.
- If Vercel reports an API connection problem, test the local readiness
  endpoint and Tunnel hostname, confirm `PICKFOOT_API_ORIGIN` in the new
  Vercel project's production environment, then redeploy.
- After schema updates, `docker compose up -d --build` runs the migration
  service before API/worker startup. On failed upgrades, preserve the
  DB volume and logs and restore from a verified backup only after
  confirming the target and recovery point.
- `docker compose down` followed by `docker compose up -d` restarts the
  stack without deleting the named database volume.

### Current service limits

- Managerial login is a single environment-configured account. No public
  user accounts or password reset flow exist.
- The login uses a signed HttpOnly cookie, PBKDF2 password hash, Origin
  checks on state-changing requests, and an in-process IP request limit.
  Public prediction/model-comparison requests are also rate-limited.
  Set secure cookies only behind HTTPS.
- Schedules and confirmed results are persisted in PostgreSQL; results
  are audited and updates use optimistic versions. Elo is rebuilt from
  the authorized base dataset plus completed stored matches. Model
  coefficients are not changed until a separately queued candidate is
  reviewed and promoted.
- Base datasets are uploaded after the administrator records source and
  permission information. The administrator's rights confirmation is an
  attestation, not an independent legal verification.
- No live service credentials, domain/DNS access, Tunnel, Vercel project
  approval, or public URL were provided. Actual internet deployment and
  hosted end-to-end verification therefore remain outstanding.

### Development build checks

```powershell
python -m pytest -q
Set-Location web
npm.cmd run lint
npm.cmd run build
```

`npm.cmd` is used because PowerShell may block the `npm.ps1` shim.

The retained Streamlit app provides a local, `PRE_MATCH` H/D/A football
prediction workflow for the Premier League, La Liga, Bundesliga, Serie A,
and Ligue 1. The deployable web/API path also uses administrator-supplied
match-result CSVs and does not fetch fixtures or results from an external
API.

Predictions are model estimates, not guaranteed outcomes. A model trained
on synthetic test fixtures is never shown as a real-match model.
`POST_LINEUP` is displayed as a future capability and is not implemented
for prediction.

## Data rights and current availability

The supplied 2025/26 results CSV is from OpenFootball's
[football.json repository](https://github.com/openfootball/football.json).
The repository README says that the football.json schema, data, and
scripts are dedicated to the public domain and may be used without
restrictions (checked 2026-10-14). A standalone `LICENSE` file was not
present at the checked URL; the license statement is in the repository
README. The project's data-source record is in
`data/processed/source_provenance.json`. The CSV does not include its
acquisition timestamp, so the actual source collection time is recorded
as unknown rather than inferred from local file timestamps. PickFoot
keeps source attribution even where the stated terms do not require it.

StatsBomb Open Data is not the initial model input. Its coverage is
limited to selected competitions and seasons, and it may be considered
later for lineup/player research only after checking its current terms
and actual coverage. No external data is downloaded by PickFoot.

The original local CSV at `data/raw/matches.csv` is not modified.
`data/processed/matches_normalized.csv` is a separately generated,
git-ignored normalized copy. Required input headers are:

```csv
league,season,match_date,home_team,away_team,home_goals,away_goals,result
```

Accepted league names include `Premier League`/`EPL`, `La Liga`,
`Bundesliga`, `Serie A`, and `Ligue 1` (and common country-qualified
aliases). Dates should be valid date values, preferably `YYYY-MM-DD`.
Season values may be a year such as `2023`, `2023/24`, `2023-24`, or
`2023-2024`; adjacent-year seasons normalize to `2023-24`. Goals must be
non-negative whole numbers. `result` must be `H`, `D`, or `A` and agree
with the goals. Each match row is a completed match. Do not include
post-match/current-match statistics as predictors.

League codes `PL`, `PD`, `BL1`, `SA`, and `FL1` normalize respectively
to Premier League, La Liga, Bundesliga, Serie A, and Ligue 1. The
verified file contains 380, 380, 306, 380, and 305 scored matches in
those leagues (1,751 total). The source preparation notes document the
Nantes–Toulouse Ligue 1 fixture as having no final score; it is excluded,
not fabricated. Match dates span 2025-08-15 through 2026-05-24. The
normalized file retains only completed scored fixtures; the omitted
unscored fixture is not a prediction target.

For future datasets, use only real match results for which you have
verified permission; if no such file exists, the UI will explain that no
real model is available.
Small synthetic rows are constructed only inside tests to verify code
behavior and are never shipped as training data.

## Windows setup (PowerShell)

Python 3.12 or newer is required. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If PowerShell blocks environment activation, run the venv interpreter
directly, for example
`.venv\Scripts\python.exe -m pip install -e ".[dev]"`.

## Train a model

Training is deliberately opt-in and requires an explicit local
confirmation of the data-use rights and source record. The command
trains one independent model per league represented in the CSV:

```powershell
python -m pickfoot.cli train `
  --csv data/raw/matches.csv `
  --artifact-dir models `
  --source-name "OpenFootball" `
  --license-reference "football.json README: public domain; checked 2026-10-14" `
  --source-url "https://github.com/openfootball/football.json" `
  --confirm-usage-rights
```

If known, `--source-collected-at` records when you acquired the local
data file as an ISO-8601 UTC timestamp. Omit it when the timestamp is
unavailable; the artifact explicitly records that it is unknown.
The rights confirmation is your attestation; the app cannot independently
verify a license or private permission record.

Training requires enough chronological history, at least three outcome
classes in the training portion, and at least one later validation date.
The command reports an actionable error rather than fabricating results
if the input is absent or insufficient. Evaluation uses an 80/20
chronological split by distinct match date (with no date split across
train and validation), Accuracy, multiclass Log Loss, multiclass Brier
Score (mean sum of the three squared one-hot probability errors), and a
training-class-frequency baseline. Elo is initialized to 1500 for new
teams, includes a fixed home advantage, and carries forward across
season boundaries without regression. For teams with fewer than five
prior matches, form uses only the available prior matches and records
that sample count; it does not fabricate missing matches. Elo and recent
form update only after all fixtures for a date have had their features
computed. Holdout metrics are computed with a pipeline whose scaler is
fit only on the earlier training dates; validation dates are transform-
only. After holdout evaluation is frozen, the deployable pipeline is
refit on the full authorized season so the future-match model can use all
available 2025/26 results. The held-out score still reflects predictions
made before those validation results were used for the final refit.

### OpenFootball 2025/26 training run

The supplied file validated as 1,751 scored matches from 2025-08-15
through 2026-05-24. League counts were PL 380, PD 380, BL1 306, SA 380,
and FL1 305. The 305 Ligue 1 rows are intentional: the local source
preparation record reports one fixture without a final score, which is
excluded rather than inferred. The raw CSV is untouched; the normalized
copy is `data/processed/matches_normalized.csv` and is git-ignored.

Models trained on 2026-10-14 with version
`pickfoot-pre-match-logreg-1`. Values below are holdout metrics followed
by the chronological training-class-frequency baseline in parentheses.
Positive accuracy delta and negative Log Loss/Brier delta favor the
model.

| League | Total | Train / validation | Train dates | Validation dates | Accuracy (baseline) | Log Loss (baseline) | Brier (baseline) |
|---|---:|---:|---|---|---:|---:|---:|
| Premier League | 380 | 314 / 66 | 2025-08-15–2026-04-11 | 2026-04-12–2026-05-24 | 0.561 (0.470) | 0.994 (1.069) | 0.597 (0.646) |
| La Liga | 380 | 290 / 90 | 2025-08-15–2026-03-22 | 2026-04-03–2026-05-24 | 0.456 (0.511) | 1.079 (1.032) | 0.643 (0.620) |
| Bundesliga | 306 | 234 / 72 | 2025-08-22–2026-03-15 | 2026-03-20–2026-05-16 | 0.333 (0.389) | 1.162 (1.092) | 0.701 (0.663) |
| Serie A | 380 | 300 / 80 | 2025-08-23–2026-03-22 | 2026-04-04–2026-05-24 | 0.488 (0.363) | 1.083 (1.087) | 0.636 (0.660) |
| Ligue 1 | 305 | 237 / 68 | 2025-08-15–2026-03-21 | 2026-03-22–2026-05-17 | 0.456 (0.382) | 1.027 (1.120) | 0.628 (0.681) |

These are one-season holdout results, not a guarantee of future
performance. For Bundesliga and La Liga, every reported model metric is
worse than the corresponding baseline; no general model improvement is
claimed. Artifacts are stored locally under `models/` (one joblib file
per league), which is git-ignored.

`--synthetic-test-data` is available only to label an intentionally
synthetic/test artifact (for example, in an isolated test directory).
The dashboard displays a prominent
synthetic-test warning and calls its output a demo, never a real match
forecast. Do not use this flag for authorized real match data.

## Elo, result uploads, and prediction evaluation

The five trained model artifacts remain under `models/` and are not
overwritten by result uploads. Each artifact contains the fitted
scaler/classifier, holdout report, 2025/26 `LeagueState`, and its model
training cutoff. The dashboard rebuilds a separate current team state by
replaying the immutable base CSV plus accepted 2026/27 results. It passes
that state to the existing model pipeline without fitting or changing
the scaler or Logistic Regression coefficients.

Elo uses initial rating `1500`, `K=20`, and `+60` Elo home advantage,
with divisor `400`. For a match:

```text
expected_home_score = 1 / (1 + 10 ** ((away_elo - (home_elo + 60)) / 400))
home_actual_score   = 1 for a home win, 0.5 for a draw, 0 for an away win
home_elo_change     = 20 * (home_actual_score - expected_home_score)
away_elo_change     = -home_elo_change
```

An unexpected win by the lower-rated side creates a larger rating change
because actual score is farther from the pre-match expected score. Elo
expected score is a scalar expected points value, **not** H/D/A outcome
probabilities. All fixtures on the same calendar date use the
start-of-day ratings and recent form; their results are batch-applied
after that date's features are generated. Season transitions carry Elo
and recent form forward without mean reversion. No schedule/results are
invented for promoted teams.

The web app includes `Elo 순위`, `팀 상세 분석`, and `경기별 Elo 분석`
pages. Tied ratings share the competition rank; tied teams are displayed
alphabetically. Once 2026/27 results have been added, the active roster
is the set of teams in those results plus manually confirmed registered
teams. Until all current participants have appeared in a result file,
register teams that have not yet played through `경기 결과 업로드`.
Registered teams start at Elo 1500 with no fabricated match/form history.
Their Elo is available for analysis, but the unchanged PRE_MATCH model
rejects teams absent from its 2025/26 training roster. Relegated teams
not in the 2026/27 active roster are not shown in the current-season
ranking.

### Upload and correction flow

Upload only completed 2026/27 results using the required CSV schema
above. UTF-8 and UTF-8 BOM are accepted. The app normalizes supported
league codes, season, date, whitespace, and registered aliases, then
validates scores, H/D/A labels, self-fixtures, duplicate rows, date
range, and conflicts. File selection only previews; the user must
confirm source/permission and press `검증된 2026/27 결과 반영` to save.

The original `data/raw/matches.csv` is never edited. Accepted results,
source/license reference, SHA-256, and upload UTC time are stored in the
git-ignored local `data/runtime/pickfoot.sqlite3`. Re-uploading the same
file or an identical fixture does not apply it twice. A conflicting score
is never silently replaced. A separate correction action is offered
only for an already stored 2026/27 result after reviewing both values
and confirming the correction; the prior and corrected values, source,
permission reference, file hash, and timestamp are audited. Conflicts
against the immutable base season cannot be corrected from the upload
page. SQLite transactions preserve the previous state on a failed
write.

Team-name differences are not guessed. Register a confirmed current
team, then create an explicit league-scoped alias (for example,
`Arsenal` → `Arsenal FC`) before uploading a file that uses the alias.
The canonical team identity is maintained across seasons.

The Elo update changes only current team-state features:
home/away Elo, Elo difference, and the same recent-five points, win
rate, goals-for/goals-against, and goal-difference features used by the
trained model. It does **not** retrain the model. The prediction page
shows model-training cutoff, current team-state cutoff, upload status,
model validation metrics, and the actual feature snapshot. The model
training cutoff remains the last date in the original 2025/26 training
season.

### Prediction records and evaluation

Successful `PRE_MATCH` button predictions are persisted in the local
SQLite database with a UUID, fixture identity, predicted-at UTC time,
model version/training cutoff, team-state cutoff and feature-state hash,
three probabilities, and evaluation status. A repeated prediction for
the same fixture/model/state reuses the original record instead of
adding a duplicate. Data acquisition cutoff is stored as unknown when
the source does not provide a timestamp. When an exact result row is
uploaded, it is linked to matching predictions, but the status remains
`unverified_kickoff_time`: the source CSV has a calendar date and no
kickoff timestamp, so a timestamped prediction alone cannot prove it was
made before kickoff. Such rows are excluded from verified pre-match
Accuracy, Log Loss, and Brier metrics.

Historical time-safe reconstruction is separate from recorded
predictions. Generate or refresh its report with:

```powershell
python -m pickfoot.cli backtest `
  --csv data/raw/matches.csv `
  --database data/runtime/pickfoot.sqlite3 `
  --output data/runtime/walk_forward_backtest.json `
  --confirm-usage-rights
```

The expanding-window evaluator makes predictions only when at least 30
earlier matches and all H/D/A classes are present. For each date, it
fits a fresh scaler and Logistic Regression only on strictly earlier
matches; all fixtures on the same date share that training window and
start-of-day features. It reports Accuracy, multiclass Log Loss, and
multiclass Brier Score alongside an expanding training-frequency
baseline. The report is bound to a SHA-256 fingerprint of its input
matches; the dashboard refuses to show it if the current data differs.
This reconstructed evaluation is not evidence that a real prediction
was actually made before a match. It never modifies the five deployed
model artifacts.

The completed 2025/26 expanding-window run evaluated 1,597 matches after
the minimum-history warm-up. Accuracy / Log Loss / Brier are shown below
as model versus time-varying training-frequency baseline:

| League | Matches | Evaluation period | Accuracy | Log Loss | Brier |
|---|---:|---|---:|---:|---:|
| Premier League | 350 | 2025-09-13–2026-05-24 | 0.437 / 0.420 | 1.146 / 1.091 | 0.685 / 0.661 |
| La Liga | 349 | 2025-09-12–2026-05-24 | 0.487 / 0.493 | 1.084 / 1.052 | 0.641 / 0.633 |
| Bundesliga | 273 | 2025-09-21–2026-05-16 | 0.469 / 0.432 | 1.101 / 1.086 | 0.640 / 0.657 |
| Serie A | 350 | 2025-09-19–2026-05-24 | 0.451 / 0.386 | 1.156 / 1.092 | 0.672 / 0.663 |
| Ligue 1 | 275 | 2025-09-14–2026-05-17 | 0.476 / 0.440 | 1.116 / 1.098 | 0.666 / 0.663 |

This walk-forward result is distinct from the original single 80/20
holdout metrics above. It shows mixed results versus the baseline; it
does not establish a reliable improvement across leagues. The generated
per-match report is git-ignored at
`data/runtime/walk_forward_backtest.json`.

The upload database and generated backtest report are local runtime data
under `data/runtime/` and are excluded from Git. For isolated checks set
`PICKFOOT_DATA_DIR` to a temporary directory; tests use synthetic rows
only in temporary locations and never modify the 2025/26 CSV or
production joblib files.

## Run the legacy local Streamlit app

```powershell
python -m streamlit run src/pickfoot/app.py --server.address 127.0.0.1
```

Open the localhost URL printed by Streamlit (normally
`http://localhost:8501`). Choose an available league, home team, away
team, and a date later than the model's data cutoff, then run the
`PRE_MATCH` estimate. Teams are restricted to that league's trained
roster. Earlier dates and unseen teams are rejected; the latest Elo/form
state is not silently applied to historical dates.

The dashboard shows H/D/A estimated probabilities, an outcome chart,
team Elo and recent-form comparisons, model version, validation
metrics, source attribution, season, and latest training-data date. It
warns that later 2026/27 results and promoted/unknown teams are not
represented in a 2025/26-only model. The synthetic test suite
checks the training and prediction path but does not establish real-world
predictive performance.

Only open local `.joblib` model files that you created with PickFoot;
joblib artifacts are trusted Python serialization files and must not be
loaded from unknown sources.

## Tests

```powershell
python -m pytest
```

Tests use small in-memory or temporary synthetic fixtures only; they do
not call a live service, download data, or claim real-match model
performance.

## Retained local MVP and limitations

-   Implemented target: per-league `PRE_MATCH` Logistic Regression with
    date-safe Elo and recent-form features.
-   The supplied OpenFootball-derived results are documented by the
    repository as public domain. The local source collection timestamp
    was not recorded, and the CSV contains no metadata from which to
    recover it.
-   `POST_LINEUP` and external API ingestion are not implemented.
-   The original single-user Streamlit interface and its local SQLite
    result-management features remain as a separate local workflow. The
    optional `migrate-sqlite` Compose profile imports operational records
    only after an explicit `--confirm-import`; it reads the source
    database read-only and never changes or removes it.
-       The public service, production persistence, and production E2E checks are
    not complete until the separate Vercel project, Windows Docker stack,
    fixed HTTPS Tunnel, persistent volumes/directories, and production
    secrets are configured and tested.
