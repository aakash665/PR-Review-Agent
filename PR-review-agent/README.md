# GitHub PR Review Agent

An evidence-grounded pull request review service. It indexes repository code into Qdrant, retrieves code and project conventions for each changed symbol, combines that evidence with deterministic static analysis, and sends structured findings through independent review and verification stages before publishing a GitHub review.

This FastAPI service includes a lightweight web dashboard. It does not claim an AI review when an LLM is unavailable: the local fixture command still demonstrates parsing, chunking, vector indexing, retrieval, and static analysis without GitHub or OpenRouter credentials.

## Architecture

```mermaid
flowchart TD
    GH[GitHub App webhook] --> API[FastAPI signature validation]
    Browser[Review dashboard] --> API
    API --> DB[(SQLite durable jobs and findings)]
    DB --> Worker[Background worker]
    Worker --> GHAPI[GitHub REST API: PR, diff, files]
    GHAPI --> Diff[Changed-line and symbol analysis]
    Diff --> Index[AST / Tree-sitter chunking]
    Index --> Embed[OpenRouter embeddings with SQLite cache]
    Embed --> Q[(Qdrant cosine vectors)]
    Q --> Hybrid[Hybrid retrieval: vector, symbol, path, tests, docs]
    Diff --> Static[Ruff, Bandit, ESLint, local Semgrep rules]
    Hybrid --> Review[Tool-enabled structured review agent]
    Static --> Review
    Review --> Verify[Independent verification agent]
    Verify --> Dedup[Deduplication and confidence thresholds]
    Dedup --> Publish[GitHub inline comments and PR summary]
    Publish --> GH
```

## What is implemented

- HMAC-SHA256 webhook verification, bounded payloads, supported pull-request actions, and idempotent `(PR, head SHA)` job creation.
- Same-origin dashboard for registering App-accessible repositories, queueing PR reviews, and live-polling job state, summaries, findings, evidence, and metrics.
- Bearer-key protection for dashboard, repository management, review result, and metrics APIs; GitHub webhooks continue to authenticate with their independent HMAC signature.
- A separate durable SQLite job worker. Webhook requests return `202`; model and GitHub work happens outside the request.
- GitHub App JWT and installation-token authentication, authoritative PR/diff retrieval, archive-based initial indexing, and bounded incremental changed-file indexing.
- Python AST and Tree-sitter declaration parsing, parent/import-aware chunks, Markdown section chunking, generated/binary/lock-file exclusions, and content-addressed embeddings.
- Qdrant cosine vector search with repository/commit payload filters, plus lexical, symbol, path, documentation, and test signals. A bounded context builder prevents whole-repository prompts.
- Cached OpenRouter embeddings, configurable model/context/confidence settings, and explicit timeouts/retries.
- Ruff, Bandit, ESLint, and repository-independent local Semgrep rules where tools are installed. Repository configuration files are not loaded by these analyzers.
- Tool-enabled OpenRouter chat review and summary with Pydantic-validated JSON; a separate calibrated Decisions API verifier, changed-line validation, semantic finding deduplication, and configurable summary/inline/verification thresholds.
- GitHub review publishing with valid added-line comments and commit-scoped idempotency marker.
- SQLite persistence for repositories, PRs, jobs, findings, embedding cache, index snapshots, and stage metrics; `/health`, `/metrics`, `/repositories`, and `/reviews/{job_id}` endpoints.
- CLI indexing, GitHub PR review, offline fixture RAG, and labeled precision/recall evaluation.

## Requirements and local setup

Use Python 3.12+, a running Qdrant instance, and an OpenRouter API key for chat, hosted embeddings, and Decisions API verification.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` and the GitHub App settings in `.env`. Do not commit `.env` or the private key. Start Qdrant and the API/worker:

```powershell
docker compose up --build
```

The dashboard is at `http://localhost:8000`; interactive API docs are at `/docs`. Generate a long random bearer key (for example, `python -c "import secrets; print(secrets.token_urlsafe(32))"`), set it as `DASHBOARD_API_KEY` in `.env`, and enter the same key into the dashboard when prompted. SQLite data is written under `data/` locally and the named Docker volume in Compose. The worker shares the same database and processes queued jobs.

Useful local checks and the no-GitHub demo:

```powershell
pytest
ruff check app tests evaluation
python -m app.cli review-fixture fixtures/python_project
```

With no OpenRouter key the fixture command accurately reports that it ran repository indexing, vector retrieval, and available static analyzers only. Add `OPENROUTER_API_KEY` to exercise review, Decisions API verification, and summary generation. The fixture uses deterministic local feature-hash embeddings only for an offline demonstration; production indexing defaults to the configured OpenRouter embeddings endpoint and model.

## GitHub App setup

1. In GitHub, create a **GitHub App**. Enable **Pull requests: Read and write**, **Contents: Read**, and repository **Metadata: Read**. No organization-wide or administrator permissions are needed.
2. Subscribe to the `pull_request` webhook event. Configure a random webhook secret and set the same value as `GITHUB_WEBHOOK_SECRET`.
3. Generate and securely store the App's private key. Set `GITHUB_APP_ID` and `GITHUB_PRIVATE_KEY_PATH` to the App ID and key file path.
4. Install the App on a test repository (or select specific repositories during installation). Repository access is constrained by the installation token.
5. For local delivery, expose port 8000 with [ngrok](https://ngrok.com/) or another HTTPS tunnel, then set the App's webhook URL to `https://<your-tunnel>/webhooks/github`.
6. Set `DASHBOARD_API_KEY`, `OPENROUTER_API_KEY`, `LLM_MODEL`, `DECISIONS_MODEL`, `EMBEDDING_MODEL`, and `QDRANT_URL`; start the API and worker with `docker compose up --build`.
7. Open the dashboard, enter `DASHBOARD_API_KEY`, and connect a repository already installed on the GitHub App. The dashboard validates the installation before storing the repository.
8. Configure the App webhook URL and open or update a test PR. The webhook returns `202`; the worker reviews it and the dashboard reflects queued/running/completed state and findings. The dashboard also lets you queue a PR manually.

For Docker, put the PEM file at `secrets/github-app.pem` (the `secrets/` directory is mounted read-only into the containers). Do not put the key in an image, source control, or logs. Supply the remaining environment variables through the invoking shell or a local `.env`; Compose intentionally does not require a checked-in secrets file.

Review-trigger actions are `opened`, `synchronize`, and `reopened`. `closed` events update the persisted pull-request state and cancel queued or in-flight jobs; the worker rechecks PR state before publishing.

## CLI

The repository must be installed to the GitHub App. Dashboard registration validates the installation and stores the repository; the webhook registers/refreshes repository metadata when PR events arrive. CLI GitHub operations discover the installation for the repository.

```powershell
python -m app.cli index owner/repo
python -m app.cli index owner/repo --commit <sha-or-ref>
python -m app.cli review owner/repo --pr 42
python -m app.cli review-fixture fixtures/python_project
python -m app.cli evaluate
```

`review` runs the durable pipeline synchronously for a convenient local demonstration; webhook reviews use the background worker. `review-fixture` does not contact GitHub. `evaluate` runs the labeled Python, Java, and TypeScript fixtures and requires an LLM key; it reports precision, recall, F1, false-positive rate, exact line accuracy, and retrieval hit rate. The aggregate and per-fixture result are printed and written to `evaluation/reports/latest.json`.

## Configuration

See [.env.example](./.env.example). Important settings:

| Variable | Purpose |
| --- | --- |
| `DATABASE_PATH` | SQLite database file; defaults to `/tmp/reviews.sqlite3` on Vercel and `data/reviews.sqlite3` elsewhere |
| `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY_PATH` | GitHub App authentication |
| `GITHUB_WEBHOOK_SECRET` | Webhook HMAC verification |
| `DASHBOARD_API_KEY` | Bearer key required for dashboard data and management APIs |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | OpenRouter authentication and API origin |
| `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME` | Optional OpenRouter app attribution headers |
| `LLM_MODEL` | OpenRouter chat model used for review and summary generation |
| `DECISIONS_MODEL`, `VERIFICATION_THRESHOLD` | Typed yes/no verifier model and minimum acceptance probability |
| `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` | OpenRouter embedding model and Qdrant vector size |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Shared vector collection; repository and commit are payload filters |
| `MAX_CONTEXT_TOKENS`, `TOP_K` | Context budget and retrieval count |
| `CONFIDENCE_THRESHOLD` | Minimum verified confidence for inline comments |
| `SUMMARY_CONFIDENCE_THRESHOLD` | Minimum verified confidence to retain in the summary |
| `INDEX_SNAPSHOT_RETENTION` | Number of indexed commit snapshots retained per repository |
| `MAX_RETRIES` | Total job attempts, including the first attempt |

The review and summary use OpenRouter's chat-completions endpoint; verification uses `/api/alpha/decisions` with a typed `noul` question for each finding batch. The Decisions model returns calibrated probabilities, not prose, so it does not replace the chat model that writes explanations and fixes. Chat and embeddings are routed through OpenRouter and use the same `OPENROUTER_API_KEY`; model names such as `openai/gpt-4o-mini` or `openai/text-embedding-3-small` are OpenRouter model IDs, not direct OpenAI API calls or credentials.

Set `COST_PER_MILLION_TOKENS` to a blended per-million rate for your selected chat and embedding models; otherwise estimated cost is reported as unknown. Cost is an approximation, not provider billing data.

## Retrieval and indexing behavior

Python declarations are extracted with `ast`; JavaScript, TypeScript, Java, Go, Rust, and Ruby declarations use Tree-sitter. Large declarations are split on source-line boundaries while retaining symbol, parent, and import context. Markdown is chunked by headings. Unsupported or malformed declarations are bounded to a file/module chunk instead of being sent wholesale to the model.

The first index reads a repository archive and filters supported source/documentation files. Later commits on the indexed lineage copy existing vectors into a new commit snapshot, re-embed only changed files, and remove deleted-file chunks from the new snapshot. Diverged histories are indexed as a separate full snapshot. Old snapshots are pruned to the configured retention window. This avoids repeated full embedding work on a linear commit history.

Retrieval filters by repository **and exact commit SHA** before Qdrant vector search. Lexical/symbol/path/test/documentation candidates are ranked alongside semantic candidates. Context is budgeted and labeled as untrusted. Tools are bounded and allow the review model to search for code, symbols/callers/dependencies, tests, docs, the diff, and static findings. Git history/blame is not currently wired into the review workflow.

## Security and operational notes

- Repository contents, including code comments and documentation, are untrusted prompt data. They are explicitly fenced/escaped and the system policy forbids following repository instructions.
- The pipeline never executes repository build scripts. Static-analysis tools are invoked without a shell; Ruff is isolated, ESLint is run without project configuration, and Semgrep uses local rules.
- API credentials are read from environment-backed settings; tokens and source contents are not included in structured completion logs.
- Qdrant and the API/worker should be deployed on a private network with TLS and network policies in production. Protect operational read endpoints at the deployment boundary.
- The SQLite worker queue uses transactional job claiming and is suitable for local/small deployments. For higher concurrency, replace the queue/persistence layer with a shared production database and add worker-level repository locks.
- On Vercel, the default SQLite path uses `/tmp` so the read-only deployment filesystem does not prevent the app from starting. Vercel's temporary filesystem is not durable, and Vercel functions do not run the separate polling worker; use persistent database storage and a separately hosted worker for functioning webhook-driven PR reviews.
- The default index model retains a bounded set of commit snapshots per repository. Large repositories and very large PRs are subject to explicit archive, file, diff, context, and API pagination limits.

## Tests and evaluation

`tests/` exercises webhook signatures, durable idempotency, dashboard authorization and APIs, diff line mapping, parsers/chunking, vector retrieval, confidence filtering, Decisions API probability validation, deduplication, publisher formatting, and fixture metrics. GitHub/OpenRouter HTTP calls are mocked in tests. Qdrant tests use its in-memory client; a running external service is not needed for unit tests.

The evaluation fixtures contain known bug/security examples plus repository conventions and tests. `evaluation/metrics.py` can also be used with a custom expected/actual dataset. Evaluation outputs are printed as JSON; no quality score is fabricated when the review model is not configured.
