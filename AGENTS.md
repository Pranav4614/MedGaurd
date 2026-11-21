# HealthcareGuard — Repository Agent Instructions and Deployment Repair Task

> These instructions are for GitHub Copilot coding agent working in this repository. Treat the checked-in source code as the source of truth. Do not rely on previous chat messages, generated patch ZIPs, assumed filenames, or claims about tests that are not reproducible in this checkout.

## 1. Product and current deployment context

HealthcareGuard is a prompt-safety screening prototype with three distinct responsibilities:

1. **Clinical safeguard-bypass detection** — identify attempts to bypass, suppress, falsify, or manipulate clinical safeguards.
2. **General jailbreak / prompt-injection detection** — identify instruction hierarchy attacks, role or authority escalation, hidden-prompt extraction, indirect prompt injection, tool/permission manipulation, concealment, and related behaviours.
3. **Patient-facing healthcare-risk screening** — identify selected potentially urgent symptoms and unsafe medication-related prompts. This is screening only, not diagnosis or medical advice.

The existing public frontend is hosted on Netlify:

- `https://medgaurd-repo.netlify.app`

The user has **not deployed a backend to Render**. The selected backend target is **Vercel Hobby**, subject to its current plan terms and limits. The user requires a deployment path with **no billing account, no payment method, no paid API and no automatic paid usage**. Do not configure Render, Google Cloud billing, paid Vercel features, paid AI APIs, or payment-dependent services.

Vercel Hobby is for personal, non-commercial use. If the repository or deployment is intended for commercial SaaS use, stop and clearly explain that this free plan may not be suitable; do not imply that Hobby is approved for commercial use.

No AI Gateway key is required for the current local ML/rules architecture. A credential resembling a Vercel AI Gateway key was accidentally shared in conversation. Do not use it, copy it, add it to the repo, or put it in examples. The owner should revoke it. Never ask the user to paste secrets into chat or commit secrets.

## 2. Non-negotiable working rules

1. **Inspect before editing.** First map the actual repository tree, entrypoints, dependency files, model artifacts, extension files, frontend framework, tests, and deployment configuration.
2. **Do not assume files exist.** In particular, earlier patches referred to `healthcareguard.jailbreak_intent` and `healthcareguard.jailbreak_ml`; verify whether these modules exist in this checkout before importing them.
3. **No broad rewrites.** Preserve working code and make the smallest coherent change set. Do not regenerate the UI or replace the complete backend unless the audit proves it is necessary.
4. **No placeholder success.** Never report a model as loaded, trained, accurate, deployed, or tested unless code, artifacts, and logs substantiate the claim.
5. **No fake models or metrics.** Do not fabricate `.joblib` artifacts, training examples from a claimed external source, accuracy, precision, recall, test counts, or benchmark scores.
6. **No database-only matching.** A prompt must not be rejected with messages such as “prompt not found in database.” Evaluate unfamiliar wording with the available detection layers and communicate uncertainty honestly.
7. **Keep detection layers separate.** A clinical-risk signal is not automatically a jailbreak signal. A jailbreak signal is not automatically a patient-risk signal. Return distinct assessments and a transparent combined result.
8. **Fail visibly and safely.** If a detector or artifact is unavailable, state that it is unavailable/degraded. Do not silently classify the prompt as safe or return an empty success response.
9. **No patient prompt logging by default.** Do not log full prompt bodies, store user prompts, or add analytics that collect health-related text. Log request IDs, timings, detector status and exception metadata only.
10. **No secrets in source control.** Add/update `.gitignore` for `.env`, `.env.*` (except a clearly safe `.env.example`), tokens, credentials, local virtual environments, caches and temporary files. Do not create or require a Vercel token for Git-based deployment.
11. Do not change the user's Netlify site or GitHub repository settings directly. Make code changes and provide exact manual deployment steps; deployment success must be verified separately.
12. Do not modify the existing trained clinical artifact unless a reproducible compatibility issue is demonstrated. Preserve its provenance and current label/feature schema.

## 3. Required audit before implementation

Before changing files, report a concise inventory covering:

- Top-level tree and actual Python package layout.
- The current FastAPI entrypoint and the exported `FastAPI()` object name.
- All API routes, request/response schemas and existing compatibility requirements.
- Every import referenced by the backend, tests and extension; identify missing modules.
- Python version and dependency manifests (`requirements-web.txt`, `requirements.txt`, `pyproject.toml`, `.python-version`, etc.).
- Existing model artifacts and their actual load paths. Verify whether `models/guard.joblib` exists and whether any jailbreak model artifact exists.
- How the frontend determines its API base URL today.
- How the Chrome extension determines its API base URL and which permissions it declares.
- Existing tests and commands that can be run in this checkout.
- Any bundle-size risks, especially scikit-learn, scipy, numpy, joblib and `.joblib` files.

Do not begin deployment-specific edits until this audit is complete. If a critical fact cannot be determined, write it down instead of guessing.

## 4. Vercel backend requirements

Use current official Vercel documentation while implementing. Relevant references:

- FastAPI deployment: https://vercel.com/docs/frameworks/backend/fastapi
- Python runtime: https://vercel.com/docs/functions/runtimes/python

Current Vercel documentation says a FastAPI instance named `app` can be exported from supported entrypoints including `app.py`, `index.py`, `server.py`, `main.py`, `wsgi.py`, or `asgi.py`, including supported locations under `src/` or `app/`. Dependencies can be declared in `requirements.txt`, `pyproject.toml`, or Pipenv manifests. Verify current documentation again rather than hardcoding assumptions.

Implementation requirements:

1. Make the existing FastAPI application discoverable by Vercel using the least disruptive supported entrypoint. Do not create a second independent FastAPI app that duplicates routes or state.
2. Ensure a supported dependency manifest exists at the repository root. If `requirements-web.txt` is the maintained runtime manifest, create a minimal `requirements.txt` that is accurately derived from it, or use a supported `pyproject.toml`. Do not blindly include development-only packages.
3. Pin a supported Python version compatible with the installed dependencies and serialized model artifacts. Do not change scikit-learn/numpy versions casually: pickle/joblib compatibility matters. Explain the selected versions.
4. Keep the function bundle within Vercel's documented size limits. Include only runtime-required model artifacts and code. Exclude training datasets, test fixtures, notebooks, caches, local environments, docs and unrelated static assets from the Python Function bundle where applicable. Do not exclude a model file required at runtime.
5. Do not train models at cold start or during a request. Load existing artifacts once per warm function instance and cache the loaded objects safely.
6. Vercel Functions are serverless. Do not rely on a persistent local filesystem, background daemon, in-memory counter across invocations, or a continuously running Uvicorn process. Any file writes must be temporary and non-essential.
7. Respect Vercel function duration/memory limits. Keep screening synchronous and bounded. If the current ML stack or model bundle cannot fit within the free function limits, stop and explain the measured blocker; do not pretend it deployed.
8. Add only the minimum required `vercel.json` or `pyproject.toml` configuration. Avoid conflicting routing rules. Preserve the existing `/api/health` and `/api/screen` routes, if they exist.
9. Add a deployment README with the exact Vercel project settings and commands. The user should be able to connect the GitHub repository through Vercel's dashboard without a token or payment method.
10. Do not add a paid database or external AI service.

## 5. API compatibility, CORS and health reporting

Preserve existing API request/response contracts after inspecting them. Do not invent field names or silently break the current website/extension. If a contract change is necessary, provide backward compatibility or update every caller and tests in the same change.

Required API behaviour:

- Keep `GET /api/health` and `POST /api/screen` if those endpoints exist in the current project.
- Health reporting must distinguish `ready`, `degraded` and `error` where appropriate.
- Report each detector separately: original clinical model loaded/status, jailbreak rules loaded/status, jailbreak ML model loaded/status, patient-risk rules loaded/status.
- Do not report `jailbreak_ml_loaded: true` unless the expected artifact is found and successfully deserialised.
- If the jailbreak ML model is not trained or absent, mark it `not_trained` / `not_loaded` and retain the available rules detector if implemented. Do not crash the whole service solely because an explicitly optional detector is absent.
- If the original clinical model is required but missing or incompatible, report degraded/error accurately. Do not fall through to an “all clear” result.
- Ensure CORS allows the exact production origin `https://medgaurd-repo.netlify.app`. Add other origins only when verified and needed. Do not use `*` with credentials.
- CORS is not authentication. Do not claim it protects the API from public abuse.
- Validate maximum prompt length and request shape. Return clear HTTP errors for malformed or excessively large requests.
- A no-signal result must say “No strong configured signal detected” or equivalent, never “safe”, “approved”, or “guaranteed safe.”
- An uncertain result must be representable and visible to the frontend.

## 6. Frontend on Netlify

Keep the current Netlify frontend deployed at `https://medgaurd-repo.netlify.app`.

1. Inspect the actual frontend stack (plain HTML/JS, Vite, React, etc.) before choosing an environment variable name.
2. Centralise API base URL selection in one place. Use the framework's correct build-time variable convention or a safe runtime config file.
3. Do not hardcode an invented Vercel URL. Use an explicit placeholder/configuration setting until the actual deployment URL is known.
4. Production frontend code must not silently fall back to `localhost`.
5. Display backend connection state, screening-in-progress state, result, uncertainty, detector availability and request errors.
6. A frontend build is not proof that the backend works. Add a simple API health test and a documented end-to-end smoke test.
7. Do not show fabricated dashboard statistics, model accuracy or detection counts.

## 7. Chrome extension

Preserve the existing Manifest V3 extension and its current UI where practical.

1. Inspect the actual extension files and supported host match patterns.
2. Make the API base URL configurable in the extension settings/popup, with no hardcoded fake domain.
3. Request only necessary host permissions. Avoid overly broad permissions where a narrow configured host can be used.
4. Ensure both the content script and popup/service worker use the same configured backend URL.
5. Handle API timeout, cold start, CORS/permission failure, non-2xx response, malformed JSON and unavailable detector explicitly in the UI.
6. Do not silently skip every unfamiliar prompt. If Demo Mode is enabled, send eligible composer text after a debounce; show a clear privacy notice because this sends ordinary text as well as healthcare prompts.
7. Do not send text from password fields, payment fields, or unrelated websites. Restrict supported AI hosts to the existing intended list (ChatGPT, Gemini and Claude) unless user explicitly configures another.
8. Do not auto-submit, rewrite, erase or block the user's prompt. Display advisory alerts only.
9. Do not claim extension functionality is tested in live ChatGPT/Gemini/Claude unless it was actually tested there.

## 8. Model and data integrity

Do not conflate code, a rules engine, a trained model artifact and an evaluation dataset.

- Preserve and identify the original clinical model artifact and its feature schema.
- If `jailbreak_intent.py` is missing, create it only if required by the inspected API/tests. Keep its rule-based findings explainable and separately labelled from ML scores.
- If `jailbreak_ml.py` is missing, create a real, tested training/inference module only if the current dataset and dependencies support it. Otherwise make the import optional and explicitly report the model as not trained/unavailable.
- Never create a dummy `.joblib` file or claim a model is trained because a Python module exists.
- If adding a public dataset, verify the canonical source, version, license, attribution and label schema. Keep its provided test split completely outside training and threshold selection. Do not upload the test split to a training function.
- Remove exact and normalised duplicates only with documented rules. Report conflicting labels; do not silently resolve them.
- Do not generate thousands of near-duplicate synthetic rows merely to inflate dataset size.
- Keep healthcare-risk screening independent from jailbreak detection. Do not claim clinical validation; require qualified clinical review for healthcare categories.
- Report actual precision, recall, F1, F2, false-positive rate, confusion matrix, per-category performance and held-out results only after running a reproducible evaluation on a leakage-safe split.
- Use an independent, untouched evaluation set. Avoid random row splitting when template-derived siblings can leak across splits; group by template/attack family/source where possible.
- Report uncertainty and limitations. Low vocabulary coverage alone must not be called malicious.

## 9. Tests and acceptance criteria

Add or update tests based on the actual API and implementation. At minimum test:

### Backend startup and packaging
- All backend imports resolve in a clean environment.
- FastAPI `app` imports from the chosen Vercel entrypoint.
- Dependency installation succeeds from the committed manifest.
- Model paths work from the repository root and deployed bundle.
- Missing optional jailbreak ML artifact produces an honest degraded status, not an import crash.
- Missing required original clinical artifact is reported clearly and never produces a false “safe” outcome.

### Screening behaviour
- Benign healthcare education questions.
- Personal medication misuse questions.
- Emergency-symptom wording.
- Direct jailbreak and instruction override prompts.
- Hidden prompt/system instruction extraction attempts.
- Indirect prompt injection in quoted or uploaded content.
- Tool/permission/audit manipulation.
- Benign hard negatives containing words such as “ignore”, “override”, “disable”, “delete”, “system”, or “instruction”.
- Unfamiliar, paraphrased and compositional prompts.
- Empty, malformed, very long and non-string input.

### Integration
- `/api/health` response reflects true loaded model states.
- `/api/screen` retains backward compatibility with actual current callers.
- Netlify frontend can call the Vercel API using the configured base URL.
- Extension popup and content script call the same backend and show errors.
- CORS preflight from the exact Netlify origin succeeds.

Run the full test suite and report exact commands, totals, failures and skipped tests. Do not say “all tests pass” if any tests were not run.

## 10. Required implementation sequence

Work in this order and do not skip steps:

1. **Audit only:** inspect repository, write a short audit report and list blockers. Do not change code during the audit.
2. **Repair imports and packaging:** make the existing API importable from a supported Vercel FastAPI entrypoint, with correct runtime dependencies.
3. **Preserve model behaviour:** make model load paths deterministic and health reporting truthful. Do not retrain or alter labels unless required and justified.
4. **Add Vercel deployment config:** minimal and compatible with current official docs.
5. **Connect frontend:** make API base URL configurable without hardcoding the not-yet-known Vercel deployment URL.
6. **Connect extension:** same backend URL and clear failure states.
7. **Test locally:** clean install, imports, API unit tests, frontend build and extension syntax checks.
8. **Produce deployment guide:** exact click-by-click instructions for importing the GitHub repository into Vercel Hobby, setting the project root/build settings only if needed, deploying without a billing account/payment method, copying the generated URL, and setting that URL in Netlify and the extension.
9. **Stop before claiming deployment:** actual Vercel deployment must be performed by the repository owner. Give a smoke-test checklist for `/api/health`, `/api/screen`, Netlify-to-API CORS, and extension-to-API connectivity.

## 11. Final report format

When finished, provide:

1. Repository inventory and the root cause(s) found.
2. Files added, modified and deleted (deletions should normally be zero).
3. The exact Vercel entrypoint and why it is valid under current Vercel documentation.
4. Python and dependency versions selected and compatibility reasoning.
5. Actual model artifacts found and their load statuses.
6. Exact tests run and real results.
7. Vercel function bundle size or best available build-size evidence; call out uncertainty if not measurable locally.
8. Exact manual Vercel deployment steps requiring no billing account.
9. Exact Netlify environment variable name/value format, with the Vercel URL left as a placeholder until deployment.
10. Exact Chrome extension backend setting and reload steps.
11. Remaining limitations and any items that need owner action.

**Definition of done:** The repository builds and tests locally from a clean environment; the FastAPI app imports using a Vercel-supported entrypoint; no missing imports or fake model states remain; the frontend and extension are configured to use one explicit API base URL; and the deployment guide does not claim a deployment has occurred before the owner performs it.

## 12. Official documentation to consult

- Vercel FastAPI: https://vercel.com/docs/frameworks/backend/fastapi
- Vercel Python runtime and dependency manifests: https://vercel.com/docs/functions/runtimes/python
- GitHub Copilot agent instructions: https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions

If current official documentation conflicts with these notes, follow current documentation and explain the change.
