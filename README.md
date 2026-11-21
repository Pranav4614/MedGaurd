# HealthcareGuard — complete GitHub package

HealthcareGuard is an experimental prompt-screening prototype with two separate layers:

1. **Clinical safeguard-bypass screening** — the original trained classifier and policy engine are preserved (`healthcareguard/guard.py`, `healthcareguard/policy.py`, `models/guard.joblib`).
2. **Patient-facing risk-pattern screening** — a new transparent rule layer (`healthcareguard/patient_risk.py`) checks for configured signals involving possible emergencies, possible overdose/poisoning, self-harm language, medication dose misuse, medication changes, allergy concerns, interaction questions, and higher-risk patient context.

The package also includes a redesigned responsive dashboard and a Manifest V3 Chrome extension for ChatGPT, Gemini and Claude.

> **Critical limitation:** the patient-facing layer is a transparent pattern detector, not a newly trained or clinically validated medical model. It does not diagnose, provide triage, calculate medication doses, verify drug interactions, or guarantee that a prompt or AI response is safe. Its rules can miss indirect wording and can produce false alarms. Do not use it as a clinical safety control. Do not submit real or identifiable patient information.

## Project structure

```text
MedGaurd/
├── server.py                     # FastAPI app and API routes
├── requirements.txt              # Vercel runtime dependencies
├── requirements-web.txt          # Local/container runtime dependencies
├── .python-version               # Python 3.13
├── vercel.json                   # Includes model and dashboard assets
├── guard.joblib                  # Existing trained safeguard-bypass model
├── healthcareguard/              # Importable Python package
│   ├── guard.py
│   ├── policy.py
│   ├── patient_risk.py
│   ├── lexicon.py
│   ├── taxonomy.py
│   ├── models.py
│   ├── features.py
│   ├── challenge.py
│   └── data_gen.py
├── index.html                    # Dashboard
├── install.html                  # Extension setup guide
├── manifest.json                # Chrome Manifest V3 (repository root)
├── background.js
├── content.js
├── content.css
├── popup.html
├── popup.js
└── icon16.png, icon32.png, icon48.png, icon128.png
```

## 1. Run locally on Windows

Install Python 3.13 and VS Code. Extract this repository and open the folder containing `server.py` in VS Code.

Open **Terminal → New Terminal** and run these commands one at a time:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-web.txt
```

Start the server:

```powershell
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- Dashboard: <http://127.0.0.1:8000/>
- Health check: <http://127.0.0.1:8000/api/health>
- Interactive API docs: <http://127.0.0.1:8000/api/docs>
- Extension instructions: <http://127.0.0.1:8000/extension>

Keep the terminal open while using the local app. Press `Ctrl+C` to stop the server.

## 2. Test the code

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Run tests:

```powershell
python -m pytest -q
```

The test suite covers the rule detector and API response structure. Passing these tests does **not** establish clinical validity, sensitivity, specificity, or safety performance.

## 3. Deploy the website and API on Vercel

The GitHub repository is connected to the Vercel project **secframework**:

- GitHub repository: `Pranav4614/MedGaurd`
- Production branch: `main`
- Framework preset: **FastAPI**
- Root directory: repository root
- Python runtime: **3.13** (from `.python-version`)
- Dependencies: `requirements.txt`
- Public dashboard and API base URL: <https://secframework.vercel.app>

Vercel detects the FastAPI application exported as `app` from the root `server.py` entrypoint. The same application serves the dashboard and API. No separate Render service, Docker container, database, paid AI API or API key is required for this prototype.

Health check:

- <https://secframework.vercel.app/api/health>

The health response reports whether the existing model loaded, whether a startup inference check passed, and whether the patient-risk rules are available.

### Important deployment and privacy notes

- The Vercel project is on the Hobby plan. Vercel documents Hobby as intended for personal, non-commercial use; review the current plan terms before using this as a commercial SaaS.
- The API is publicly reachable so the dashboard and browser extension can call it. It does not authenticate individual users or provide rate limiting in this prototype.
- Do not submit names, dates of birth, medical record numbers, or any other identifiable patient information.
- The API does not intentionally persist prompt text, but hosting infrastructure can retain operational logs. Do not treat this prototype as suitable for confidential or production healthcare workloads.
- A successful health check does not establish clinical validity, sensitivity, specificity or safety.

## 4. Install the Chrome extension (unpacked developer build)

1. Deploy the API first and copy its base URL, e.g. `https://your-service.onrender.com` (no trailing `/api`).
2. Open Chrome and visit `chrome://extensions`.
3. Turn on **Developer mode**.
4. Click **Load unpacked** and select the project's `extension` folder.
5. Pin HealthcareGuard from Chrome's Extensions menu.
6. Open the extension popup, paste the Render base URL, click **Save address**, and then **Test connection**.
7. Open ChatGPT, Gemini or Claude. Type a healthcare-related prompt and pause for about 1–2 seconds. A warning card appears if the API reports a configured risk signal.

For local testing, set the popup's API address to `http://127.0.0.1:8000` and keep the local server running.

### If you use a different hosting domain

The extension has host permissions for `*.onrender.com`, localhost and 127.0.0.1. If you deploy the API on another domain, add that API host pattern (for example `https://api.example.com/*`) to `host_permissions` in `extension/manifest.json`, save the file, and click **Reload** for the extension on `chrome://extensions`.

### Extension behavior and privacy

- Supported sites in this build: `chatgpt.com`, `chat.openai.com`, `gemini.google.com`, and `claude.ai`.
- The content script uses a local healthcare-keyword prefilter. Text that does not match the prefilter is not sent to the API.
- Healthcare-related text is sent to the configured API only after the user pauses typing for about 1.4 seconds.
- The extension does not save prompt history, read AI responses, change prompts, or automatically block sending.
- A warning is an advisory signal. It can be late, incomplete or incorrect. Do not rely on it as a clinical safeguard.
- Browser extension permissions and supported page DOM structures can change; retest after changes to target websites.

## API

### `GET /api/health`

Returns service health, version and detector metadata.

### `POST /api/screen`

Request:

```json
{
  "text": "Can I take twice my prescribed medication dose?",
  "role": "unknown",
  "verified": false,
  "source": "user",
  "flags": []
}
```

The response retains the original safeguard-bypass verdict fields and adds:

- `patient_risk`: categories, severity labels, matched phrases and advisory messages from the transparent rules layer.
- `combined_alert`: whether the dashboard/extension should show an alert, its level and summary.
- `screening_disclaimer`: product limitations.

Text is limited to 4,000 characters. The app does not intentionally persist submitted prompt text. Do not assume hosting providers, browsers or network infrastructure have no operational logs; review hosting privacy settings before public testing.

## Development notes

- Do not replace `models/guard.joblib` unless you have a separately evaluated replacement and have checked scikit-learn compatibility.
- Do not describe the new rule layer as a trained ML model. It is intentionally explainable and returns matched signals, not a probability score.
- Before any patient-facing or production use, build a clinically reviewed and independently labelled evaluation dataset, measure false-positive and false-negative behavior by category, add abuse controls and privacy controls, and obtain appropriate regulatory/legal review.
- No API key or secret is required by this prototype. The public API can be called by anyone who knows its URL; it is not suitable for confidential or production workloads as-is.

## License

No license is included. Add an appropriate license only after reviewing ownership and usage rights for all included source code and model artifacts.
