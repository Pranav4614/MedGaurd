# HealthcareGuard — complete GitHub package

HealthcareGuard is an experimental prompt-screening prototype with two separate layers:

1. **Clinical safeguard-bypass screening** — the original trained classifier and policy engine are preserved (`healthcareguard/guard.py`, `healthcareguard/policy.py`, `models/guard.joblib`).
2. **Patient-facing risk-pattern screening** — a new transparent rule layer (`healthcareguard/patient_risk.py`) checks for configured signals involving possible emergencies, possible overdose/poisoning, self-harm language, medication dose misuse, medication changes, allergy concerns, interaction questions, and higher-risk patient context.

The package also includes a redesigned responsive dashboard and a Manifest V3 Chrome extension for ChatGPT, Gemini and Claude.

> **Critical limitation:** the patient-facing layer is a transparent pattern detector, not a newly trained or clinically validated medical model. It does not diagnose, provide triage, calculate medication doses, verify drug interactions, or guarantee that a prompt or AI response is safe. Its rules can miss indirect wording and can produce false alarms. Do not use it as a clinical safety control. Do not submit real or identifiable patient information.

## Project structure

```text
healthcareguard_project/
├── server.py                     # FastAPI app; preserves /api/screen contract and adds risk layer
├── requirements-web.txt          # Runtime dependencies
├── requirements-dev.txt          # Test dependencies
├── render.yaml                   # Render deployment blueprint
├── Dockerfile                    # Optional container build
├── models/guard.joblib           # Existing prebuilt safeguard-bypass model
├── healthcareguard/
│   ├── guard.py                  # Existing ML safeguard-bypass detector
│   ├── policy.py                 # Existing context/policy layer
│   ├── patient_risk.py           # New patient-facing pattern detector
│   └── ...                       # Existing taxonomy, lexicon, model and evaluation code
├── static/index.html             # Redesigned dashboard
├── extension/                    # Chrome Manifest V3 extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── content.css
│   ├── popup.html
│   ├── popup.js
│   └── icons/
├── tests/                         # Automated screening and API tests
└── .github/workflows/tests.yml   # CI test workflow
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

## 3. Deploy the website and API on Render (free prototype)

The FastAPI app serves both the redesigned dashboard and its API from the same service.

1. Create a GitHub repository and upload the **extracted contents** of this project folder. Do not upload only the ZIP file.
2. Sign in to <https://render.com/> and connect your GitHub account.
3. Choose **New → Blueprint** and select the repository (the root contains `render.yaml`). If using **New → Web Service** instead, use the settings below.
4. Confirm the service is on the **Free** plan. Do not add a paid plan for this prototype.
5. Wait for the first deployment to complete.
6. Open the Render URL, then check `/api/health` and `/api/docs`.

Manual Web Service settings:

- Runtime: Python
- Build command: `pip install -r requirements-web.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`
- Python version environment variable: `PYTHON_VERSION=3.13.5`

The included `render.yaml` supplies these settings.

### Important free-tier behavior

Render's free web services currently spin down after 15 minutes without inbound traffic; the next request can take about a minute to wake the service. Free instances may restart, have an ephemeral filesystem and are documented by Render as unsuitable for production. This means the service has a public URL, but it is **not guaranteed to be continuously responsive or available 24/7**. See <https://render.com/docs/free> and <https://render.com/docs/faq>.

Do not try to bypass free-tier idle limits with artificial keep-alive traffic. The extension displays a connection message if the API is asleep or unavailable.

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
