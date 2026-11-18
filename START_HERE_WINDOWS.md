# Start here — HealthcareGuard (Windows)

## A. Run the web app locally

1. Install Python 3.13 from <https://www.python.org/downloads/>. Tick **Add Python to PATH**.
2. Extract the project ZIP.
3. Open the extracted folder (the one containing `server.py`) in VS Code.
4. Choose **Terminal → New Terminal** and run:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-web.txt
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

5. Visit <http://127.0.0.1:8000/>.
6. Health check: <http://127.0.0.1:8000/api/health>.
7. API testing: <http://127.0.0.1:8000/api/docs>.

Keep the VS Code terminal open. Press `Ctrl+C` to stop the app.

## B. Run automated tests

In a second terminal (with `.venv` active):

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

## C. Deploy to the internet free for prototype testing

Upload the extracted project files (not just the ZIP) to a GitHub repository. In Render choose **New → Blueprint**, connect the repository, and use its `render.yaml`. The resulting service hosts the dashboard and API together.

The Render free service sleeps after 15 minutes without inbound traffic and may take about a minute to wake. It is not a 24/7 production service. See the main README for current limitations and deployment steps.

## D. Install the Chrome extension

The extension is in the `extension` folder. In Chrome visit `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select that folder. Open its popup, enter your deployed Render URL, save, and test the connection. Full steps are served at `/extension` once the app is running.

## Important

The original safeguard-bypass model is preserved. The new patient-facing risk layer is a pattern-based screening prototype, not a clinically validated model, diagnostic system, emergency triage tool or interaction database. Never enter identifiable patient information.
