# Upload this project to GitHub

**Important:** GitHub needs the extracted project files. Do not upload only `HealthcareGuard_Complete_GitHub_Package.zip` and expect Render to run it.

## Easiest method: GitHub Desktop

1. Install GitHub Desktop: <https://desktop.github.com/>.
2. Sign in with your GitHub account.
3. On GitHub.com, create a new repository named `HealthcareGuard`. Choose Public or Private. Do not initialize it with a README (this package already has one).
4. In GitHub Desktop choose **File → Clone Repository**, select `HealthcareGuard`, and clone it to your computer.
5. Extract the HealthcareGuard ZIP you downloaded from ChatGPT.
6. Open the extracted project folder. Copy **everything inside it** (including the hidden `.github` folder) into the cloned `HealthcareGuard` repository folder.
7. In GitHub Desktop, review the changed files. Enter a summary such as `Upload HealthcareGuard project` and click **Commit to main**.
8. Click **Push origin**.
9. Open the repository on GitHub.com and check that `server.py`, `render.yaml`, `healthcareguard`, `models`, `static` and `extension` are visible at the repository root.

Then follow the Render instructions in `README.md`.
