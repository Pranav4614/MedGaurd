(() => {
  "use strict";
  if (window.__healthcareGuardInstalled) return;
  window.__healthcareGuardInstalled = true;

  // Local-only prefilter: unrelated text is not sent to the backend.
  const HEALTHCARE_TERMS = /\b(medical|medicine|medication|prescription|dose|dosage|pill|tablet|drug|interaction|allergy|allergic|symptom|pain|chest|breath|breathing|shortness of breath|overdose|poison|insulin|warfarin|antibiotic|pregnan|blood pressure|diabetes|cancer|stroke|seizure|fever|doctor|pharmacist|nurse|patient|treatment|diagnos|side effect|kidney|liver|heart attack|addiction|addictive|alcohol|drinking|substance|craving|relapse|withdrawal|recovery|rehab|opioid|fentanyl|sedative|benzodiazepine|drug use|misinformation|viral post|health claim|harm reduction)\w*\b/i;
  const DEBOUNCE_MS = 1400;
  let timer = null;
  let lastSubmittedText = "";
  let lastObservedText = "";
  let dismissedText = "";
  let requestSequence = 0;
  let alertNode = null;

  function composerFrom(target) {
    if (!(target instanceof Element)) return null;
    const candidate = target.closest('textarea, input[type="text"], [contenteditable="true"], [role="textbox"]');
    if (!candidate) return null;
    const text = candidate.isContentEditable ? candidate.innerText : candidate.value;
    if (typeof text !== "string") return null;
    return { element: candidate, text: text.trim() };
  }

  function showAlert(result, text) {
    removeAlert();
    if (!result?.combined_alert?.alert || dismissedText === text) return;
    const patient = result.patient_risk || {};
    const combo = result.combined_alert;
    const urgent = combo.level === "urgent";
    const high = combo.level === "high";

    alertNode = document.createElement("section");
    alertNode.className = "hg-alert" + (urgent ? " hg-urgent" : high ? " hg-high" : " hg-review");
    alertNode.setAttribute("role", "alert");
    alertNode.setAttribute("aria-live", "assertive");

    const top = document.createElement("div");
    top.className = "hg-alert-top";
    const icon = document.createElement("span");
    icon.className = "hg-alert-icon";
    icon.textContent = urgent ? "!" : high ? "⚠" : "⌕";
    const titleWrap = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = combo.title || "HealthcareGuard warning";
    const subtitle = document.createElement("small");
    subtitle.textContent = "HealthcareGuard · advisory screening";
    titleWrap.append(title, subtitle);
    const close = document.createElement("button");
    close.className = "hg-close";
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss alert");
    close.textContent = "×";
    close.addEventListener("click", () => { dismissedText = text; removeAlert(); });
    top.append(icon, titleWrap, close);

    const message = document.createElement("p");
    const firstFinding = patient.findings?.[0];
    message.textContent = urgent && firstFinding?.message ? firstFinding.message : (combo.summary || "A configured healthcare safety pattern was detected. Review before sending.");

    const tags = document.createElement("div");
    tags.className = "hg-tags";
    (patient.findings || []).slice(0, 3).forEach(f => {
      const tag = document.createElement("span");
      tag.textContent = f.label;
      tags.append(tag);
    });
    (result.substance_risk?.findings || []).slice(0, 3).forEach(f => {
      const tag = document.createElement("span");
      tag.textContent = f.label;
      tags.append(tag);
    });
    if (result.patient_risk_ml?.detected && result.patient_risk_ml?.label) {
      const tag = document.createElement("span");
      tag.textContent = result.patient_risk_ml.label + " (experimental)";
      tags.append(tag);
    }
    if (result.flagged) {
      const tag = document.createElement("span");
      tag.textContent = "Clinical safeguard bypass signal";
      tags.append(tag);
    }

    const actions = document.createElement("div");
    actions.className = "hg-actions";
    const open = document.createElement("button");
    open.className = "hg-open";
    open.type = "button";
    open.textContent = "Open report";
    open.addEventListener("click", () => chrome.runtime.sendMessage({ type: "HG_OPEN_DASHBOARD" }));
    const dismiss = document.createElement("button");
    dismiss.className = "hg-dismiss";
    dismiss.type = "button";
    dismiss.textContent = "Dismiss";
    dismiss.addEventListener("click", () => { dismissedText = text; removeAlert(); });
    actions.append(open, dismiss);

    const note = document.createElement("div");
    note.className = "hg-note";
    note.textContent = urgent
      ? "Possible emergency signal: do not wait for an AI response. Seek urgent real-world help."
      : "This extension does not diagnose, validate treatment or block sending. Check concerns with a qualified professional.";
    alertNode.append(top, message, tags, actions, note);
    document.documentElement.append(alertNode);
  }

  function removeAlert() {
    if (alertNode?.isConnected) alertNode.remove();
    alertNode = null;
  }

  function scheduleScan(target) {
    const found = composerFrom(target);
    if (!found) return;
    const text = found.text;
    if (text !== lastObservedText) {
      lastObservedText = text;
      requestSequence++; // discard any response for an earlier version of the prompt
      dismissedText = "";
    }
    if (text.length < 15 || !HEALTHCARE_TERMS.test(text)) {
      lastSubmittedText = "";
      removeAlert();
      return;
    }
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (text === lastSubmittedText) return;
      lastSubmittedText = text;
      const seq = ++requestSequence;
      chrome.runtime.sendMessage({ type: "HG_SCREEN_PROMPT", text }, response => {
        if (chrome.runtime.lastError || !response?.ok || seq !== requestSequence) return;
        showAlert(response.result, text);
      });
    }, DEBOUNCE_MS);
  }

  document.addEventListener("input", event => scheduleScan(event.target), true);
})();
