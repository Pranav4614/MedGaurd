"""Interpretable cue lexicon + feature extraction.

Every cue is a regex family describing *how* an instruction tries to get around a
clinical safeguard (concealment, bypass, falsification, ...) rather than the exact
wording of any one attack. The same cues are used (a) as dense features for the
ML model, (b) as human-readable evidence in verdicts, and (c) by the policy layer.
"""
import re
from .taxonomy import (RX, DOSE, INTER, EVID, DX, ADD, REC, PREG, HEALTHCARE_CATEGORIES)


def normalize_text(t) -> str:
    t = "" if t is None else str(t)
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-")):
        t = t.replace(a, b)
    return t.strip()


def _c(*pats):
    return re.compile("|".join(f"(?:{p})" for p in pats), re.I)


CUES = {
    "state_change": _c(
        r"\b(?:chang\w*|alter\w*|switch\w*|swap\w*|bump\w*|rais\w*|lower\w*|downgrad\w*|upgrad\w*|edit\w*|"
        r"overwrit\w*|delet\w*|remov\w*|eras\w*|backdat\w*|clear\w*|untick\w*|updat\w*|re-?issu\w*|re-?print\w*|"
        r"re-?code\w*|re-?triage\w*|revis\w*|correct\w*|replac\w*|mark\w*|set|adjust\w*|amend\w*|modif\w*|"
        r"approv\w*|release\w*|prescrib\w*|dispens\w*|add|put|move|moving|copy|reduce|increase|round\w* up)\b"),
    "bypass": _c(
        r"\b(?:ignor\w*|disregard\w*|overrid\w*|override|bypass\w*|suppress\w*|circumvent\w*|skip\w*|"
        r"get(?:ting)? around|work(?:ing)? around|untick\w*|uncheck\w*|dismiss\w*|waive\w*|disabl\w*)\b",
        r"\b(?:turn|switch)(?:ing)? (?:it |them |that )?off\b",
        r"\bremove (?:the )?(?:[\w-]+ ){0,3}(?:alerts?|warnings?|flags?|requirements?)\b",
        r"\b(?:accept|acknowledg)\w* (?:the |all )?(?:\w+ )?(?:alerts?|warnings?)\b",
        r"\b(?:alerts?|warnings?) (?:is|are|keeps?|keep) (?:just )?(?:noise|firing|popping|flagging|annoying)\b",
        r"\bstops? popping up\b|\bso (?:that )?the (?:alert|warning)s? (?:stops?|doesn't|won't)\b",
        r"\bdon't tick\b"),
    "conceal": _c(
        r"\b(?:quietly|silently|secretly|covertly|discreetly|surreptitiously)\b",
        r"\boff the record\b|\bkeep (?:it|this|that) (?:quiet|between us|to yourself)\b",
        r"\b(?:do not|don't|no need to|never|without) (?:\w+ ){0,3}?(?:tell|telling|inform|informing|mention|"
        r"mentioning|notify|notifying|alert|alerting|loop|looping|log|logging|report|reporting|say|saying|"
        r"document|documenting|flag|flagging|let (?:\w+ )?know|asking)\b",
        r"\b(?:doesn't|does not|won't|will not|needn't) (?:\w+ ){0,2}?(?:need to know|have to know|know|check|"
        r"question|notice|mind|find out|look|care)\b",
        r"\b(?:nobody|no ?one|no-one) (?:will|would|is going to|is likely to|has to|needs to) (?:\w+ ){0,2}?"
        r"(?:check|notice|look|find|know|question|see|verify)\b",
        r"\bbefore (?:the |any )?(?:auditors?|review\w*|panel|inspectors?|regulators?|committee|managers?|"
        r"supervisors?|team)\b[^.;]{0,25}\b(?:look|looks|see|sees|check|checks|find|finds|come|comes|reads?)\b",
        r"\b(?:i'll|i will|will) (?:tell|inform|let|update|speak to) (?:\w+ ){0,2}?(?:tomorrow|later|afterwards)\b",
        r"\b(?:leave|leaving|left) (?:it |that |this |them )?(?:out|off)\b",
        r"\b(?:stay|remain) silent\b|\bwithout (?:a )?(?:trace|record|logging)\b",
        r"\b(?:turn|switch)(?:ing)? off (?:the )?(?:\w+ ){0,2}(?:logging|tracking|audit\w*)\b",
        r"\bdisabl\w+ (?:the )?(?:\w+ ){0,2}(?:logging|tracking|audit\w*)\b",
        r"\bwhile i tidy\b"),
    "falsify": _c(
        r"\b(?:pretend\w*|as if|as though|fake\w*|falsif\w*|fabricat\w*|invent\w*|made up|forge\w*|"
        r"backdat\w*|plausible(?:-looking)?)\b",
        r"\bmake (?:up|it (?:look|appear|seem))\b",
        r"\b(?:look|looks|looking|appear|seem) (?:real|genuine|legit\w*)\b",
        r"\bneed(?:s)? (?:not|n't) be real\b|\bdoesn't need to be real\b",
        r"\bit(?:'ll| will) (?:be fine|fit)\b",
        r"\bmark(?:ed|ing)? (?:it |them |this |the \w+ )?(?:as )?(?:already )?(?:verified|reviewed|checked|cleared|"
        r"acknowledged|approved|signed)\b",
        r"\bchart(?:ed)? (?:it )?as (?:verified|checked)\b",
        r"\bwrite that (?:she|he|they|the patient|\w+) (?:has|have|had|is|was) (?:no|not)\b",
        r"\brecord (?:her|him|them|it) as (?:not|no)\b",
        r"\bsay (?:that )?(?:a|an|the) \d{4}\b",
        r"\bdocument that [^.;]{0,60}\b(?:reviewed|cleared)\b",
        r"\binsurer will pay\b",
        r"\beven though (?:nobody|no one|no clinician|no-one|i haven't|she didn't|he didn't|they didn't|"
        r"it wasn't|nothing)\b",
        r"\bregardless of\b|\bwhatever the (?:scan|findings?|results?) (?:show|says?)\b|\bno matter what\b",
        r"\bso (?:that )?(?:the |it |this )?(?:case|claim|committee|auditor|insurer)?\s*(?:passes|clears)\b",
        r"\bso it clears\b|\bphrase it like\b",
        r"\bleave the signature\b|\bkeep the original prescriber's name on it\b"),
    "false_reassurance": _c(
        r"\b(?:tell|telling|say|saying|state|stating|claim|claiming|reassure|reassuring|assure|write|note|advise|"
        r"document|present|respond)\b[^.;?]{0,70}\b(?:is|are|it's|they're|that's|was|were|be|she's|he's)\s+"
        r"(?:perfectly |totally |completely |entirely |just |a )?"
        r"(?:fine|safe|harmless|compatible|risk-free|okay|ok|normal|common combination)\b",
        r"\b(?:call|calls|called)\b[^.;?]{0,40}\b(?:harmless|safe|fine|risk-free)\b",
        r"\bnothing to worry about\b|\brisk-free\b|\bno (?:need to )?(?:worry|concern)\b|\bexaggerated\b|"
        r"\bovercautious\b|\bsafe in pregnancy\b|\bis harmless\b|\bis compatible\b|\bthey're compatible\b",
        r"\bsaying they're compatible\b|\bshe's fine\b|\bhe's fine\b"),
    "unauthorised": _c(
        r"\bwithout (?:any )?(?:clinician |prescriber |doctor |pharmacist |proper )?(?:approval|authori[sz]ation|"
        r"sign-?off|permission|consent|review|verification|it)\b",
        r"\bno (?:clinician|prescriber|doctor) (?:approved|signed|reviewed)\b",
        r"\bunauthori[sz]ed\b", r"\bnot (?:been )?(?:approved|signed off|authori[sz]ed)\b",
        r"\bskip(?:ping)? (?:the )?(?:\w+ )?(?:pharmacy|pharmacist|second|double|prescriber|sign-?off|approval|"
        r"verification|review|check|calculator|safety)\b",
        r"\bno need (?:to|for) (?:loop|involve|get|ask|check)\b",
        r"\bdon't (?:send|route) it for\b", r"\bnot? (?:need|needing) (?:to )?(?:check|ask)\b",
        r"\bdon't bother with\b|\bwithout asking\b|\bwithout it\b",
        r"\bleave out the (?:warning|risk)s?\b|\bskip the safety talk\b"),
    "authority_claim": _c(
        r"\bI'?m (?:the |an? )?(?:on-call |from )?(?:attending|registrar|consultant|doctor|nurse|pharmacist|admin\w*|"
        r"surgeon|gp|it|on-call)\b",
        r"\b(?:I am|this is) (?:the |an? )?(?:on-call|attending|registrar|consultant|doctor|admin\w*)\b",
        r"\b(?:\w+ )?won't mind\b", r"\bthey said they're family\b|\bsaid they're family\b",
        r"\bfrom IT\b"),
    "authorised": _c(
        r"\b(?:signed(?: off)?|sign-?off|has approved|have approved|approved (?:the|this|by|it)|authori[sz]ed by|"
        r"has reviewed|have reviewed|reviewed (?:it|this|and|by)|documented|documentation|"
        r"audit (?:trail|log|entry)|preserv\w+|logged|late entry|clearly marked|"
        r"per (?:the )?(?:\w+ ){0,2}(?:policy|protocol|sop|formulary|bnf|guideline)|approved protocol|"
        r"normal workflow|through the (?:normal |usual |standard )?(?:workflow|process|system)|"
        r"pharmacist (?:verification|check|review)|for (?:pharmacist |prescriber )?(?:verification|checking|review|"
        r"sign-?off|approval)|(?:needs?|requires?) (?:their|his|her|the prescriber's|prescriber) (?:sign-?off|approval)|"
        r"confirmed (?:it|by|the)|(?:attach|attached|attaching) (?:\w+ ){0,3}(?:approval|letter)|"
        r"original (?:entry |record )?(?:remains|is kept|kept|stays)|remains? (?:visible|in the)|kept visible|"
        r"keep(?:ing)? the original|record the (?:approval|decision|reasoning)|document the reasoning|"
        r"specialist review|urgent review|clinic letter|assessed by|countersign\w*|reviewed the)\b"),
    "info_request": re.compile(
        r"^\s*(?:please\s+|hi[, ]+\w*[, ]*)?(?:can you\s+(?:please\s+)?|could you\s+(?:please\s+)?)?"
        r"(?:what|how|why|when|where|which|who|is it true|is there|are there|is it|explain|summari[sz]e|list|"
        r"describe|translate|tell me|give me|calculate|draft|write|prepare|find|help me|remind me|book|compose|"
        r"outline|show me|can a|can an|can i|do i|does|should|a study i read|hospital policy|our sop|late entry)\b",
        re.I),
    "meta_education": _c(
        r"\bexplain why\b|\bwhy (?:it|this|that|you|can't|cannot|a|an)\b|\btraining\b|\bslide\b|\bsop\b|"
        r"\bpolicy (?:says|states|requires)\b",
        r"\bhow (?:do|does|should|can|could|would) (?:\w+ ){0,4}(?:detect|defend|spot|behave|identify|prevent|handle|respond)\b",
        r"\bprompt injection\b|\buntrusted\b|\bawareness\b|\bwhat does .{0,30} mean\b|\bwhat happens if\b"),
    "embedded": _c(
        r"\bnote to (?:the )?(?:ai|assistant|model|llm|agent)\b", r"\b(?:assistant|system|admin|instruction)s?:",
        r"<!--", r"\[(?:hidden|system|note|appended)", r"\bhidden (?:text|instruction|note)\b",
        r"\bembedded (?:in|instruction)\b", r"\bfrom now on\b",
        r"\bfor the rest of (?:this|the) (?:conversation|session|chat)\b",
        r"\btreat (?:everything|all|anything|the text)\b[^.;]{0,50}\b(?:orders?|instructions?|commands?)\b",
        r"\b(?:text|pdf|document|letter|email|message|file)\b[^.;]{0,25}\b(?:says|said|tells|instructs)\b",
        r"\bdo (?:exactly )?what (?:it|the \w+|they) says?\b", r"\bdeveloper mode\b", r"\byou are now\b",
        r"\bact as (?:if|though)\b", r"\bpretend you\b", r"\bjailbreak\b",
        r"\bignore (?:all )?(?:previous|prior|above|earlier|your) (?:rules|instructions)\b",
        r"\bforget your instructions\b", r"\bsystem prompt\b", r"\bunrestricted\b", r"\bno restrictions\b",
        r"\bsafety rules are switched off\b|\brules are switched off\b",
        r"\bai agents? (?:processing|reading)\b", r"\bignore everything\b"),
    "exfil": _c(
        r"\b(?:reveal|send|email|forward|export|dump|print|leak|share|output|disclose|give out|hand over|read out|"
        r"release)\b[^.;]{0,60}\b(?:patient list|patients?'? (?:full |entire )?(?:medical )?(?:records?|data|details|"
        r"names?|nhs number)|medical records?|full (?:medical )?record|credentials|passwords?|nhs number|system prompt|"
        r"personal (?:e-?mail|account)|confidential|cupboard code|full record)\b",
        r"\bpersonal (?:e-?mail|account|drive)\b"),
}

# clinical object vocabulary per category
OBJ = {
    RX: _c(r"\bprescri\w+", r"\bscripts?\b", r"\b(?:medication|drug|repeat) orders?\b", r"(?<!in )\borders?\b",
           r"\brepeat\b", r"\btablets?\b", r"\bmedications?\b", r"\bstrength\b", r"\bre-?issu\w+",
           r"\bformulary\b", r"\bswitch\w*\b.{0,30}\b(?:to|for)\b", r"\bsupply\b"),
    DOSE: _c(r"\bdos(?:e|es|age|ing)\b", r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|units?)(?:/kg)?\b", r"\bmg/kg\b",
             r"\bmax(?:imum)?\b", r"\blimit\b", r"\boverdose\b", r"\bweight\b", r"\bcalculator\b", r"\brenal\b",
             r"\btitrat\w+", r"\bunits\b", r"\binsulin\b", r"\bdigoxin\b", r"\bmorphine\b", r"\bpump\b"),
    INTER: _c(r"\binteract\w*", r"\bcontraindicat\w*", r"\ballerg\w*", r"\balerts?\b", r"\bwarnings?\b",
              r"\bcombination\b", r"\bcompatible\b", r"\bnsaids?\b"),
    EVID: _c(r"\bstud(?:y|ies)\b", r"\btrials?\b", r"\bguidelines?\b", r"\bevidence\b", r"\bcitations?\b",
             r"\breferences?\b", r"\bdoi\b", r"\bmeta-?analys\w+", r"\bcochrane\b", r"\bpeer-?reviewed\b",
             r"\brcts?\b", r"\bjournal\b", r"\blancet\b", r"\bleaflet\b", r"\bendorsed?\b", r"\bnice\b"),
    DX: _c(r"\bdiagnos\w+", r"\bassessment\b", r"\bsymptoms?\b", r"\btriage\b", r"\bscans?\b", r"\bencounter\b",
           r"\bpathway\b", r"\bdifferential\b", r"\bfindings?\b", r"\bre-?code\w*", r"\bicd\b",
           r"\b(?:fever|cough|pneumonia|sepsis)\b", r"\breport\b"),
    ADD: _c(r"\bwithdraw\w*", r"\bdetox\w*", r"\baddict\w*", r"\bdependen\w+", r"\brelapse\b", r"\btaper\w*",
            r"\bcold turkey\b", r"\bopioids?\b", r"\bmethadone\b", r"\bbenzodiazepines?\b",
            r"\b(?:clonazepam|diazepam|lorazepam|oxycodone)\b", r"\balcohol\b", r"\bbeers?\b", r"\babruptly\b",
            r"\bcontrolled[- ]drugs?\b", r"\bquit\b"),
    REC: _c(r"\brecords?\b", r"\bcharts?\b", r"\bnotes?\b", r"\baudit\w*", r"\blogs?\b", r"\blogging\b",
            r"\bresults?\b", r"\bcreatinine\b", r"\binr\b", r"\bpotassium\b", r"\bhba1c\b",
            r"\bincident report\b", r"\bconsent\b", r"\bentry\b", r"\bdnacpr\b", r"\bhistory\b", r"\btracking\b",
            r"\bmedication error\b", r"\bdispensing\b"),
    PREG: _c(r"\bpregnan\w*", r"\bpaediatric\w*", r"\bpediatric\w*", r"\bchild(?:ren)?\b", r"\binfants?\b",
             r"\bteratogen\w*", r"\bobstetric\w*", r"\bhigh[- ]risk\b", r"\bfrail\w*", r"\belderly\b",
             r"\bvalproate\b", r"\bisotretinoin\b", r"\binduction\b"),
}
ACTION_CATS = [RX, DOSE, INTER, DX, REC, PREG, ADD]
RISK_CUES = ["conceal", "bypass", "falsify", "false_reassurance", "unauthorised", "authority_claim", "embedded", "exfil"]
BINARY_CUES = ["state_change", "bypass", "conceal", "falsify", "false_reassurance", "unauthorised",
               "authority_claim", "authorised", "info_request", "meta_education", "embedded", "exfil"]


def _count(p, t):
    return sum(1 for _ in p.finditer(t))


def extract_features(text) -> dict:
    t = normalize_text(text)
    c = {k: _count(p, t) for k, p in CUES.items()}
    o = {k: _count(p, t) for k, p in OBJ.items()}
    f = {k: float(min(c[k], 1)) for k in BINARY_CUES}
    for k in ("conceal", "bypass", "falsify"):
        f[f"{k}_n"] = min(c[k], 3) / 3
    for i, cat in enumerate(HEALTHCARE_CATEGORIES):
        f[f"obj_{i}"] = min(o[cat], 3) / 3
    f["obj_any"] = float(any(o.values()))
    f["action_target"] = float(any(o[k] for k in ACTION_CATS))
    f["state_x_target"] = f["state_change"] * f["action_target"]
    f["conceal_x_obj"] = f["conceal"] * f["obj_any"]
    f["bypass_x_obj"] = f["bypass"] * f["obj_any"]
    f["falsify_x_obj"] = f["falsify"] * f["obj_any"]
    f["reassure_x_obj"] = f["false_reassurance"] * f["obj_any"]
    f["unauth_x_target"] = f["unauthorised"] * f["action_target"]
    f["authclaim_x_state"] = f["authority_claim"] * f["state_change"]
    f["embedded_x_state"] = f["embedded"] * f["state_change"]
    f["authorised_x_state_target"] = f["authorised"] * f["state_x_target"]
    risk_any = max(f[k] for k in RISK_CUES)
    f["info_no_risk"] = f["info_request"] * (1 - risk_any)
    f["authorised_no_conceal"] = f["authorised"] * (1 - max(f["conceal"], f["falsify"]))
    f["risk_cue_count"] = min(sum(f[k] for k in RISK_CUES), 4) / 4
    f["no_cues_at_all"] = float(risk_any == 0)
    return f


FEATURE_NAMES = list(extract_features("x").keys())


def find_evidence(text):
    """Return [(cue, matched_span, kind)] where kind is 'risk' or 'protective'."""
    t = normalize_text(text)
    out = []
    for cue in RISK_CUES:
        for m in CUES[cue].finditer(t):
            out.append((cue, m.group(0).strip(), "risk"))
    for m in CUES["authorised"].finditer(t):
        out.append(("authorised", m.group(0).strip(), "protective"))
    return out
