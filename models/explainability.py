"""models/explainability.py — SHAP-based explanations and clinical rationale generation.

Public API
----------
generate_shap_explanation(model, X_test, feature_names, *, display_names=None) -> dict
    Compute SHAP values via TreeExplainer and return top-3 features with direction.

generate_llm_rationale(shap_results, patient_metadata) -> str
    Produce a template-based natural-language rationale from SHAP results.
    (In production this will be swapped for a Llama 3 / GPT-4o API call.)

compute_cognitive_match_score(rationale_text, shap_top3) -> float
    Hallucination-detection metric: fraction of top-3 SHAP features verifiably
    mentioned in the rationale text.

build_clinical_rationale(model, X_test, feature_names, patient_metadata, ...) -> ClinicalRationale
    Convenience wrapper that runs the full pipeline in one call.

ClinicalRationale (dataclass)
    Structured container for the full explainability output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ── Direction vocabulary used by the template rationale ──────────────────────

_DIR_PHRASES: dict[str, list[str]] = {
    "positive": ["elevated", "increased", "above-baseline"],
    "negative": ["declining", "reduced", "below-baseline"],
}


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class ClinicalRationale:
    """Structured output from the AI explainability pipeline.

    Attributes:
        shap_values:
            feature_name → signed mean SHAP across test observations.
            Positive values push toward the "high risk" class.
        top_features:
            Display names of the top-3 SHAP features, ordered by mean |SHAP|.
        rationale_text:
            Natural-language explanation generated from SHAP results (template-
            based now; LLM-generated in Phase 2).
        cognitive_match_score:
            Precision in [0, 1] measuring how many top-3 SHAP features the
            rationale actually mentions.  This is the hallucination-detection
            metric described in the ML for Health proposal.
            1.0 = all three features verifiably mentioned.
        confidence:
            Qualitative label derived from cognitive_match_score.
            "High" (≥2/3 features) · "Medium" (≥1/3) · "Uncertain" (0/3).
    """

    shap_values: dict[str, float]
    top_features: list[str]
    rationale_text: str
    cognitive_match_score: float
    confidence: str  # "High" | "Medium" | "Uncertain"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _snake_to_title(name: str) -> str:
    """'rem_sleep_pct' → 'Rem Sleep Pct' — fallback when no display_names map given."""
    return " ".join(w.capitalize() for w in name.split("_"))


def _feature_is_mentioned(feature_str: str, text_lower: str) -> bool:
    """Return True if at least one significant token from *feature_str* appears in *text_lower*.

    'Significant' means length > 3 characters (prevents false positives on
    prepositions or noise tokens).  Short acronyms like 'hrv', 'rem', 'adl'
    are checked as exact whole words instead.
    """
    tokens = feature_str.replace("_", " ").lower().split()
    for token in tokens:
        if not token.isalnum():
            continue  # skip punctuation tokens like '%'
        if len(token) <= 3:
            # Whole-word match (pad text with spaces to handle boundaries)
            if f" {token} " in f" {text_lower} ":
                return True
        else:
            if token in text_lower:
                return True
    return False


def _confidence_label(score: float) -> str:
    """Map a cognitive match score to a qualitative confidence label."""
    if score >= 0.6:   # ≥2 of 3 features mentioned
        return "High"
    if score >= 0.3:   # ≥1 of 3 features mentioned
        return "Medium"
    return "Uncertain"


# ── Public functions ──────────────────────────────────────────────────────────

def generate_shap_explanation(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    *,
    display_names: dict[str, str] | None = None,
) -> dict:
    """Compute SHAP values for a fitted tree-based model and return the top-3 features.

    Uses ``shap.TreeExplainer``, which is optimised for XGBoost and Random Forest.
    For other model families (e.g. LSTM) a ``ValueError`` is raised — kernel SHAP
    would be too slow for interactive clinical use.

    Args:
        model:         Fitted scikit-learn-compatible estimator (XGBClassifier or
                       RandomForestClassifier).
        X_test:        Feature matrix on which SHAP values are computed.  Accepts a
                       pandas DataFrame (preferred) or a NumPy array.
        feature_names: Ordered column names matching the model's input features.
                       Required when *X_test* is a NumPy array; ignored when *X_test*
                       is already a DataFrame with the correct columns.
        display_names: Optional ``{column_name: display_label}`` map.  When provided,
                       each entry in the returned ``top3`` list carries a
                       ``'display_name'`` drawn from this map; otherwise the column
                       name is title-cased as a fallback.

    Returns:
        A dict with two keys:

        ``'top3'`` — list of up to 3 dicts, each containing::

            {
                'name':         str,    # internal column name
                'display_name': str,    # human-readable label
                'shap_value':   float,  # mean signed SHAP (+ → increases risk)
                'abs_shap':     float,  # mean |SHAP| used for ranking
                'direction':    str,    # 'positive' | 'negative'
            }

        ``'all_shap'`` — ``{feature_name: signed_mean_shap}`` for every feature.

    Raises:
        ImportError: If the ``shap`` library is not installed.
        ValueError:  If *X_test* is empty or the model is not tree-based.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "shap is required for explainability. Run: pip install shap"
        ) from exc

    # Normalise X_test to a DataFrame
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)

    if X_test.empty:
        raise ValueError("X_test must contain at least one row.")

    # Compute SHAP values
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception as exc:
        raise ValueError(
            f"TreeExplainer failed for {type(model).__name__!r}. "
            "generate_shap_explanation supports XGBClassifier and "
            f"RandomForestClassifier. Original error: {exc}"
        ) from exc

    # Normalise to a 2-D array (n_samples, n_features) for the positive class.
    #
    # shap returns different shapes depending on version and estimator type:
    #   Old API, RandomForest → list[2-D array per class]: take index 1
    #   Old API, XGBoost      → single 2-D array
    #   New API (shap ≥1.x)   → 3-D array (n_samples, n_features, n_classes)
    #                            or (n_classes, n_samples, n_features)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[1])          # binary: positive-class slice
    else:
        sv = np.array(shap_values)

    # 3-D from newer shap or multi-output models
    if sv.ndim == 3:
        if sv.shape[-1] == 2:
            sv = sv[:, :, 1]                   # (n_samples, n_features, n_classes) → class 1
        elif sv.shape[0] == 2:
            sv = sv[1]                         # (n_classes, n_samples, n_features) → class 1
        else:
            sv = sv[:, :, -1]                  # best-effort: last slice

    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    cols         = list(X_test.columns)
    mean_signed  = sv.mean(axis=0)       # signed mean — indicates direction
    mean_abs     = np.abs(sv).mean(axis=0)  # magnitude — used for ranking

    all_shap = {
        col: round(float(mean_signed[i]), 4)
        for i, col in enumerate(cols)
    }

    # Rank features by mean absolute SHAP, take top 3
    ranked = sorted(enumerate(cols), key=lambda t: float(mean_abs[t[0]]), reverse=True)
    top3: list[dict] = []
    for feat_idx, col in ranked[:3]:
        signed_val = float(mean_signed[feat_idx])
        abs_val    = float(mean_abs[feat_idx])
        dname      = (display_names or {}).get(col) or _snake_to_title(col)
        top3.append({
            "name":         col,
            "display_name": dname,
            "shap_value":   round(signed_val, 4),
            "abs_shap":     round(abs_val, 4),
            "direction":    "positive" if signed_val >= 0 else "negative",
        })

    return {"top3": top3, "all_shap": all_shap}


def generate_llm_rationale(
    shap_results: dict,
    patient_metadata: dict,
) -> str:
    """Generate a template-based clinical rationale from SHAP results.

    Builds a one-paragraph English explanation from the top-3 SHAP features,
    their directions, and optional patient context.  The output deliberately
    mirrors the style of a real LLM response so this function can be replaced
    by a Llama 3 / GPT-4o API call in Phase 2 with no other code changes.

    Args:
        shap_results:     Output of :func:`generate_shap_explanation` (must contain
                          the ``'top3'`` key).
        patient_metadata: Arbitrary dict providing context.  Recognised keys:

                          * ``'patient_id'``  — shown in the opening clause.
                          * ``'risk_label'``  — defaults to ``'high risk'``.
                          * ``'data_source'`` — appended as a provenance note.

    Returns:
        A human-readable rationale string.  Example::

            "This patient is classified as high risk primarily due to elevated
            REM Sleep % (SHAP: +0.42) and declining HRV Balance (SHAP: +0.31).
            Body Temp Deviation also contributes moderately (SHAP: +0.15).
            [Source: Oura V2 API — model interpretation only, not a clinical diagnosis.]"
    """
    top3       = shap_results.get("top3", [])
    risk_label = patient_metadata.get("risk_label", "high risk")
    data_src   = patient_metadata.get("data_source", "")

    if not top3:
        pid = patient_metadata.get("patient_id", "This patient")
        return (
            f"Insufficient SHAP data to generate a rationale for {pid}. "
            "Please run the experiment with at least one feature selected."
        )

    def _phrase(entry: dict, phrase_idx: int) -> str:
        opts = _DIR_PHRASES[entry["direction"]]
        return opts[phrase_idx % len(opts)]

    def _signed(val: float) -> str:
        return f"+{val:.2f}" if val >= 0 else f"{val:.2f}"

    # ── Sentence construction ────────────────────────────────────────────────
    f0    = top3[0]
    clauses: list[str] = [
        f"This patient is classified as {risk_label} primarily due to "
        f"{_phrase(f0, 0)} {f0['display_name']} (SHAP: {_signed(f0['shap_value'])})"
    ]

    if len(top3) >= 2:
        f1 = top3[1]
        clauses.append(
            f" and {_phrase(f1, 1)} {f1['display_name']} (SHAP: {_signed(f1['shap_value'])})"
        )

    clauses.append(".")

    if len(top3) >= 3:
        f2 = top3[2]
        clauses.append(
            f" {f2['display_name']} also contributes moderately"
            f" (SHAP: {_signed(f2['shap_value'])})."
        )

    if data_src:
        clauses.append(
            f" [Source: {data_src} — model interpretation only,"
            f" not a clinical diagnosis.]"
        )

    return "".join(clauses)


def compute_cognitive_match_score(
    rationale_text: str,
    shap_top3: list[str],
) -> float:
    """Measure how faithfully the rationale text mentions the top SHAP features.

    This is the *hallucination-detection* metric described in the ML for Health
    proposal.  For each entry in *shap_top3* we check whether at least one
    significant token from that feature name appears in *rationale_text*.  The
    score is the fraction of top-3 features that are verifiably mentioned.

    Args:
        rationale_text: Natural-language string produced by
                        :func:`generate_llm_rationale` or a real LLM.
        shap_top3:      Feature identifiers to verify — typically the
                        ``'display_name'`` values from :func:`generate_shap_explanation`.
                        Column (snake_case) names also work.

    Returns:
        Precision in ``[0.0, 1.0]``.

        * ``1.0`` — all top-3 SHAP features are mentioned in the rationale.
        * ``0.0`` — none are mentioned (potential hallucination).

        Returns ``0.0`` when *shap_top3* is empty.

    Examples::

        >>> compute_cognitive_match_score(
        ...     "Risk driven by elevated HRV Balance and declining REM Sleep %.",
        ...     ["HRV Balance", "REM Sleep %", "Body Temp Deviation"],
        ... )
        0.6667   # 2 of 3 features mentioned
    """
    if not shap_top3:
        return 0.0

    text_lower = rationale_text.lower()
    n_matched  = sum(
        1 for feat in shap_top3 if _feature_is_mentioned(feat, text_lower)
    )
    return round(n_matched / len(shap_top3), 4)


def build_clinical_rationale(
    model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    patient_metadata: dict,
    *,
    display_names: dict[str, str] | None = None,
) -> ClinicalRationale:
    """Run the full explanation pipeline and return a :class:`ClinicalRationale`.

    Calls :func:`generate_shap_explanation` → :func:`generate_llm_rationale` →
    :func:`compute_cognitive_match_score` and assembles the result into a single
    dataclass, making it easy to pass around and serialise.

    Args:
        model:            Fitted tree-based estimator.
        X_test:           Feature matrix used for SHAP computation.
        feature_names:    Column names matching the model's input.
        patient_metadata: Passed verbatim to :func:`generate_llm_rationale`.
        display_names:    Optional ``{column_name: display_label}`` map.

    Returns:
        A fully populated :class:`ClinicalRationale`.
    """
    shap_results   = generate_shap_explanation(
        model, X_test, feature_names, display_names=display_names
    )
    rationale      = generate_llm_rationale(shap_results, patient_metadata)
    top_display    = [e["display_name"] for e in shap_results["top3"]]
    cog_score      = compute_cognitive_match_score(rationale, top_display)

    return ClinicalRationale(
        shap_values=shap_results["all_shap"],
        top_features=top_display,
        rationale_text=rationale,
        cognitive_match_score=cog_score,
        confidence=_confidence_label(cog_score),
    )
