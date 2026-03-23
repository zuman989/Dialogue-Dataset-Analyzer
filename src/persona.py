from src.feature_extraction import compute_zscore

# ────────────────────────────────────────────────────────
# TRAIT RECIPES
# ────────────────────────────────────────────────────────

TRAIT_RECIPES = {
    "expressive": {
        "avg_length":        0.30,
        "words":             0.20,
        "messages":          0.15,
        "vocab_richness":    0.10,
        "avg_tree_depth":    0.15,
        "entity_count":      0.10,
    },

    "curious": {
        "questions":         0.50,
        "messages":          0.15,
        "avg_length":        0.15,
        "hedges":            0.20,
    },

    "enthusiastic": {
        "positive":          0.40,
        "negative":         -0.20,
        "avg_sentiment":     0.20,
        "certainty_markers": 0.10,
        "avg_length":        0.10,
    },

    "critical": {
        "negative":          0.40,
        "positive":         -0.20,
        "avg_sentiment":    -0.20,
        "questions":         0.10,
        "certainty_markers": 0.10,
    },

    "withdrawn": {
        "messages":         -0.30,
        "words":            -0.25,
        "avg_length":       -0.15,
        "questions":        -0.10,
        "entity_count":     -0.10,
        "hedges":           -0.10,
    },

    "articulate": {
        "vocab_richness":    0.35,
        "avg_length":        0.20,
        "avg_tree_depth":    0.25,
        "formality":         0.10,
        "words":             0.10,
    },

    "reserved": {
        "messages":         -0.25,
        "questions":        -0.20,
        "neutral":           0.25,
        "hedges":            0.15,
        "formality":         0.15,
    },

    "assertive": {
        "messages":          0.25,
        "words":             0.20,
        "avg_length":        0.15,
        "certainty_markers": 0.20,
        "formality":        -0.10,
        "hedges":           -0.10,
    },

    "analytical": {
        "avg_tree_depth":    0.30,
        "formality":         0.25,
        "vocab_richness":    0.20,
        "entity_count":      0.15,
        "hedges":            0.10,
    },
}

ACTIVATION_THRESHOLD = 0.5


def compute_trait_score(speaker_stats, population, trait_weights):
    score = 0.0

    for feature, weight in trait_weights.items():
        if feature not in population:
            continue
        z = compute_zscore(
            speaker_stats[feature],
            population[feature]["mean"],
            population[feature]["std"],
        )
        score += weight * z

    return score


def _classify_style(zscores):
    complexity = (
        zscores.get("avg_length", 0) * 0.4
        + zscores.get("vocab_richness", 0) * 0.3
        + zscores.get("avg_tree_depth", 0) * 0.3
    )

    if complexity > 1.0:
        return "articulate"
    elif complexity > 0.5:
        return "expressive"
    elif complexity < -0.5:
        return "concise"
    else:
        return "moderate"


def _classify_engagement(zscores):
    engagement_z = (
        zscores.get("messages", 0) * 0.4
        + zscores.get("words", 0) * 0.3
        + zscores.get("questions", 0) * 0.3
    )

    if engagement_z > 0.5:
        return "highly engaged"
    elif engagement_z > -0.5:
        return "moderately engaged"
    else:
        return "passive"


def _classify_tone(zscores):
    pos_z = zscores.get("positive", 0)
    neg_z = zscores.get("negative", 0)
    avg_z = zscores.get("avg_sentiment", 0)

    if avg_z > 0.5 and neg_z <= 0:
        return "positive"
    elif avg_z < -0.5 and pos_z <= 0:
        return "negative"
    elif pos_z > 0.5 and neg_z > 0.5:
        return "mixed emotions"
    else:
        return "neutral"


def _classify_register(zscores):
    f_z = zscores.get("formality", 0)
    h_z = zscores.get("hedges", 0)

    if f_z > 0.5:
        return "formal"
    elif f_z < -0.5:
        return "casual"
    elif h_z > 0.5:
        return "tentative"
    else:
        return "balanced"


def _summarize_ground_truth(stats):
    gt = {}

    acts = stats.get("gt_acts", {})
    if acts:
        total = sum(acts.values())
        gt["speech_acts"] = {
            act: f"{count} ({count/total*100:.0f}%)"
            for act, count in acts.most_common()
            if act != "unknown"
        }
        dominant = acts.most_common(1)
        if dominant:
            gt["primary_role"] = dominant[0][0]

    emotions = stats.get("gt_emotions", {})
    if emotions:
        gt["emotions_expressed"] = dict(
            sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        )

    return gt if gt else None


def _summarize_entities(stats):
    if stats["entity_count"] == 0:
        return None

    top_types = stats["entity_types_readable"].most_common(3)

    # Deduplicate and use readable labels
    seen = set()
    unique_ents = []
    for text, _, readable in stats["entities"]:
        if text not in seen:
            seen.add(text)
            unique_ents.append(f"{text} ({readable})")
    unique_ents = unique_ents[:5]

    return {
        "total": stats["entity_count"],
        "types": {label: count for label, count in top_types},
        "examples": unique_ents,
    }


def generate_persona(stats, comparison, population):
    persona = {"A": {}, "B": {}}

    for speaker in ["A", "B"]:
        s = stats[speaker]

        # Compute z-scores
        zscores = {}
        for feat in population:
            zscores[feat] = compute_zscore(
                s[feat],
                population[feat]["mean"],
                population[feat]["std"],
            )

        # Classify dimensions
        style = _classify_style(zscores)
        engagement = _classify_engagement(zscores)
        tone = _classify_tone(zscores)
        register = _classify_register(zscores)

        dom = comparison[f"{speaker}_dominance"]
        if dom > 0.6:
            role = "dominant"
        elif dom < 0.4:
            role = "passive"
        else:
            role = "balanced"

        # Personality traits
        trait_scores = {}
        for trait_name, weights in TRAIT_RECIPES.items():
            trait_scores[trait_name] = compute_trait_score(s, population, weights)

        traits = [
            trait
            for trait, score in trait_scores.items()
            if score >= ACTIVATION_THRESHOLD
        ]
        traits.sort(key=lambda t: trait_scores[t], reverse=True)

        if not traits:
            best = max(trait_scores, key=trait_scores.get)
            traits.append(best)

        # Content
        entity_summary = _summarize_entities(s)

        persona[speaker] = {
            "style": style,
            "engagement": engagement,
            "tone": tone,
            "speech_style": register,
            "role": role,
            "traits": traits,
            "topics": entity_summary,
            "_trait_scores": {k: round(v, 2) for k, v in trait_scores.items()},
            "_zscores": {k: round(v, 2) for k, v in zscores.items()},
        }

    return persona