import math
import spacy
import nltk
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

analyzer = SentimentIntensityAnalyzer()

#Readable entity type names

ENTITY_TYPE_LABELS = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Place",
    "LOC": "Location",
    "DATE": "Date/Time",
    "TIME": "Time",
    "MONEY": "Money",
    "CARDINAL": "Number",
    "ORDINAL": "Ordinal",
    "NORP": "Group/Nationality",
    "QUANTITY": "Measurement",
    "PERCENT": "Percentage",
    "FAC": "Facility",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "WORK_OF_ART": "Creative Work",
    "LAW": "Law",
    "LANGUAGE": "Language",
}

#Hedge & certainty lexicons

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "seems", "apparently", "somewhat", "sort of", "kind of",
    "i think", "i guess", "i suppose", "not sure", "i believe",
}

CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "clearly", "obviously",
    "always", "never", "must", "undoubtedly", "sure", "exactly",
    "no doubt",
}


def readable_entity_type(spacy_label):
    """Convert spaCy entity label to plain English."""
    return ENTITY_TYPE_LABELS.get(spacy_label, spacy_label)


def _compute_syntactic_depth(doc):
    depths = []
    for sent in doc.sents:
        root = sent.root

        def _depth(token):
            children = list(token.children)
            if not children:
                return 1
            return 1 + max(_depth(c) for c in children)

        depths.append(_depth(root))

    return sum(depths) / len(depths) if depths else 0.0


def _compute_formality(doc):
    pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)
    total = sum(pos_counts.values())

    if total == 0:
        return 50.0

    formal = (
        pos_counts.get("NOUN", 0)
        + pos_counts.get("ADJ", 0)
        + pos_counts.get("ADP", 0)
        + pos_counts.get("DET", 0)
    )
    informal = (
        pos_counts.get("PRON", 0)
        + pos_counts.get("VERB", 0)
        + pos_counts.get("ADV", 0)
        + pos_counts.get("INTJ", 0)
    )

    return ((formal - informal) / total * 100 + 100) / 2


def _detect_questions_syntactic(doc):
    count = 0
    for sent in doc.sents:
        text = sent.text.strip()

        has_wh = any(
            token.tag_ in ("WDT", "WP", "WP$", "WRB")
            for token in sent
        )

        tokens = [t for t in sent if t.is_alpha]
        has_aux_inversion = (
            len(tokens) >= 2
            and tokens[0].pos_ in ("AUX", "VERB")
            and any(t.dep_ == "nsubj" for t in tokens[1:])
        )

        if has_wh or has_aux_inversion or "?" in text:
            count += 1

    return count


def _count_hedges_and_certainty(text):
    text_lower = text.lower()
    hedges = sum(1 for h in HEDGE_WORDS if h in text_lower)
    certainty = sum(1 for c in CERTAINTY_WORDS if c in text_lower)
    return hedges, certainty


def _extract_entities(doc):
    """Extract named entities with readable type labels."""
    return [
        (ent.text, ent.label_, readable_entity_type(ent.label_))
        for ent in doc.ents
    ]


#Speaker-level stats

def _empty_stats():
    return {
        "messages": 0,
        "words": 0,
        "avg_length": 0,
        "unique_words": set(),
        "vocab_richness": 0,

        "positive": 0,
        "negative": 0,
        "neutral": 0,
        "avg_sentiment": 0.0,

        "questions": 0,
        "avg_tree_depth": 0.0,
        "formality": 0.0,

        "hedges": 0,
        "certainty_markers": 0,

        "entities": [],
        "entity_count": 0,
        "entity_types": Counter(),       # raw spaCy labels
        "entity_types_readable": Counter(),  # plain English labels

        "gt_acts": Counter(),
        "gt_emotions": Counter(),
    }


def get_speaker_stats(conversation):
    stats = {"A": _empty_stats(), "B": _empty_stats()}

    sentiment_scores = {"A": [], "B": []}
    tree_depths = {"A": [], "B": []}
    formality_scores = {"A": [], "B": []}

    for msg in conversation:
        speaker = msg["speaker"]
        text = msg["text"]
        s = stats[speaker]

        doc = nlp(text)

        words = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and token.text.lower() not in stop_words
        ]

        s["messages"] += 1
        s["words"] += len(words)
        s["unique_words"].update(words)

        s["questions"] += _detect_questions_syntactic(doc)

        depth = _compute_syntactic_depth(doc)
        tree_depths[speaker].append(depth)

        formality = _compute_formality(doc)
        formality_scores[speaker].append(formality)

        hedges, certainty = _count_hedges_and_certainty(text)
        s["hedges"] += hedges
        s["certainty_markers"] += certainty

        ents = _extract_entities(doc)
        s["entities"].extend(ents)
        s["entity_count"] += len(ents)
        for _, raw_label, readable_label in ents:
            s["entity_types"][raw_label] += 1
            s["entity_types_readable"][readable_label] += 1

        compound = analyzer.polarity_scores(text)["compound"]
        sentiment_scores[speaker].append(compound)

        if compound >= 0.05:
            s["positive"] += 1
        elif compound <= -0.05:
            s["negative"] += 1
        else:
            s["neutral"] += 1

        if "gt_act" in msg:
            s["gt_acts"][msg["gt_act"]] += 1
        if "gt_emotion" in msg and msg["gt_emotion"] != "none":
            s["gt_emotions"][msg["gt_emotion"]] += 1

    # Finalize
    for speaker in stats:
        s = stats[speaker]

        if s["messages"] > 0:
            s["avg_length"] = s["words"] / s["messages"]

        if s["words"] > 0:
            s["vocab_richness"] = len(s["unique_words"]) / math.sqrt(s["words"])

        if sentiment_scores[speaker]:
            s["avg_sentiment"] = sum(sentiment_scores[speaker]) / len(
                sentiment_scores[speaker]
            )

        if tree_depths[speaker]:
            s["avg_tree_depth"] = sum(tree_depths[speaker]) / len(
                tree_depths[speaker]
            )

        if formality_scores[speaker]:
            s["formality"] = sum(formality_scores[speaker]) / len(
                formality_scores[speaker]
            )

    return stats


def compare_speakers(stats):
    total_messages = stats["A"]["messages"] + stats["B"]["messages"]

    if total_messages == 0:
        return {"A_dominance": 0, "B_dominance": 0}

    return {
        "A_dominance": stats["A"]["messages"] / total_messages,
        "B_dominance": stats["B"]["messages"] / total_messages,
    }


#Population-level stats

FEATURES = [
    "messages", "words", "avg_length", "questions",
    "vocab_richness", "positive", "negative", "neutral",
    "avg_sentiment", "avg_tree_depth", "formality",
    "hedges", "certainty_markers", "entity_count",
]

# Readable names for population stats display
FEATURE_DISPLAY_NAMES = {
    "messages": "Messages per speaker",
    "words": "Content words",
    "avg_length": "Words per message",
    "questions": "Questions asked",
    "vocab_richness": "Vocabulary diversity",
    "positive": "Positive utterances",
    "negative": "Negative utterances",
    "neutral": "Neutral utterances",
    "avg_sentiment": "Avg sentiment score",
    "avg_tree_depth": "Sentence complexity",
    "formality": "Formality (0-100)",
    "hedges": "Hedge words",
    "certainty_markers": "Certainty words",
    "entity_count": "Named entities",
}


def collect_all_speaker_stats(all_conversations):
    all_stats = []
    all_pairs = []

    total = len(all_conversations)
    for idx, convo in enumerate(all_conversations):
        if idx % 500 == 0:
            print(f"  Processing conversation {idx}/{total}...")

        stats = get_speaker_stats(convo)
        comparison = compare_speakers(stats)
        all_pairs.append((stats, comparison))

        for speaker in ["A", "B"]:
            if stats[speaker]["messages"] > 0:
                all_stats.append(stats[speaker])

    return all_stats, all_pairs


def compute_population_stats(all_stats):
    population = {}

    for feat in FEATURES:
        values = [s[feat] for s in all_stats]

        mean = float(np.mean(values))
        std = float(np.std(values))

        if std == 0:
            std = 1.0

        population[feat] = {"mean": mean, "std": std}

    return population


def compute_zscore(value, mean, std):
    return (value - mean) / std