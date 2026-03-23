from src.persona import generate_persona
from src.preprocess import load_data, preprocess_dialogs
from src.feature_extraction import (
    get_speaker_stats,
    compare_speakers,
    collect_all_speaker_stats,
    compute_population_stats,
    FEATURE_DISPLAY_NAMES,
    FEATURES,
)


df = load_data("data/train.csv")
processed = preprocess_dialogs(df)

#Pass 1: Population stats

print("Computing population stats across full dataset...\n")
all_stats, all_pairs = collect_all_speaker_stats(processed)
population = compute_population_stats(all_stats)

print("\n" + "=" * 60)
print("  DATASET BASELINE (what 'average' looks like)")
print("=" * 60)

for feat in FEATURES:
    name = FEATURE_DISPLAY_NAMES.get(feat, feat)
    vals = population[feat]
    print(f"  {name:25s}  avg: {vals['mean']:6.2f}   spread: {vals['std']:.2f}")

# format entity mentions as a short string

def _mentions_str(s):
    if not s['entities']:
        return "-"
    seen = set()
    parts = []
    for text, _, readable in s['entities']:
        if text not in seen:
            seen.add(text)
            parts.append(f"{text} ({readable})")
    return ", ".join(parts[:3])  # max 3 to fit in table


# Pass 2: Analyze first 10 conversations

conversations = processed[:10]

for i, convo in enumerate(conversations):
    print(f"\n\n{'=' * 60}")
    print(f"  CONVERSATION {i+1}")
    print(f"{'=' * 60}")

    # Dialogue

    print("\n  Dialogue:")
    print(f"  {'-' * 54}")

    for msg in convo:
        print(f"    {msg['speaker']}: {msg['text']}")

    stats = get_speaker_stats(convo)
    comparison = compare_speakers(stats)
    a = stats["A"]
    b = stats["B"]

    # Stats Table

    print(f"\n  Stats:")
    print(f"  {'─' * 54}")
    print(f"  {'':30s} {'Speaker A':>10s}  {'Speaker B':>10s}")
    print(f"  {'─' * 54}")

    print(f"  {'Messages':30s} {a['messages']:>10}  {b['messages']:>10}")
    print(f"  {'Words':30s} {a['words']:>10}  {b['words']:>10}")
    print(f"  {'Avg words/message':30s} {a['avg_length']:>10.1f}  {b['avg_length']:>10.1f}")
    print(f"  {'Questions asked':30s} {a['questions']:>10}  {b['questions']:>10}")
    print(f"  {'Vocabulary diversity':30s} {a['vocab_richness']:>10.2f}  {b['vocab_richness']:>10.2f}")

    print(f"  {'─' * 54}")
    print(f"  {'Sentiment':30s}")
    print(f"  {'  Positive':30s} {a['positive']:>10}  {b['positive']:>10}")
    print(f"  {'  Negative':30s} {a['negative']:>10}  {b['negative']:>10}")
    print(f"  {'  Neutral':30s} {a['neutral']:>10}  {b['neutral']:>10}")
    print(f"  {'  Avg score':30s} {a['avg_sentiment']:>+10.2f}  {b['avg_sentiment']:>+10.2f}")

    print(f"  {'─' * 54}")
    print(f"  {'Sentence complexity':30s} {a['avg_tree_depth']:>10.1f}  {b['avg_tree_depth']:>10.1f}")
    print(f"  {'Formality (0-100)':30s} {a['formality']:>10.0f}  {b['formality']:>10.0f}")
    print(f"  {'Hedge words':30s} {a['hedges']:>10}  {b['hedges']:>10}")
    print(f"  {'Certainty words':30s} {a['certainty_markers']:>10}  {b['certainty_markers']:>10}")

    print(f"  {'─' * 54}")
    print(f"  {'Mentions':30s}")
    print(f"  {'  Speaker A':30s} {_mentions_str(a)}")
    print(f"  {'  Speaker B':30s} {_mentions_str(b)}")

    print(f"  {'─' * 54}")
    a_turn = f"A {comparison['A_dominance']:.0%}"
    b_turn = f"B {comparison['B_dominance']:.0%}"
    print(f"  {'Turn balance':30s} {a_turn:>10s}  {b_turn:>10s}")
    print(f"  {'─' * 54}")

    # Persona
    persona = generate_persona(stats, comparison, population)

    for spk in ["A", "B"]:
        p = persona[spk]
        print(f"\n  Persona {spk}:")
        print(f"  {'-' * 54}")
        print(f"    Communication:  {p['style']}")
        print(f"    Engagement:     {p['engagement']}")
        print(f"    Emotional tone: {p['tone']}")
        print(f"    Speech style:   {p['speech_style']}")
        print(f"    Role:           {p['role']}")
        print(f"    Personality:    {', '.join(p['traits'])}")

        if p['topics']:
            types_str = ", ".join(
                f"{count} {label.lower()}"
                for label, count in p['topics']['types'].items()
            )
            print(f"    Talks about:   {types_str}")