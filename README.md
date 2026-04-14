## Dialogue Dataset Analyzer

A Python-based NLP pipeline that extracts persona profiles and behavioral traits from multi-turn conversational datasets. Built to analyze how people communicate - their sentiment patterns, vocabulary, formality, and conversational role - and automatically generate structured persona profiles for each speaker.

Tested on the [DailyDialog](http://yanran.li/dailydialog) dataset (11,118 conversations).

---

## What It Does

The pipeline processes dialogue datasets in two passes:

**Pass 1 - Population Baseline:** Computes dataset-wide averages across all 11,118 conversations to establish what "normal" speaker behavior looks like.

**Pass 2 - Conversation Analysis:** Analyzes individual conversations, extracts per-speaker features, benchmarks them against the population baseline, and generates automated persona profiles.

### Features Extracted Per Speaker

**Volume:** Messages, word count, avg words/message

**Engagement:** Questions asked, turn balance (dominance %)

**Vocabulary:** Vocabulary diversity (unique/total ratio)

**Sentiment:** Positive/negative/neutral counts, avg sentiment score (VADER)

**Style:** Sentence complexity (dependency tree depth), formality score (0-100)

**Confidence:** Hedge words, certainty markers

**Content:** Named entities (people, places, orgs, dates) via spaCy NER

---

## Automated Persona Generation

Each speaker is classified across six dimensions based on their extracted features compared to population norms:

- **Communication style:** concise / moderate / articulate / expressive
- **Engagement level:** passive / moderately engaged / highly engaged
- **Emotional tone:** negative / neutral / positive / mixed emotions
- **Speech style:** casual / balanced / formal / tentative
- **Conversational role:** balanced / dominant / supportive
- **Personality traits:** curious, analytical, critical, enthusiastic, assertive, withdrawn, etc.

---

## Sample Output

### Terminal Output

```
============================================================
  DATASET BASELINE (what 'average' looks like)
============================================================
  Messages per speaker       avg:   3.92   spread: 2.03
  Content words              avg:  20.18   spread: 16.63
  Words per message          avg:   5.08   spread: 3.37
  Questions asked            avg:   1.86   spread: 1.75
  Vocabulary diversity       avg:   3.72   spread: 1.31
  Positive utterances        avg:   1.96   spread: 1.52
  Negative utterances        avg:   0.51   spread: 0.80
  Neutral utterances         avg:   1.45   spread: 1.33
  Avg sentiment score        avg:   0.18   spread: 0.23
  Sentence complexity        avg:   3.59   spread: 0.86
  Formality (0-100)          avg:  43.76   spread: 12.69
  Hedge words                avg:   0.37   spread: 0.73
  Certainty words            avg:   0.28   spread: 0.57
  Named entities             avg:   1.69   spread: 2.17


============================================================
  CONVERSATION 1
============================================================

  Dialogue:
  ------------------------------------------------------
    A: Say , Jim , how about going for a few beers after dinner ?
    B: You know that is tempting but is really not good for our fitness .
    A: What do you mean ? It will help us to relax .
    B: Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ?
    A: I guess you are right.But what shall we do ? I don't feel like sitting at home .
    B: I suggest a walk over to the gym where we can play singsong and meet some of our friends .
    A: That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them .
    B: Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too .
    A: Good.Let ' s go now .
    B: All right .

  Stats:
  ──────────────────────────────────────────────────────
                                  Speaker A   Speaker B
  ──────────────────────────────────────────────────────
  Messages                                5           5
  Words                                  31          34
  Avg words/message                     6.2         6.8
  Questions asked                         4           3
  Vocabulary diversity                 5.03        5.49
  ──────────────────────────────────────────────────────
  Sentiment                     
    Positive                              2           3
    Negative                              1           1
    Neutral                               2           1
    Avg score                         +0.21       +0.30
  ──────────────────────────────────────────────────────
  Sentence complexity                   3.7         4.9
  Formality (0-100)                      37          33
  Hedge words                             2           1
  Certainty words                         0           0
  ──────────────────────────────────────────────────────
  Mentions                      
    Speaker A                    Jim (Person), Mary (Person), Sally (Person)
    Speaker B                    -
  ──────────────────────────────────────────────────────
  Turn balance                        A 50%       B 50%
  ──────────────────────────────────────────────────────

  Persona A:
  ------------------------------------------------------
    Communication:  moderate
    Engagement:     highly engaged
    Emotional tone: neutral
    Speech style:   casual
    Role:           balanced
    Personality:    curious
    Talks about:   3 person

  Persona B:
  ------------------------------------------------------
    Communication:  articulate
    Engagement:     highly engaged
    Emotional tone: mixed emotions
    Speech style:   casual
    Role:           balanced
    Personality:    articulate, expressive, curious

```

### HTML Report

The pipeline also generates a styled HTML report with:
- Dataset baseline statistics
- Personality trait frequency charts
- Formality vs. complexity scatter plots
- Sentiment breakdowns per conversation
- Per-conversation analysis with side-by-side speaker stats and persona cards

Run `python report.py` and open `reports/report.html` in a browser to view.

---

## How It Works

1. **Preprocessing** (`preprocess.py`): Loads the DailyDialog CSV, parses multi-turn dialogues, and structures them as speaker-labeled message sequences.

2. **Feature Extraction** (`feature_extraction.py`): For each speaker in a conversation, computes 14 behavioral features covering volume, sentiment, vocabulary, style, and content. Also runs a full pass across all 11,118 conversations to compute population-level baselines (mean and standard deviation for each feature).

3. **Persona Generation** (`persona.py`): Compares each speaker's features against population baselines to classify them along six persona dimensions. A speaker with vocabulary diversity 2 standard deviations above the mean gets labeled "expressive"; one with high question count and high sentiment gets "curious" and "enthusiastic."

4. **Reporting** (`report.py`): Generates a styled HTML report with dataset baseline tables, data visualizations, and per-conversation breakdowns with persona cards.

---


## Tech Stack

- **Python**
- **NLTK** - VADER sentiment analysis, tokenization
- **spaCy** - Dependency parsing, named entity recognition, sentence complexity
- **Pandas** - Data processing and aggregation
- **Matplotlib** - Chart generation for HTML reports

---

## How to Run

```bash
git clone https://github.com/zuman989/Dialogue-Dataset-Analyzer.git
cd Dialogue-Dataset-Analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the analysis:
```bash
python main.py
```

Generate reports:
```bash
python report.py
```

Reports are saved to the `reports/` directory.

---

## Author

**Zuman** - [GitHub](https://github.com/zuman989) | [LinkedIn](https://www.linkedin.com/in/zuman-9ba8922a4/) | [Behance](https://www.behance.net/zuman1)
