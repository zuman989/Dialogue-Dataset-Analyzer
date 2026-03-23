import pandas as pd
import re


# dailyDialog label mappings 

ACT_LABELS = {
    1: "statement",
    2: "question",
    3: "suggestion/request",
    4: "commitment/promise",
}

EMOTION_LABELS = {
    0: "none",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise",
}


def load_data(file_path):
    return pd.read_csv(file_path)


def parse_dialog(dialog_str):
    matches = re.findall(r"'(.*?)'|\"(.*?)\"", dialog_str)

    dialog_list = []
    for m in matches:
        text = m[0] if m[0] else m[1]
        if text:
            dialog_list.append(text)

    return dialog_list


def parse_label_sequence(label_str):
    """Parse '[3 4 2 2]' → [3, 4, 2, 2]"""
    return [int(x) for x in re.findall(r"\d+", str(label_str))]


def preprocess_dialogs(df):
    processed_data = []

    has_acts = "act" in df.columns
    has_emotions = "emotion" in df.columns

    for _, row in df.iterrows():
        dialog_list = parse_dialog(row["dialog"])

        acts = parse_label_sequence(row["act"]) if has_acts else []
        emotions = parse_label_sequence(row["emotion"]) if has_emotions else []

        structured_dialog = []
        speaker = "A"

        for idx, msg in enumerate(dialog_list):
            clean_msg = msg.strip()

            if clean_msg:
                entry = {
                    "speaker": speaker,
                    "text": clean_msg,
                }

                if idx < len(acts):
                    entry["gt_act"] = ACT_LABELS.get(acts[idx], "unknown")
                if idx < len(emotions):
                    entry["gt_emotion"] = EMOTION_LABELS.get(emotions[idx], "none")

                structured_dialog.append(entry)
                speaker = "B" if speaker == "A" else "A"

        processed_data.append(structured_dialog)

    return processed_data