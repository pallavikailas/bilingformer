import json
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def prepare_data(data, emotion_to_label, trigger_to_label):
    utterances = []
    emotions = []
    triggers = []
    for item in data:
        utterances.extend(item['utterances'])
        emotions.extend(item['emotions'])
        triggers.extend(item['triggers'])
    triggers = [convert_to_int_or_nan(t) for t in triggers]
    labels = [emotion_to_label[emotion] for emotion in emotions]
    trigger_labels = [trigger_to_label.get(t, t) for t in triggers]
    return pd.DataFrame({'utterances': utterances, 'labels': labels, 'triggers': trigger_labels})

def convert_to_int_or_nan(t):
    if not pd.isna(t):
        try:
            return int(t)
        except ValueError:
            return 'nan'
    return 'nan'
