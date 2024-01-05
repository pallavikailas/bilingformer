import json
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    utterances, emotions = [], []
    for item in data:
        utterances.extend(item['utterances'])
        emotions.extend(item['emotions'])
    
    emotion_to_label = {
        'neutral': 0, 
        'joy': 1, 
        'contempt': 2, 
        'anger': 3,
        'surprise': 4,
        'fear': 5, 
        'disgust': 6,
        'sadness': 7
    }
    labels = [emotion_to_label[emotion] for emotion in emotions]
    df = pd.DataFrame({'utterances': utterances, 'labels': labels})

    return train_test_split(df, test_size=0.2, random_state=42)
