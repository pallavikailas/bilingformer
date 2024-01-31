import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, include_emotion_labels=True, include_trigger_labels=False):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_emotion_labels = include_emotion_labels
        self.include_trigger_labels = include_trigger_labels and 'triggers' in dataframe.columns

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        utterance = str(self.dataframe.iloc[idx]['utterances'])
        tokens = self.tokenizer.encode_plus(
            utterance,
            max_length=self.max_length,  
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        decoder_input_ids = input_ids.clone()
        attention_mask = tokens['attention_mask'].squeeze()

        item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids
        }

        if self.include_emotion_labels:
            emotion = self.dataframe.iloc[idx]['labels']
            item['labels'] = torch.tensor(emotion, dtype=torch.long)
    
        if self.include_trigger_labels:
            trigger = self.dataframe.iloc[idx]['triggers']
            try:
                trigger_numeric = int(float(trigger))  # Handles string representations of floats and integers
            except ValueError:
                trigger_numeric = 1 
            item['trigger_labels'] = torch.tensor(trigger_numeric, dtype=torch.long)

        return item
