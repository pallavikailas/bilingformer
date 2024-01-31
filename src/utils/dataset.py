import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, include_emotion_labels=True, include_trigger_labels=False):
        self.dataframe = dataframe  # DataFrame containing the data
        self.tokenizer = tokenizer  # Tokenizer for encoding the utterances
        self.max_length = max_length  # Maximum length of the tokenized sequences
        self.include_emotion_labels = include_emotion_labels  # Whether to include emotion labels
        self.include_trigger_labels = include_trigger_labels  # Whether to include trigger labels

    def __len__(self):
        return len(self.dataframe)  # Return the length of the dataframe

    def __getitem__(self, idx):
        # Retrieve an item by its index
        utterance = str(self.dataframe.iloc[idx]['utterances'])
        tokens = self.tokenizer.encode_plus(
            utterance, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()  # Token IDs
        decoder_input_ids = input_ids.clone()  # Clone input IDs for the decoder
        attention_mask = tokens['attention_mask'].squeeze()  # Attention mask

        item = {'input_ids': input_ids, 'attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids}

        if self.include_emotion_labels:
            emotion = self.dataframe.iloc[idx]['labels']
            item['labels'] = torch.tensor(emotion, dtype=torch.long)  # Emotion labels

        if self.include_trigger_labels:
            trigger = self.dataframe.iloc[idx]['triggers']
            try:
                trigger_numeric = int(float(trigger))  # Convert triggers to numeric
            except ValueError:
                trigger_numeric = 1  # Default trigger value
            item['trigger_labels'] = torch.tensor(trigger_numeric, dtype=torch.long)  # Trigger labels

        return item

