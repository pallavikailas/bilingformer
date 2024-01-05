import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initialize_model(num_labels):
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    return model, optimizer, criterion

def train_and_evaluate(model, train_dl, val_dl, device, optimizer, criterion, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

        average_loss = total_loss / len(train_dl)
        train_accuracy = total_correct / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds)

        accuracy = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")

def save_model(model, path):
    model.save_pretrained(path)
