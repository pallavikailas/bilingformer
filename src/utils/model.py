import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

class T5ForBinaryClassification(nn.Module):
    def __init__(self, pretrained_model=model_path):
        super().__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)  # Load the pre-trained T5 model
        self.classifier = nn.Linear(self.t5.config.d_model, 1)  # Classifier for binary classification
        self.pre_classifier = nn.Linear(self.t5.config.vocab_size, self.t5.config.d_model)  # Pre-classifier layer

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        first_token_logits = outputs.logits[:, 0, :]  # Get the logits for the first token
        transformed_logits = self.pre_classifier(first_token_logits)  # Transform the logits
        return self.classifier(transformed_logits)  # Return the final classification logits

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Alpha parameter for focal loss
        self.gamma = gamma  # Gamma parameter for focal loss

    def forward(self, inputs, targets):
        inputs = inputs.squeeze()  # Squeeze inputs to remove any extra dimensions
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # Calculate pt as per the focal loss formula
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss  # Calculate the final focal loss
        return F_loss.mean()  # Return the mean focal loss

