import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM

class T5ForBinaryClassification(nn.Module):
    def __init__(self, pretrained_model=model_path):
        super().__init__()
        self.t5 = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.classifier = torch.nn.Linear(self.t5.config.d_model, 1)
        # Additional linear layer to transform the dimensions
        self.pre_classifier = torch.nn.Linear(self.t5.config.vocab_size, self.t5.config.d_model)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        # Extract the logits of the first token and transform dimensions
        first_token_logits = outputs.logits[:, 0, :]
        transformed_logits = self.pre_classifier(first_token_logits)
        return self.classifier(transformed_logits)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.squeeze() 
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
