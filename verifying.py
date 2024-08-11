import tensorflow as tf
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Initialize a tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

print("Hugging Face Transformers installed and working!")
