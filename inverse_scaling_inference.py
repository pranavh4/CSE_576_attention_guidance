from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import torch
from utils.dataset_utils import load_dataset, encode_inverse_scaling_dataset
from utils.inference_utils import calculate_classification_accuracy, calculate_sequence_loss

MODEL_NAME = "gpt2"
MODEL_PATH = "checkpoints/checkpoint-137746"
INVERSE_SCALING_DATASET_DIR = "./dataset/inverse_scaling"
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)

print("\nLoading Datasets")
print("----------------")
classification_datasets = [f for f in os.listdir(INVERSE_SCALING_DATASET_DIR) if "classification" in f]
encoded_datasets = {}
for dataset_name in classification_datasets:
    print(f"Loaded {dataset_name}")
    dataset = load_dataset(os.path.join(INVERSE_SCALING_DATASET_DIR, dataset_name))
    encoded_dataset = encode_inverse_scaling_dataset(dataset, tokenizer)
    encoded_datasets[dataset_name] = encoded_dataset

print("\nClassification Accuracies")
print("-------------------------")
for dataset_name in encoded_datasets.keys():
    prediction_losses = calculate_sequence_loss(model, device, encoded_datasets[dataset_name], batch_size=BATCH_SIZE)
    print(f"{dataset_name}: {calculate_classification_accuracy(encoded_datasets[dataset_name], prediction_losses)}")
