import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import readline  # For better CLI input handling


class StreamingTextDataset(IterableDataset):
    """Memory efficient dataset for large text files"""

    def __init__(self, data_path: str, vocab: Dict[str, int], max_len: int):
        self.data_path = Path(data_path)
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = 0

    def __iter__(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    # Tokenize and convert to indices
                    words = line.strip().split()[: self.max_len]
                    indices = [
                        self.vocab.get(word, self.vocab["<UNK>"]) for word in words
                    ]

                    # Pad sequence
                    if len(indices) < self.max_len:
                        indices = indices + [self.pad_token] * (
                            self.max_len - len(indices)
                        )

                    tensor = torch.tensor(indices)
                    yield tensor, tensor


class SequenceCompressor(nn.Module):
    """Neural model for one-shot sequence compression and generation"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 512,
        latent_size: int = 256,
        max_len: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_len = max_len

        # Embedding with position encoding
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, hidden_size))

        # Encoder layers
        encoder_layers = []
        curr_size = hidden_size * max_len
        for _ in range(num_layers):
            encoder_layers.extend(
                [
                    nn.Linear(curr_size, curr_size // 2),
                    nn.LayerNorm(curr_size // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            curr_size = curr_size // 2
        encoder_layers.append(nn.Linear(curr_size, latent_size))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers
        decoder_layers = []
        curr_size = latent_size
        for _ in range(num_layers):
            decoder_layers.extend(
                [
                    nn.Linear(curr_size, curr_size * 2),
                    nn.LayerNorm(curr_size * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            curr_size = curr_size * 2
        self.decoder = nn.Sequential(*decoder_layers)

        self.out = nn.Linear(hidden_size, vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Add positional encoding
        embedded = self.embed(x) + self.pos_encoding
        flat = embedded.view(batch_size, -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        hidden = self.decoder(z)
        hidden = hidden.view(batch_size, self.max_len, self.hidden_size)
        return self.out(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def build_vocabulary(
    data_path: str, min_freq: int = 5, max_vocab: int = 50000
) -> Dict[str, int]:
    """Build vocabulary from data file with frequency threshold"""
    word_freq = {}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building vocabulary"):
            for word in line.strip().split():
                word_freq[word] = word_freq.get(word, 0) + 1

    # Filter and sort by frequency
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    for word, freq in sorted(word_freq.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and len(vocab) < max_vocab:
            vocab[word] = idx
            idx += 1

    return vocab


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Train the sequence compression model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(batch)
            loss = F.cross_entropy(
                output.view(-1, model.vocab_size), target.view(-1), ignore_index=0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        scheduler.step()


def interactive_inference(
    model: nn.Module,
    vocab: Dict[str, int],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Interactive CLI for testing one-shot sequence generation"""
    model.eval()
    rev_vocab = {v: k for k, v in vocab.items()}

    print("\nOne-Shot Sequence Generation")
    print("Enter a sequence (or 'quit' to exit):")

    while True:
        try:
            sequence = input("\n> ").strip()
            if sequence.lower() == "quit":
                break

            # Tokenize input
            words = sequence.split()[: model.max_len]
            indices = [vocab.get(word, vocab["<UNK>"]) for word in words]
            if len(indices) < model.max_len:
                indices = indices + [0] * (model.max_len - len(indices))

            # Generate
            x = torch.tensor(indices).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model.encode(x)
                output = model.decode(z)
                pred_indices = output[0].argmax(dim=-1)

                pred_words = [
                    rev_vocab[idx.item()]
                    for idx in pred_indices
                    if idx.item() not in [0, 1]
                ]
                print("Generated: ", " ".join(pred_words))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sentences.txt")
    parser.add_argument("--model_path", type=str, default="sequence_compressor.pt")
    parser.add_argument("--vocab_path", type=str, default="vocab.json")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        # Training mode
        print("Building vocabulary...")
        vocab = build_vocabulary(args.data_path)
        with open(args.vocab_path, "w") as f:
            json.dump(vocab, f)

        # Create dataset and model
        dataset = StreamingTextDataset(args.data_path, vocab, max_len=20)
        train_loader = DataLoader(dataset, batch_size=64, num_workers=4)

        model = SequenceCompressor(len(vocab))
        train_model(model, train_loader)

        # Save model
        torch.save(model.state_dict(), args.model_path)

    else:
        # Inference mode
        if not Path(args.model_path).exists():
            print(f"No model found at {args.model_path}")
            return

        # Load vocab and model
        with open(args.vocab_path) as f:
            vocab = json.load(f)

        model = SequenceCompressor(len(vocab))
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(device)

        # Start interactive session
        interactive_inference(model, vocab, device)


if __name__ == "__main__":
    main()
