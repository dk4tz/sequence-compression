import math
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import gensim.downloader as api


batch_size = 16
max_len = 32
n_latent = 16
n_epochs = 1
train_percent = 0.01


class TextDataset(IterableDataset):
    def __init__(self, path, vocab, max_len):
        self.path = path
        self.vocab = vocab
        self.max_len = max_len

    def __iter__(self):
        while True:
            with open(self.path) as f:
                for line in f:
                    tokens = line.strip().split()
                    if len(tokens) <= self.max_len:
                        x = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
                        if len(x) < self.max_len:
                            x = x + [self.vocab["<pad>"]] * (self.max_len - len(x))
                        yield torch.tensor(x)

    def count_lines(self):
        return sum(1 for _ in open(self.path))


class Compressor(nn.Module):
    def __init__(self, vocab_size, n_latent, max_len, vocab, word_vectors):
        super().__init__()
        n_dims = word_vectors.vector_size

        self.max_len = max_len
        self.vocab_size = vocab_size

        embedding_matrix = torch.zeros((vocab_size, n_dims))

        for word, idx in vocab.items():
            if word in ["<pad>", "<unk>"]:
                embedding_matrix[idx] = torch.zeros(n_dims)
            elif word in word_vectors:
                embedding_matrix[idx] = torch.tensor(word_vectors[word])
            else:
                nn.init.xavier_uniform_(embedding_matrix[idx].unsqueeze(0))

        self.embed = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

        pos = torch.arange(max_len).unsqueeze(1).float()
        pe = torch.zeros(1, max_len, n_dims)
        div = torch.exp(torch.arange(0, n_dims, 2) * -(np.log(10000.0) / n_dims))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        self.encoder = nn.Sequential(
            nn.Linear(n_dims * max_len, n_dims * 2),
            nn.LayerNorm(n_dims * 2),
            nn.ReLU(),
            nn.Linear(n_dims * 2, n_latent),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_latent, n_dims * 2),
            nn.LayerNorm(n_dims * 2),
            nn.ReLU(),
            nn.Linear(n_dims * 2, n_dims * max_len),
        )
        self.out = nn.Linear(n_dims, vocab_size)

    def forward(self, x):
        b, t = x.size()
        h = self.embed(x) + self.pe
        h = h.view(b, -1)
        z = self.encoder(h)
        h = self.decoder(z)
        h = h.view(b, t, -1)
        return self.out(h)


def load_pretrained_embeddings():
    word_vectors = api.load("glove-wiki-gigaword-100")
    vocab = {
        "<pad>": 0,  # Index 0 reserved for padding
        "<unk>": 1,  # Index 1 reserved for unknown words
    }

    word_vectors.add_vector("<pad>", np.zeros(word_vectors.vector_size))

    unk_vector = np.mean(
        [word_vectors[word] for word in word_vectors.index_to_key[:1000]], axis=0
    )
    word_vectors.add_vector("<unk>", unk_vector)

    for word in word_vectors.index_to_key:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab, word_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    # embeddings
    vocab, word_vectors = load_pretrained_embeddings()

    # model
    model = Compressor(len(vocab), n_latent, max_len, vocab, word_vectors)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    rev_vocab = {v: k for k, v in vocab.items()}

    # train
    if args.train:
        dataset = TextDataset("data/sentences.txt", vocab, max_len)
        loader = DataLoader(dataset, batch_size=batch_size)
        opt = torch.optim.Adam(model.parameters())
        rev_vocab = {v: k for k, v in vocab.items()}

        print(f"training on {device}")
        steps_per_epoch = math.floor(
            dataset.count_lines() // batch_size * train_percent
        )

        for epoch in range(n_epochs):
            # Track stats
            losses = []
            accs = []
            model.train()

            # Training loop with two progress bars
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            pbar = tqdm(enumerate(loader), total=steps_per_epoch, desc="train")
            for it, batch in pbar:
                # Forward pass
                x = batch.to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, model.vocab_size), x.view(-1), ignore_index=0
                )

                # Backward pass
                opt.zero_grad()
                loss.backward()
                opt.step()

                # Track metrics
                losses.append(loss.item())
                pred = logits.argmax(dim=-1)
                acc = (pred == x).float().mean().item()
                accs.append(acc)

                # Update progress bar
                pbar.set_description(
                    f"loss: {loss.item():.3f} | "
                    f"avg_loss: {np.mean(losses[-100:]):.3f} | "
                    f"avg_acc: {np.mean(accs[-100:]):.3f}"
                )

                # Sample every N steps
                if it % 500 == 0:
                    idx = torch.randint(0, len(x), (1,)).item()
                    pred_tokens = [
                        rev_vocab[i.item()] for i in pred[idx] if i.item() not in [0, 1]
                    ]
                    true_tokens = [
                        rev_vocab[i.item()] for i in x[idx] if i.item() not in [0, 1]
                    ]
                    print(f"\nSample at step {it}:")
                    print(f'pred: {" ".join(pred_tokens)}')
                    print(f'true: {" ".join(true_tokens)}')

            # Epoch summary
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accs)
            print(f"\nEpoch {epoch+1} summary:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Average accuracy: {avg_acc:.4f}")

            # Save checkpoint
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "vocab": vocab,
                "epoch": epoch,
                "loss": avg_loss,
                "acc": avg_acc,
            }
            torch.save(ckpt, "model.pt")

    # inference
    else:
        if not Path("model.pt").exists():
            print("no checkpoint found, train first")
            return

        ckpt = torch.load("model.pt")
        model.load_state_dict(ckpt["model"])
        print("\nenter text (ctrl-c to exit)")
        while True:
            try:
                text = input("> ")
                tokens = text.strip().split()
                if not tokens:
                    continue

                x = [vocab.get(t, vocab["<unk>"]) for t in tokens]
                if len(x) > max_len:
                    print(f"warning: trimming to {max_len} tokens")
                    x = x[:max_len]
                elif len(x) < max_len:
                    x = x + [vocab["<pad>"]] * (max_len - len(x))
                x = torch.tensor(x).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits[0], dim=-1)
                    tokens = []
                    for i in range(max_len):
                        idx = probs[i].argmax().item()
                        if idx == vocab["<pad>"]:
                            break
                        tokens.append(rev_vocab[idx])
                print(" ".join(tokens))

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
