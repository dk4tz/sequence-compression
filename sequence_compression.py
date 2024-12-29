import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import random
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, sequences: List[str], vocab: Dict[str, int], max_len: int):
        self.sequences = sequences
        self.vocab = vocab
        self.max_len = max_len
        self.pad_token = 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        words = self.sequences[idx].split()[: self.max_len]
        indices = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]
        if len(indices) < self.max_len:
            indices = indices + [self.pad_token] * (self.max_len - len(indices))
        return torch.tensor(indices), torch.tensor(indices)


def generate_semantic_space() -> List[str]:
    """Generate diverse sequences covering broad semantic domains"""
    sequences = []

    # Academic Discussion
    academic = [
        "the research suggests a correlation between neural network depth and performance",
        "recent studies have shown promising results in natural language understanding",
        "we observed significant improvements in model accuracy after optimization",
        "the experimental results contradict our initial hypothesis",
        "further analysis reveals interesting patterns in the data distribution",
    ]

    # Technical Analysis
    technical = [
        "the system bottleneck appears to be in the database query optimization",
        "memory usage spikes during peak processing periods",
        "network latency significantly affects response times",
        "implementing caching improved performance by an order of magnitude",
        "load balancing effectively distributed the computational burden",
    ]

    # Creative Discussion
    creative = [
        "the design philosophy emphasizes user experience over technical complexity",
        "visual elements should guide users through the interface naturally",
        "we need to balance functionality with aesthetic appeal",
        "the color scheme reflects the brand's core values",
        "user feedback suggests the interface feels intuitive and responsive",
    ]

    # Business Context
    business = [
        "market analysis indicates growing demand for cloud solutions",
        "customer retention metrics have improved significantly",
        "quarterly revenue exceeded projected targets",
        "stakeholder feedback has been predominantly positive",
        "operational costs decreased after implementing automation",
    ]

    # Social Interaction
    social = [
        "community engagement has increased substantially",
        "user feedback provides valuable insights for improvement",
        "collaboration between teams enhanced project outcomes",
        "regular communication helped prevent misunderstandings",
        "peer review process identified several critical issues",
    ]

    # Problem Solving
    problems = [
        "the root cause appears to be in the authentication layer",
        "debugging revealed an edge case in error handling",
        "performance profiling identified several bottlenecks",
        "log analysis shows intermittent connection failures",
        "stress testing exposed scalability limitations",
    ]

    # Strategic Planning
    strategic = [
        "long term sustainability requires architectural changes",
        "resource allocation needs optimization for efficiency",
        "roadmap priorities align with business objectives",
        "risk assessment suggests needed security improvements",
        "competitive analysis reveals market opportunities",
    ]

    # Domain variations
    domains = [academic, technical, creative, business, social, problems, strategic]

    # Common modifiers that add natural variation
    modifiers = [
        "interestingly",
        "notably",
        "surprisingly",
        "importantly",
        "critically",
        "fundamentally",
        "essentially",
        "particularly",
        "significantly",
        "evidently",
        "apparently",
        "presumably",
        "arguably",
        "conceivably",
        "potentially",
    ]

    # Epistemic markers
    epistemic = [
        "i believe",
        "i think",
        "in my view",
        "from my perspective",
        "as far as i can tell",
        "based on my experience",
        "it seems that",
        "data suggests",
        "evidence indicates",
        "analysis shows",
        "research implies",
    ]

    # Generate varied sequences
    target_size = 100000  # Large dataset

    while len(sequences) < target_size:
        # Base sequence from random domain
        base = random.choice(random.choice(domains))

        # Apply variations
        if random.random() < 0.3:
            base = f"{random.choice(epistemic)} {base}"
        if random.random() < 0.3:
            words = base.split()
            insert_pos = random.randint(1, len(words) - 1)
            words.insert(insert_pos, random.choice(modifiers))
            base = " ".join(words)

        sequences.append(base.lower())

        # Generate related follow-ups
        if random.random() < 0.3:
            follow_up = random.choice(
                [
                    f"this suggests that {base.lower()}",
                    f"we found that {base.lower()}",
                    f"analysis shows that {base.lower()}",
                    f"data indicates that {base.lower()}",
                    f"evidence suggests that {base.lower()}",
                ]
            )
            sequences.append(follow_up)

    return sequences


class SequenceCompressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        latent_size: int = 384,
        max_len: int = 24,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_len = max_len

        # Encoder
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * max_len, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_size),
        )

        # Decoder (one-shot generation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * max_len),
        )

        self.out = nn.Linear(hidden_size, vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        embedded = self.embed(x)
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


def build_vocab(sequences: List[str], min_freq: int = 3) -> Dict[str, int]:
    """Build vocabulary with lower frequency threshold to capture more terms"""
    word_freq = {}
    for seq in sequences:
        for word in seq.split():
            word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2

    # Add words by frequency
    for word, freq in sorted(word_freq.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab


def main():
    print("Generating semantic sequences...")
    sequences = generate_semantic_space()
    print(f"Total sequences: {len(sequences)}")

    vocab = build_vocab(sequences)
    print(f"Vocabulary size: {len(vocab)}")

    # Show vocabulary distribution
    print("\nSample vocabulary by domain:")
    sample_size = 10
    print(f"Total unique words: {len(vocab) - 2}")  # Excluding PAD and UNK

    dataset = TextDataset(sequences, vocab, max_len=24)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceCompressor(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for batch, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
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

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch, target in val_loader:
                batch, target = batch.to(device), target.to(device)
                output = model(batch)
                loss = F.cross_entropy(
                    output.view(-1, model.vocab_size), target.view(-1), ignore_index=0
                )
                val_loss += loss.item()

        print(
            f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
            f"Val Loss = {val_loss/len(val_loader):.4f}"
        )


if __name__ == "__main__":
    main()
