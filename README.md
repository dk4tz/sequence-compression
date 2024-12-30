# sequence-compression

Hypothesis - neural networks can learn to encode positional embeddings (i.e. sequences of words) into latent space vectors, enabling all-at-once answer generation instead of sequential, auto-regressive token generation.

## status

Work in progress:

1. Implement proper train, validation, test partitions
2. Up batch size and train on larger GPU

## usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python download_openwebtext.py
python sequence_compressor.py
```
