# sequence-compression

Testing if neural networks can learn to encode complete sequences (including word order) in single vectors, enabling parallel text generation (one-shot) instead of sequential generation (word-by-word, auto-regressive).

## Results Summary

In-progress.

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
python download_openwebtext.py
python sequence_compressor.py
```

## Analysis & Future Work

1. Why it works for short sequences:

    - Vector capacity sufficient for small content + order
    - Clean geometric separation of orderings
    - Fast parallel generation

2. Why it fails for longer sequences:

    - Vector capacity saturates
    - Order information gets entangled
    - Error compounds across positions

3. Potential improvements:
    - Hierarchical compression for longer sequences
    - Sparse vector representations
    - Separate content/order spaces
