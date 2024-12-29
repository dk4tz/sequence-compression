# sequence-compression

Testing if neural networks can learn to encode complete sequences (including word order) in single vectors, enabling parallel text generation instead of word-by-word generation.

## Results Summary

We tested whether we can generate sequences in parallel by compressing them into vectors that encode both content and order. Key findings after a week of experiments:

1. It works, with clear limitations:

    - Sequences up to length 4: >95% accuracy
    - Length 5-6: Rapidly degrading performance
    - Length 7+: Complete breakdown

2. Speed vs. Accuracy tradeoff:

    - Parallel generation is 3x faster for short sequences
    - Standard word-by-word is more reliable for longer text
    - Sweet spot: 3-4 word phrases

3. Vector Analysis:
    - Order information creates clear geometric patterns
    - Content and order compete for vector capacity
    - Longer sequences cause increasing entanglement

## Usage

```bash
# Install dependencies
pip install torch numpy matplotlib

# Run experiments
python sequence_compression.py
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
