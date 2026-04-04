from collections import Counter
import re

def compute_gc(seq: str) -> float:
    """GC content as a fraction (0.0 - 1.0)"""
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return round(gc / len(seq), 4)

def compute_length(seq: str) -> int:
    """Return sequence length"""
    return len(seq)

def detect_replicon(seq: str) -> str:
    """
    Naive replicon detection based on known signature k-mers.
    Returns: IncF | IncI | IncN | Unknown
    """
    # Simplified signature patterns (replace with real HMM/BLAST in production)
    signatures = {
        "IncF": ["GGCGTGGTCG", "TGCCGAGCGT", "ATGCGCAATG"],
        "IncI": ["GCAATTTCGG", "TTGCGCTATG", "CAGGCGTAAT"],
        "IncN": ["CGCTTAACGG", "TTACGCCTGT", "AATGCGCTAG"],
    }
    for replicon, patterns in signatures.items():
        for pat in patterns:
            if pat in seq:
                return replicon
    return "Unknown"

def kmer_features(seq: str, k: int = 4) -> list:
    """
    Compute normalised k-mer frequency vector (4^k features).
    Default k=4 → 256 features.
    """
    bases = "ACGT"
    # Build all possible k-mers
    from itertools import product
    all_kmers = ["".join(p) for p in product(bases, repeat=k)]
    kmer_index = {km: i for i, km in enumerate(all_kmers)}

    counts = [0] * len(all_kmers)
    total = 0

    for i in range(len(seq) - k + 1):
        km = seq[i:i + k]
        if km in kmer_index:          # skip k-mers with non-ACGT bases
            counts[kmer_index[km]] += 1
            total += 1

    if total == 0:
        return counts  # all zeros if sequence is empty / invalid

    # Normalise to frequencies
    return [c / total for c in counts]