import re

def clean_sequence(seq: str) -> str:
    """
    Clean a raw DNA/FASTA sequence:
    - Strip FASTA header lines (starting with '>')
    - Remove whitespace and newlines
    - Convert to uppercase
    - Keep only valid ACGT bases (strip ambiguous/invalid chars)
    """
    if not seq:
        return ""

    lines = seq.strip().split("\n")
    # Remove FASTA header lines
    lines = [l for l in lines if not l.startswith(">")]
    joined = "".join(lines).upper()

    # Remove anything that isn't a standard nucleotide base
    cleaned = re.sub(r"[^ACGT]", "", joined)
    return cleaned