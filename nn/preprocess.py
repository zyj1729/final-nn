# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    positive_seqs = [seq for seq, label in zip(seqs, labels) if label]
    negative_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    max_size = max(len(positive_seqs), len(negative_seqs))

    if len(positive_seqs) < max_size:
        positive_seqs = np.random.choice(positive_seqs, size=max_size, replace=True).tolist()
    else:
        negative_seqs = np.random.choice(negative_seqs, size=max_size, replace=True).tolist()

    sampled_seqs = positive_seqs + negative_seqs
    sampled_labels = [True] * len(positive_seqs) + [False] * len(negative_seqs)
    
    combined = list(zip(sampled_seqs, sampled_labels))
    np.random.shuffle(combined)
    sampled_seqs, sampled_labels = zip(*combined)

    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    nucleotide_map = {'A': [1, 0, 0, 0],
                      'T': [0, 1, 0, 0],
                      'C': [0, 0, 1, 0],
                      'G': [0, 0, 0, 1]}

    encodings = []
    for seq in seq_arr:
        encodings.append(np.array([nucleotide_map[nt] for nt in seq]).flatten())

    return np.array(encodings)