"""
HuggingFace-style tokenizer for modular addition tasks.

Handles vocabulary mapping and encoding/decoding for all training stages
(pretrain, SFT, RL).

Token vocabulary:
    0..p-1: numbers
    p: '='
    p+1: '<bos>'
    p+2: '<eos>'
    p+3: '<pad>'
"""

from typing import List, Union


class ModularAdditionTokenizer:
    """HuggingFace-style tokenizer for modular addition tasks."""

    def __init__(self, p: int):
        """
        Initialize tokenizer with prime modulus p.

        Args:
            p: Prime modulus. Numbers 0..p-1 are valid tokens.
        """
        self.p = p

        # Special token strings
        self.eq_token = "="
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

        # Special token IDs (numbers are 0..p-1)
        self.eq_token_id = p
        self.bos_token_id = p + 1
        self.eos_token_id = p + 2
        self.pad_token_id = p + 3

        # Vocabulary size
        self.vocab_size = p + 4

        # HuggingFace-compatible attribute for HookedTransformer.generate()
        self.padding_side = "right"

        # Build token-to-id and id-to-token mappings
        self._id_to_token = {i: str(i) for i in range(p)}
        self._id_to_token[self.eq_token_id] = self.eq_token
        self._id_to_token[self.bos_token_id] = self.bos_token
        self._id_to_token[self.eos_token_id] = self.eos_token
        self._id_to_token[self.pad_token_id] = self.pad_token

        self._token_to_id = {v: k for k, v in self._id_to_token.items()}

    def encode(self, sequence: List[Union[str, int]]) -> List[int]:
        """
        Convert sequence of numbers/symbols to token IDs.

        Args:
            sequence: List of integers (0..p-1) or strings ('=', '<bos>', etc.)

        Returns:
            List of token IDs
        """
        result = []
        for item in sequence:
            if isinstance(item, int):
                if 0 <= item < self.p:
                    result.append(item)
                else:
                    raise ValueError(f"Number {item} out of range [0, {self.p-1}]")
            elif isinstance(item, str):
                if item in self._token_to_id:
                    result.append(self._token_to_id[item])
                elif item.isdigit():
                    result.append(int(item))
                else:
                    raise ValueError(f"Unknown token: {item}")
            else:
                raise TypeError(f"Expected int or str, got {type(item)}")
        return result

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False, stop_at_eos: bool = True) -> str:
        """
        Convert token IDs back to readable string.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: If True, omit <bos>, <eos>, <pad> from output
            stop_at_eos: If True, stop decoding at first <eos> token (inclusive)

        Returns:
            Space-separated string representation
        """
        # Truncate at first EOS if requested
        if stop_at_eos and self.eos_token_id in token_ids:
            eos_idx = token_ids.index(self.eos_token_id) if isinstance(token_ids, list) else token_ids.tolist().index(self.eos_token_id)
            token_ids = token_ids[:eos_idx + 1]

        parts = []
        for tid in token_ids:
            if tid in self._id_to_token:
                token = self._id_to_token[tid]
                if skip_special_tokens and token in [self.bos_token, self.eos_token, self.pad_token]:
                    continue
                parts.append(token)
            else:
                parts.append(f"<unk:{tid}>")
        return " ".join(parts)

    def __call__(self, sequence: List[Union[str, int]]) -> List[int]:
        """Shorthand for encode."""
        return self.encode(sequence)

    def __repr__(self) -> str:
        return f"ModularAdditionTokenizer(p={self.p}, vocab_size={self.vocab_size})"
