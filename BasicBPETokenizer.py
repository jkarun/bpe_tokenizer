class BasicBPETokenizer:
    """A minimal byte-level BPE tokenizer for educational purposes."""

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        tokens = list(text.encode("utf-8"))
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i in range(num_merges):
            pair_counts = self._get_pair_counts(tokens)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            tokens = self._merge(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = (self.vocab[best_pair[0]]
                                  + self.vocab[best_pair[1]])
        print(f"Trained: {len(self.merges)} merges, "
              f"vocab size = {len(self.vocab)}")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            pair_counts = self._get_pair_counts(tokens)
            candidate = None
            for pair in pair_counts:
                if pair in self.merges:
                    if (candidate is None
                            or self.merges[pair] < self.merges[candidate]):
                        candidate = pair
            if candidate is None:
                break
            tokens = self._merge(tokens, candidate, self.merges[candidate])
        return tokens

    def decode(self, token_ids):
        byte_seq = b"".join(self.vocab[tid] for tid in token_ids)
        return byte_seq.decode("utf-8", errors="replace")

    def _get_pair_counts(self, token_ids):
        counts = {}
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, token_ids, pair, new_id):
        new_tokens = []
        i = 0
        while i < len(token_ids):
            if (i < len(token_ids) - 1
                    and (token_ids[i], token_ids[i + 1]) == pair):
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(token_ids[i])
                i += 1
        return new_tokens
