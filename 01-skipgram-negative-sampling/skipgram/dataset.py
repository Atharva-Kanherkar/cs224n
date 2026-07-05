"""Step 2: turn tokenized sentences + a vocabulary into (center, context)
training pairs.
"""


def generate_skipgram_pairs(
    tokenized_sentences: list[list[str]],
    word2idx: dict,
    window_size: int,
) -> list[tuple[int, int]]:
    """For every word in every sentence, treat it as the center word and
    pair it with every other word within `window_size` positions on either
    side (the context words).

    Example: sentence ["the", "quick", "brown", "fox"], window_size=1:
        center="the":   context="quick"                  -> (the, quick)
        center="quick": context="the", "brown"            -> (quick, the), (quick, brown)
        center="brown": context="quick", "fox"            -> (brown, quick), (brown, fox)
        center="fox":   context="brown"                   -> (fox, brown)

    Rules:
      - Never generate a pair where center and context come from different
        sentences (no crossing sentence boundaries).
      - If a word isn't in word2idx (e.g. it was dropped by min_count in
        build_vocab), skip it entirely — it can't be a center word and it
        can't be a context word.
      - Return integer ids (via word2idx), not strings.
      - A pair is directional: (center, context) is a distinct training
        example from (context, center). Both should appear in the output
        when both words are within each other's window.

    Returns:
        A flat list of (center_id, context_id) tuples.
    """
    results = []
    for sentence in tokenized_sentences:
        for center_pos, center_word in enumerate(sentence):
            for near in range(center_pos - window_size, center_pos + window_size + 1):
                if near == center_pos:
                    continue
                if near <0 or near >= len(sentence):continue
                word_append = sentence[near]
                if center_word in word2idx and word_append in word2idx:
                    results.append((word2idx[center_word], word2idx[word_append]))

                    
              
    return results


                
            
            
            
