# container: pytorch
# needs: network
"""What is a token? The same sentence in English and French, cut up by three tokenizers.

An LLM reads and writes tokens: pieces of words from a fixed vocabulary that
its tokenizer learned from the training text. APIs charge per token, and the
context window is counted in tokens. This script downloads only the
tokenizers (a few MB each, no model weights) and prints how many tokens each
sentence takes, and the pieces, with "|" between tokens.

    python week2-tokens.py
"""

# region: pieces
from transformers import AutoTokenizer

def pieces(tok, text):
    """Its tokens, each as text."""
    ids = tok(text).input_ids
    return [tok.decode([i]) for i in ids]
# endregion

SENTENCES = {
    "English": "The model predicts whether a customer will cancel their subscription next month.",
    "French": "Le modèle prédit si un client annulera son abonnement le mois prochain.",
}
TOKENIZERS = ["openai-community/gpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"]

print()
for name in TOKENIZERS:
    tok = AutoTokenizer.from_pretrained(name)
    counts = {}
    for lang, text in SENTENCES.items():
        p = pieces(tok, text)
        counts[lang] = len(p)
        print(f"{name} {lang}: {len(text)} characters, {len(p)} tokens")
        print("   ", "|".join(p))
    print(f"    French needs {counts['French'] / counts['English']:.1f}x the tokens\n")
