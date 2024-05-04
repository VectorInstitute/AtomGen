"""Test AtomTokenizer."""
from atomgen.data.tokenizer import AtomTokenizer


def test_tokenizer():
    """Test AtomTokenizer."""
    tokenizer = AtomTokenizer(vocab_file="atomgen/data/tokenizer.json")
    text = "MgCCHeNNN"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["Mg", "C", "C", "He", "N", "N", "N"]
