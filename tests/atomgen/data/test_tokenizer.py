from atomgen.data.tokenizer import AtomTokenizer

def test_tokenizer():
    tokenizer = AtomTokenizer(vocab_file="atomgen/data/tokenizer.json")
    text = "BaCCHeNNN"
    tokens = tokenizer.tokenize(text)
    assert tokens == ["Ba", "C", "C", "He", "N", "N", "N"]