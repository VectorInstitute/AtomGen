from atomgen.data.data_collator import DataCollatorForAtomModeling

def test_data_collator():
    data_collator = DataCollatorForAtomModeling()
    input_ids = [1,2,10]
    coords = [[0.5, 0.2, 0.1], [0.3, 0.4, 0.5], [0.1, 0.2, 0.3]]
    labels = [[1, 2, 3], [4, 5, 6]]
    batch = data_collator(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert batch["input_ids"].tolist() == input_ids
    assert batch["attention_mask"].tolist() == attention_mask
    assert batch["labels"].tolist() == labels
    assert batch["decoder_input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert batch["decoder_attention_mask"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert batch["decoder_labels"].tolist() == [[1, 2, 3], [4, 5, 6]]