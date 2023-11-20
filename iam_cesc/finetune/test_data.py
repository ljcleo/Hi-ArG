from data import IAMCESCDataset
from transformers import RobertaTokenizerFast

if __name__ == "__main__":
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("./tokenizer")
    tr, de, te = IAMCESCDataset.load_train_dev_test("iam")

    for mode, ds in (("train", tr), ("dev", de), ("test", te)):
        print(mode, len(ds))
        sample = ds[0]
        print(tokenizer.decode(sample["input_ids"]))
        print(tokenizer.decode(sample["node_attr"]))
        print(tokenizer.decode(sample["edge_attr"]))
        print(sample["labels"])
