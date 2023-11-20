import torch
from data import ArgKP2021Dataset, ContextFormat
from transformers import RobertaTokenizerFast

if __name__ == "__main__":
    tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained("./tokenizer")

    for i in ContextFormat:
        tr, de, te = ArgKP2021Dataset.load_train_dev_test("argkp", i)

        for mode, ds in (("train", tr), ("dev", de), ("test", te)):
            print(i, mode, len(ds), len(ds[0]))
            sample = ds[0][0]

            print(tokenizer.decode(sample["input_ids"]))
            print(tokenizer.decode(sample["node_attr"]))
            print(tokenizer.decode(sample["edge_attr"]))

            print(
                torch.cat(
                    [sample["choice_mask"].nonzero().ravel().unsqueeze(0), sample["labels"]]
                ).mT
            )

            top_mask = sample["top_mask"]
            print(top_mask)

            spec = top_mask.argsort(descending=True)[: torch.sum(top_mask != -100)].flip(0)
            print(spec)
            print(tokenizer.decode(sample["node_attr"][spec]))
