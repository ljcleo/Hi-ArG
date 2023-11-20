import torch
from collate import DataCollatorForJointInput
from data import ArgKP2021Dataset, ContextFormat
from torch_geometric.data import Batch
from transformers import RobertaTokenizer

if __name__ == "__main__":
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("./tokenizer")

    for format in ContextFormat:
        ds = ArgKP2021Dataset.load_train_dev_test("argkp", format)[2]
        collator = DataCollatorForJointInput(tokenizer)
        sample = collator([ds[i] for i in range(4)])
        print(len(sample["labels"]))

        for pos in range(len(sample["labels"])):
            print(pos, sample["labels"][pos])
            for x in sample["input_ids"][pos]:
                print(tokenizer.decode(x))

            print(sample["attention_mask"][pos])
            print(sample["choice_mask"][pos])

            sample_graphs = Batch.from_data_list(sample["graphs"][pos])
            print(tokenizer.decode(sample_graphs.x))
            print(tokenizer.decode(sample_graphs.edge_attr))

            top_mask = sample_graphs.top_mask
            print(top_mask)

            for k, (i, j) in enumerate(zip(sample_graphs.ptr[:-1], sample_graphs.ptr[1:])):
                spec = (
                    top_mask[i:j]
                    .argsort(descending=True)[: torch.sum(top_mask[i:j] != -100)]
                    .flip(0)
                )

                print(spec)
                print(tokenizer.decode(sample_graphs.get_example(k).x[spec]))
