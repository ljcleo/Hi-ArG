from collate import DataCollatorForJointInput
from data import IAMCESCDataset
from torch_geometric.data import Batch
from transformers import RobertaTokenizer

if __name__ == "__main__":
    tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained("./tokenizer")
    ds = IAMCESCDataset.load_train_dev_test("iam")[2]
    collator = DataCollatorForJointInput(tokenizer)
    sample = collator([ds[i] for i in range(4)])

    print(sample["labels"])
    for x in sample["input_ids"]:
        print(tokenizer.decode(x))

    print(sample["attention_mask"])

    sample_graphs = Batch.from_data_list(sample["graphs"])
    print(tokenizer.decode(sample_graphs.x))
    print(tokenizer.decode(sample_graphs.edge_attr))

    for k, (i, j) in enumerate(zip(sample_graphs.ptr[:-1], sample_graphs.ptr[1:])):
        print(tokenizer.decode(sample_graphs.get_example(k).x[j - i - 2 :]))
