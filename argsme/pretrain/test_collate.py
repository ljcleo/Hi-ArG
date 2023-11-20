from collate import DataCollatorForJointInput
from data import AKGDataset

if __name__ == "__main__":
    for comment in range(3):
        dataset = AKGDataset("argsme", "eval", comment=comment)

        for mode in (1, 2, 3):
            print(f"COMMENT {comment} MODE {mode}")
            input_text: bool = bool(mode & 1)
            input_graph: bool = bool(mode & 2)

            collator = DataCollatorForJointInput(input_text=input_text, input_graph=input_graph)
            samples: list[dict[str, list[int]]] = [dataset[i] for i in range(4)]
            batch = collator(samples)

            if input_text:
                print(batch["input_ids"])
                print(batch["input_ids_labels"])

            if input_graph:
                print(batch["graphs"].x)
                print(batch["graphs"].y)
                print(batch["graphs"].edge_index)
                print(batch["graphs"].edge_attr)
                print(batch["graphs"].edge_y)
                print(batch["graphs"].pos)
