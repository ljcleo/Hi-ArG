from data import AKGDataset

if __name__ == "__main__":
    for comment in range(3):
        print(f"COMMENT {comment}")
        ds = AKGDataset("argsme", "eval", comment=comment)
        avg_nodes = 0
        max_nodes = -1
        avg_edges = 0
        max_edges = -1

        for i in range(len(ds)):
            sample = ds[i]
            cur_nodes = sample["node_attr"].shape[0]
            avg_nodes = (avg_nodes * i + cur_nodes) / (i + 1)
            max_nodes = max(max_nodes, cur_nodes)
            cur_edges = sample["edge_attr"].shape[0]
            avg_edges = (avg_edges * i + cur_edges) / (i + 1)
            max_edges = max(max_edges, cur_edges)

            if (i + 1) % 10000 == 0:
                print(
                    f"NODES AVG {avg_nodes} MAX {max_nodes} "
                    f"EDGES AVG {avg_edges} MAX {max_edges} "
                )

        print(f"NODES AVG {avg_nodes} MAX {max_nodes} " f"EDGES AVG {avg_edges} MAX {max_edges} ")
