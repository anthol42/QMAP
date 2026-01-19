from qmap.toolkit import train_test_split
import json

with open("../build/dbaasp.json", "r") as f:
    dataset = json.load(f)

for i, seed in enumerate([1, 3, 7, 12, 404]):
    print(f"Splitting with seed {seed}")
    train_seq, test_seq, train_idx, test_idx = train_test_split(
        [sample["sequence"] for sample in dataset],
        [i for i in range(len(dataset))],
        random_state=seed)

    test = [dataset[i] for i in test_idx]

    with open(f"../build/benchmark_split_{i}.json", "w") as f:
        json.dump(test, f)

    print(f"Saved benchmark split {i} with {len(test)} samples.")

