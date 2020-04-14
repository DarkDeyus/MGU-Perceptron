import pandas as pd
import os

def main():
    regression = []
    classification = []
    reg_path = "data/projekt1/regression"
    reg_raw_files = os.listdir(reg_path)
    cls_path = "data/projekt1/classification"
    cls_raw_files = os.listdir(cls_path)
    for file in reg_raw_files:
        f1 = reg_path + "/" + file
        f2 = f1.replace("projekt1", "projekt1_test").replace("train", "test")
        regression.append((f1, f2))
    for file in cls_raw_files:
        f1 = cls_path + "/" + file
        f2 = f1.replace("projekt1", "projekt1_test").replace("train", "test")
        classification.append((f1, f2))
    with open("res.txt", 'w') as f:
        for r in regression:
            df = pd.read_csv(r[0])
            f.write(f"{r[0]}: min {df.min()}, max {df.max()})")


if __name__ == "__main__":
    main()