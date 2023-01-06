from attrbench.metrics import DeletionResult


if __name__ == "__main__":
    res = DeletionResult.load("deletion.h5")
    df, higher_is_better = res.get_df("constant", "linear")
