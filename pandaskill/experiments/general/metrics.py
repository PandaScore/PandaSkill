import numpy as np
import pandas as pd

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, nbins: int = 10, binning_method="equal-width") -> float:
    if binning_method == "equal-width":
        binned_df = bin_predictions_equal_width(y_true,  y_prob, nbins)
    elif binning_method == "equal-size":
        binned_df = bin_predictions_equal_size(y_true,  y_prob, nbins)
    ece = compute_ece_from_binned_df(binned_df)
    return ece

def bin_predictions_equal_size(
    y_true: np.ndarray, y_prob: np.ndarray, nbins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df.loc[:, "bin"] = pd.qcut(df["y_prob"], q=nbins, duplicates="drop")
    df.loc[:, "count"] = 1
    df.loc[:, "error"] = df["y_prob"] - df["y_true"]

    binned_df = df.groupby("bin", observed=True)[
        [
            "y_prob",
            "y_true",
            "count",
            "error",
        ]
    ].agg(
        {
            "y_prob": "mean",
            "y_true": "mean",
            "count": "sum",
            "error": "sem",
        }
    )

    return binned_df


def bin_predictions_equal_width(
    y_true: np.ndarray, y_prob: np.ndarray, nbins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df.loc[:, "bin"] = pd.cut(df["y_prob"], bins=nbins, include_lowest=True)
    df.loc[:, "count"] = 1
    df.loc[:, "error"] = df["y_prob"] - df["y_true"]

    binned_df = df.groupby("bin", observed=True)[
        [
            "y_prob",
            "y_true",
            "count",
            "error",
        ]
    ].agg(
        {
            "y_prob": "mean",
            "y_true": "mean",
            "count": "sum",
            "error": "sem",
        }
    )

    return binned_df

def compute_ece_from_binned_df(binned_df: pd.DataFrame) -> float:
    ece = np.sum(
        np.abs(
            (binned_df["y_prob"] - binned_df["y_true"]) * binned_df["count"] / np.sum(binned_df["count"])
        )
    )
    return ece
