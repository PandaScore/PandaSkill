from pandaskill.experiments.general.metrics import compute_ece_from_binned_df, bin_predictions_equal_size
from pandaskill.experiments.general.utils import ALL_REGIONS, ROLES
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import pandas as pd
import seaborn as sns

def plot_model_calibration(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    nbins: int, 
    title: str, 
    saving_folder: str, 
    file_name: str
) -> None:
    binned_df = bin_predictions_equal_size(y_true, y_prob, nbins)
    ece = compute_ece_from_binned_df(binned_df)

    fig_title = f"{title}\nECE:{100*ece:0.2f}% - {nbins} bins of size {binned_df.mean()['count']:0.1f}"
    _, ax = plt.subplots(figsize=(12, 8))

    binned_df["y_true"].plot.bar(
        ax=ax,
        yerr=binned_df["error"],
        width=1,
        grid=True,
        color="purple",
        label=f"True bin winrate",
    )
    binned_df["y_prob"].plot.bar(
        ax=ax,
        width=1,
        grid=True,
        facecolor="none",
        edgecolor="grey",
        label=f"Predicted bin winrate",
    )
    ax.set_title(fig_title)
    ax.set_xlabel(f"bins")
    ax.legend(loc="upper left")
    figure_name = f"{file_name}.png"
    folder = join(saving_folder, "calibration_plots")
    os.makedirs(folder, exist_ok=True)
    path = join(folder, figure_name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_violin_distributions(
    df: pd.DataFrame, 
    x_column: str, 
    y_column: str, 
    title: str, 
    x_label: str, 
    y_label: str, 
    saving_dir: str, 
    file_name: str
) -> None:
    color_palette = sns.color_palette()

    df = df.copy()
    
    if x_column in ["role", "region"]:
        reference_list = ALL_REGIONS if x_column == "region" else ROLES
        region_order_dict = {
            region: i
            for i, region in enumerate(reference_list)
        }
        df["order"] = df[x_column].map(region_order_dict)
        df = df.sort_values("order")

        if x_column == "region": # format region names
            format_region = lambda region: region.replace(" ", "\n").replace("-", "\n")
            df[x_column] = df[x_column].apply(format_region)
            reference_list = [
                format_region(region) for region in reference_list
            ]
        
        color_palette = dict(zip(reference_list, color_palette))        
    else:
        nb_unique_hue = len(df[x_column].unique())
        color_palette = color_palette[:nb_unique_hue]


    fig, ax = plt.subplots(figsize=(7, 6))

    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    sns.violinplot(
        x=x_column,
        y=y_column,
        hue=x_column,
        palette=color_palette,
        data=df,
        inner="box",
        ax=ax
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.tight_layout()

    plt.savefig(join(saving_dir, f"{file_name[:-4]}.pdf"))
    plt.close()