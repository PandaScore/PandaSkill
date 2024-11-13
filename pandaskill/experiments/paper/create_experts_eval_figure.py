"""
This script can be used to generate the figure comparing the concordance between different models and experts for the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from os.path import join
import pandas as pd
from pandaskill.experiments.general.utils import ARTIFACTS_DIR

models = {
    ("pscore", "ewma"): "PScore +\nEWMA",
    ("pscore", "openskill") : "PScore +\nOpenSkill",
    ("pscore", "meta_openskill") : "PScore +\nMeta_OpenSkill",
    ("pscore", "ffa_openskill") : "PScore +\nFFA_OpenSkill",
    ("pscore", "meta_ffa_openskill") : "PScore +\nMeta_FFA_OpenSkill",
    ("pscore", "meta_ffa_trueskill") : "PScore +\nMeta_FFA_TrueSkill",
    ("playerank", "ewma"): "PlayeRank+\nEWMA",
    ("playerank", "meta_ffa_openskill"): "PlayeRank +\nMeta_FFA_OpenSkill",
    ("performance_index", "ewma"): "PI +\nEWMA",
    ("performance_index", "meta_ffa_openskill"): "PI +\nMeta_FFA_OpenSkill",    
}

file_name = "ranking_experts_concordance_comparison_paper.pdf"

labels = ["global", "korea", "china", "europe", "north_america"]
labels_display_dict = {
    "global": "Global",
    "korea": "Korea",
    "china": "China",
    "europe": "Europe",
    "north_america": "North America",
}


data = {}
for (perf_model, rating_model) in models.keys():
    file_path = join(
        ARTIFACTS_DIR,
        "experiments",
        perf_model,
        "skill_rating",
        rating_model,
        "ranking_experts_evaluation.yaml",
    )
    try:
        with open(file_path, "r") as file:
            ranking_data = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file {file_path}: {exc}")
        continue

    model_data = {
        "majority": [],
        "unanimity": []
    }
    for label in labels:
        region_data = ranking_data.get(label, {})
        if "openskill_metrics" in region_data:
            region_data = region_data["openskill_metrics"]
        model_data["majority"].append(
            region_data.get("majority_concordance", 0) * 100
        )
        model_data["unanimity"].append(
            region_data.get("unanimous_concordance", 0) * 100
        )
    data[(perf_model, rating_model)] = model_data

n_models = len(models)
n_regions = len(labels)

if n_models == 0:
    raise ValueError("No models loaded. Please check the YAML file paths and contents.")

def prepare_data(concordance_type):
    plot_data = []
    for (perf_model, rating_model), model_data in data.items():
        for region_idx, region in enumerate(labels):
            plot_data.append({
                'Model': models[(perf_model, rating_model)],
                'Region': labels_display_dict[region],
                'Concordance': model_data[concordance_type][region_idx]
            })
    return pd.DataFrame(plot_data)

majority_data = prepare_data('majority')
unanimity_data = prepare_data('unanimity')

def get_best_models(data):
    best_models = {}
    grouped_data = data.groupby('Region')
    for region, group in grouped_data:
        max_concordance = group['Concordance'].max()
        # Handle multiple models with the same max concordance
        best_models[region] = group[group['Concordance'] == max_concordance]['Model'].tolist()
    return best_models

best_majority_models = get_best_models(majority_data)
best_unanimity_models = get_best_models(unanimity_data)

print("\n*** Majority Concordance Average ***")
print(majority_data.groupby("Model")["Concordance"].mean())
print(majority_data.groupby("Model")["Concordance"].std())
print("\n*** Unanimity Concordance Average ***")
print(unanimity_data.groupby("Model")["Concordance"].mean())
print(unanimity_data.groupby("Model")["Concordance"].std())

# Add a 'Best' column to the DataFrames
def mark_best_models(df, best_models):
    df['Best'] = df.apply(lambda row: row['Model'] in best_models[row['Region']], axis=1)
    return df

majority_data = mark_best_models(majority_data, best_majority_models)
unanimity_data = mark_best_models(unanimity_data, best_unanimity_models)

palette = [sns.color_palette()[7]] + sns.color_palette()[:4]
region_palette = {
    region: color
    for region, color in zip(labels_display_dict.values(), palette)
}


models_order = list(models.values())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), sharex=True)
ax1.grid(True,zorder=0)
ax2.grid(True,zorder=0)

hue_order = labels_display_dict.values()

sns.barplot(
    x="Model", y="Concordance", hue="Region", data=majority_data, ax=ax1, 
    palette=region_palette, hue_order=hue_order, order=models_order, gap=0.1,zorder=3
)
ax1.set_title('Majority Concordance', fontsize=18)
ax1.set_ylabel("")
ax1.set_ylim(50, 100)
ax1.tick_params(axis='y', which='major', labelsize=14)

sns.barplot(
    x="Model", y="Concordance", hue="Region", data=unanimity_data, ax=ax2, 
    palette=region_palette, hue_order=hue_order, order=models_order, gap=0.1,zorder=3
)
ax2.set_title('Unanimity Concordance', fontsize=18)
ax2.set_ylabel("")
ax2.set_ylim(50, 100)
ax2.tick_params(axis='y', which='major', labelsize=13)
ax2.tick_params(axis='x', which='major', labelsize=13)
ax2.set_xticklabels(models_order, fontsize=13)

def annotate_best_bars(ax, df):
    """
    Annotate the best bars with a star (*) on the given axis.

    Parameters:
    - ax: The matplotlib axis containing the bars.
    - df: The DataFrame used to plot the bars.
    """
    # Iterate through each bar and data point simultaneously
    expected_patches = len(df)

    # Filter patches that correspond to the bars
    # Seaborn adds extra patches for things like the legend, so we'll limit to expected_patches
    bars = ax.patches[:expected_patches]
    df["model_order"] = df["Model"].apply(lambda x: list(models.values()).index(x))
    reverse_label_display_dict = {v: k for k, v in labels_display_dict.items()}
    df["region_order"] = df["Region"].apply(lambda x: labels.index(reverse_label_display_dict[x]))
    for bar, (_, row) in zip(bars, df.sort_values(["region_order", "model_order"]).iterrows()):
        if row['Best']:
            # Get the coordinates of the bar
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            y = height - 8

            # Annotate with a star
            ax.text(x, y + 1, '★', ha='center', va='bottom', color='gold', fontsize=20, fontweight='bold')

annotate_best_bars(ax1, majority_data)
annotate_best_bars(ax2, unanimity_data)

ax1.legend_.remove()
ax2.legend_.remove()
ax1.set_ylabel("")
ax2.set_xlabel("")

handles, labels_ = ax1.get_legend_handles_labels()
fig.legend(
    handles, labels_, title='Region', bbox_to_anchor=(0.98, 0.94), 
    loc='upper right', ncol=5, fontsize=14, title_fontsize=14)

fig.text(0.5, -0.02, 'Model', ha='center', fontsize=16)
fig.text(-0.02, 0.5, 'Concordance (%)', va='center', fontsize=16, rotation=90)

plt.tight_layout()

plt.savefig(join(ARTIFACTS_DIR, "paper", file_name), format="pdf", bbox_inches="tight")
