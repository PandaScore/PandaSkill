from pandaskill.experiments.performance_score.shap_beeswarm import beeswarm
from pandaskill.experiments.general.visualization import plot_violin_distributions
import os
from os.path import join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve

def visualize_performance_scores(
    data: pd.DataFrame,
    performance_scores: pd.DataFrame,
    evaluation_metrics: dict,
    experiment_dir: str
) -> None:
    feature_importances_mean = {
        role: {
            feature_name: role_metric["mean"]
            for feature_name, role_metric in role_metrics.items()
        }
        for role, role_metrics in evaluation_metrics["features_importance"].items()
    }
    _plot_feature_importance_per_role(feature_importances_mean, experiment_dir)
    
    data = data.join(performance_scores)
    plot_violin_distributions(
        data, 
        "role",
        "performance_score",
        "Performance score distribution per role",
        "Role",
        "PScore",
        experiment_dir,
        "performance_score_per_role.png"
    )

def _plot_feature_importance_per_role(
    features_weights: dict, 
    saving_folder: str
) -> None:
    df = pd.DataFrame.from_dict(features_weights)
    fig, ax = plt.subplots()
    ax = df.plot.barh(figsize=(7, 6), ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    figure_name = "features_importance.pdf"
    path = join(saving_folder, figure_name)
    plt.savefig(path)
    plt.close()


def plot_all_models_calibration(
    calibration_data: list[dict[str, list[int]]], 
    saving_dir: str
) -> None:
    roles = list(calibration_data[0].keys())

    y_probs = {role: [] for role in roles}
    y_trues = {role: [] for role in roles}
    for fold in calibration_data:
        for role in roles:
            probs, trues = fold.get(role)
            y_probs[role].extend(probs)
            y_trues[role].extend(trues)

    fig, ax1 = plt.subplots(figsize=(7, 6))
    ax1.grid(True)
    ax2 = ax1.twinx()
    for role in roles:
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_trues[role], y_probs[role], n_bins=25)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s" % (role, ))

        ax2.hist(y_probs[role], range=(0, 1), bins=25, label=role,
                histtype="step", lw=2)
    ax1.plot([0, 1], [0, 1], 'k:', label='Perfect calibration', zorder=1)
    ax1.set_ylabel("Fraction of Positives")
    ax1.legend(loc="upper center")
    ax1.set_ylim([0.0, 1.00])
    ax1.set_xlim([0.0, 1.00])
    ax1.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()

    plt.savefig(join(saving_dir, "full_calibration_plots.pdf"))
    plt.close()

feature_str_dict = {
    'gold_per_minute': 'Gold per minute',
    'cs_per_minute': 'Creep score per minute',
    'xp_per_minute': 'Experience per minute',
    'damage_dealt_per_total_kills': 'Damage dealt total kills ratio',
    'damage_dealt_per_total_kills_per_gold': 'Damage dealt per gold total kills ratio',
    'damage_taken_per_total_kills': 'Damage taken total kills ratio',
    'damage_taken_per_total_kills_per_gold': 'Damage taken per gold total kills ratio',
    'kla': 'Kill-life-assist ratio',
    'largest_multi_kill': 'Largest multi kill',
    'largest_killing_spree_per_total_kills': 'Largest killing spree total kills ratio',
    'wards_placed_per_minute': 'Wards placed per minute',
    'objective_contest_loserate': 'Objective contest loserate',
    'objective_contest_winrate': 'Objective contest winrate',
    'free_kill_ratio': 'Free kill ratio',
    'worthless_death_ratio': 'Worthless death ratio',
    'free_kill_total_kills_ratio': 'Free kill total kills ratio',
    'worthless_death_total_kills_ratio': 'Worthless death total kills ratio'
}

def plot_shap_game_features_impact(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    feature_values_df: pd.DataFrame,
    file_name: str,
    saving_folder: str,
    nb_features_to_display: int = 5,
    show_xlabel: bool = True
) -> None:
    os.makedirs(saving_folder, exist_ok=True)
    feature_values_df = feature_values_df.rename(index=feature_str_dict)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=feature_values_df.round(2),
        feature_names=feature_values_df.index
    )
    shap.waterfall_plot(
        explanation,
        max_display=nb_features_to_display,
        show=False,
    )
    fig = plt.gcf()
    if show_xlabel:
        fig.axes[0].set_xlabel("SHAP Values", labelpad=20, fontsize=14)  # Set xlabel for the last axis (waterfall plot)
    fig.tight_layout()  
    plt.savefig(
        join(saving_folder, file_name[:-4] + ".pdf"),
        bbox_inches='tight'
    )
    plt.close()

def plot_multiple_shap_features_impact(
    shap_values_dict: dict,
    feature_values_dict: dict,
    roles: list,
    file_name: str,
    saving_folder: str,
    max_display: int = 10
) -> None:
    os.makedirs(saving_folder, exist_ok=True)
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12,6))
    
    feature_order = np.argsort(
        np.mean([np.abs(shap_values_dict[role]).mean(axis=0) for role in roles], axis=0)
    )[::-1][:max_display]

    for idx, role in enumerate(roles):    
        role_str = role if role else "All"    
        ax = axes[idx]
        shap_values = shap_values_dict[role][:, feature_order]
        feature_values = feature_values_dict[role].iloc[:, feature_order]
        feature_values = feature_values.rename(columns=feature_str_dict)

        shap_explanation = shap.Explanation(
            values=shap_values, data=feature_values
        )
        beeswarm(
            shap_explanation,
            show=False,
            max_display=max_display,
            color_bar=False,
            plot_size=None,
            ax=ax,
            order=list(range(0,max_display))
        )

        # Only display feature names for the first plot
        if idx > 0:
            ax.set_yticklabels([])
        ax.set_title(f"{role_str}")

        # Remove x-axis labels for all but the last plot
        if idx != len(roles) // 2:
            ax.set_xlabel('')
        else:
            ax.set_xlabel("SHAP Value")
    
    # Add a single colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=shap.plots.colors.red_blue)
    sm._A = []  # Needed to fix ScalarMappable issue
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.set_label('Feature Value')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(saving_folder, file_name), bbox_inches='tight')
    plt.savefig(os.path.join(saving_folder, file_name[:-4] + ".pdf"), bbox_inches='tight')
    plt.close()
