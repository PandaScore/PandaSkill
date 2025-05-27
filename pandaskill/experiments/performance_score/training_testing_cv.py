from pandaskill.experiments.performance_score.visualization import *
from pandaskill.experiments.general.utils import ROLES
from pandaskill.experiments.general.metrics import compute_ece
from pandaskill.experiments.general.visualization import *
from pandaskill.libs.performance_score.base_model import BaseModel
import logging
import numpy as np
import os
from os.path import join
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
from typing import Tuple, Optional
import yaml

def compute_performance_scores_cv_loop(
    data: pd.DataFrame, 
    features: list, 
    Model: BaseModel,
    model_parameters: dict, 
    training_config: dict,
    experiment_dir: str,
    evaluation_config: dict
) -> Tuple[pd.DataFrame, dict]:
    train_game_ids, test_game_ids = _compute_game_id_cross_validation(data, training_config["n_splits"], training_config["random_state"])
    metrics_list, performance_scores_list, models_list, features_importance_list, calibration_data = [], [], [], [], []
    for fold_index, (train_game_ids_fold, test_game_ids_fold) in enumerate(zip(train_game_ids, test_game_ids)):
        roles = ROLES if training_config["one_model_per_role"] else [None]
        metrics_fold, models_fold, feature_importances_fold, calibration_data_fold = {}, {}, {}, {}
        for role in roles:
            logging.info(f"Training and evaluating model for role `{role}` and fold `{fold_index}`")
            X_train, y_train, _ = _get_role_cv_training_data(data, features, train_game_ids_fold, role)
            model_fold = _train_model(X_train, y_train, Model, model_parameters)
            models_fold[role] = model_fold

            X_test, y_test, index_test = _get_role_cv_training_data(data, features, test_game_ids_fold, role)
            y_prob = model_fold.predict_proba(X_test)[:,1]
            metrics_fold[role] = _evaluate_game_perf_model(y_prob, y_test)
            calibration_data_fold[role] = [y_prob, y_test]

            feature_importances_fold[role] = model_fold.compute_features_importance()            

            performance_scores = model_fold.compute_performance_scores(X_test)
            performance_scores_fold_df = pd.DataFrame(data=performance_scores, index=index_test, columns=["performance_score"])
            performance_scores_list.append(performance_scores_fold_df)

        metrics_list.append(metrics_fold)
        models_list.append(models_fold)
        features_importance_list.append(feature_importances_fold)
        calibration_data.append(calibration_data_fold)

    _visualize_shap_values_single_game(
        data, features, test_game_ids, evaluation_config["specific_games_analysis"], models_list, roles, experiment_dir
    )
    if evaluation_config["visualize_shap_values_distributions"]:
        _visualize_shap_values_distributions(
            data, features, test_game_ids, models_list, roles, experiment_dir
        )

    plot_all_models_calibration(calibration_data, experiment_dir)

    performance_scores_df = pd.concat(performance_scores_list, axis=0)
    performance_scores_df = performance_scores_df.sort_index()

    evaluation_metrics = _format_evaluation_metrics(metrics_list, features_importance_list, features)

    return performance_scores_df, evaluation_metrics

def _visualize_shap_values_single_game(
    data: pd.DataFrame, 
    features: list, 
    game_ids_cv: np.ndarray, 
    specific_game_ids: list,
    models_cv: dict, 
    roles: list,
    experiment_dir: str
) -> None:    
    if not specific_game_ids:
        return
    
    saving_folder = join(experiment_dir, "shap_values")
    explainers_dict = {
        role: [shap.Explainer(model_role_dict[role].model) for model_role_dict in models_cv]
        for role in roles
    }
    for game_id in specific_game_ids:
        fold_idx = next(i for i, g in enumerate(game_ids_cv) if game_id in g)
        game_saving_folder = os.path.join(saving_folder, f"game_{game_id}")
        os.makedirs(game_saving_folder, exist_ok=True)

        X_game = data.xs(game_id, level="game_id")
        for player_id, row in X_game.iterrows():
            role = row["role"]
            player_name = row["player_name"]

            X  = row[features].values.reshape(1, -1)
            model_obj = models_cv[fold_idx][role]
            X_norm = model_obj.scaler.transform(X)

            explainer = explainers_dict[role][fold_idx]
            shap_vals = explainer(X_norm)

            plot_shap_game_features_impact(
                explainer=explainer,
                shap_values=shap_vals.values[0],
                feature_values_df= pd.Series(X[0], index=features, name=player_id),
                file_name= f"{role}_{game_id}_{player_name}.png",
                saving_folder=game_saving_folder,
                nb_features_to_display=10,
                show_xlabel=True
            )

def _visualize_shap_values_distributions(
    data: pd.DataFrame, 
    features: list, 
    game_ids_cv: np.ndarray,
    models_cv: dict, 
    roles: list,
    experiment_dir: str
) -> None:    
    saving_folder = join(experiment_dir, "shap_values")
    shap_values_dict = {}
    feature_values_dict = {}
    for role in roles:
        shap_values, feature_values, all_game_ids, explainers = [], [], [], []
        for game_ids, model_role_dict in zip(game_ids_cv, models_cv):
            game_data = data[data.index.get_level_values("game_id").isin(game_ids)]
            if role is not None:
                game_data = game_data[game_data["role"] == role]
            X = game_data.loc[:, features]
            explainer_fold, shap_values_fold = model_role_dict[role].compute_shap_values(X.values)
            explainers.append(explainer_fold)
            shap_values.append(shap_values_fold)
            feature_values.append(X)
            all_game_ids.extend(game_ids)
        feature_values_df = pd.concat(feature_values, axis=0)
        shap_values_array = np.concatenate(shap_values, axis=0)
        shap_values_dict[role] = shap_values_array
        feature_values_dict[role] = feature_values_df
    
    file_name = "combined_shap_features_impact.png"
    plot_multiple_shap_features_impact(
        shap_values_dict=shap_values_dict,
        feature_values_dict=feature_values_dict,
        roles=roles,
        file_name=file_name,
        saving_folder=saving_folder,
        max_display=len(features)
    )
            
def _compute_game_id_cross_validation(
    data: pd.DataFrame,
    n_splits: int,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    game_ids = data.index.get_level_values("game_id")
    unique_game_ids = game_ids.unique()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_game_ids, test_game_ids = [], []
    for train_game_id_index, test_game_id_index in kfold.split(unique_game_ids):
        train_game_ids.append(unique_game_ids[train_game_id_index])
        test_game_ids.append(unique_game_ids[test_game_id_index])
    return train_game_ids, test_game_ids

def _get_role_cv_training_data(
    data: pd.DataFrame, 
    features: list, 
    game_ids: np.ndarray, 
    role: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    data = data[data.index.get_level_values("game_id").isin(game_ids)]

    if role is not None:
        data = data[data["role"] == role]

    X = data.loc[:, features].values
    y = data.loc[:, "win"].values

    return X, y, data.index


def _train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    Model: BaseModel, 
    parameters: dict
) -> BaseModel: 
    model = Model(**parameters)
    model.fit(X, y)
    return model

def _evaluate_game_perf_model(
    y_prob: np.ndarray,
    y: np.ndarray, 
) -> dict:
    nbins = 25

    metrics = {
        "accuracy": float(accuracy_score(y, y_prob > 0.5)),
        "f1": float(f1_score(y, y_prob > 0.5)),
        "auc": float(roc_auc_score(y, y_prob)),
        "ece": float(compute_ece(y, y_prob, nbins)),
    }

    return metrics 

def _format_evaluation_metrics(
    metrics: dict, 
    features_importance_dict: dict, 
    features: list
) -> dict:
    roles = list(metrics[0].keys())
    metrics = {
        metric_name: {
            role: {
                "cv": (value_list := [metrics_fold[role][metric_name] for metrics_fold in metrics]),
                "mean": float(np.mean(value_list)),
                "std": float(np.std(value_list))
            }
            for role in roles
        }
        for metric_name in metrics[0][roles[0]].keys()
    }
    for metric_name in metrics.keys():
        metric_values = [metrics[metric_name][role]["mean"] for role in roles]
        metrics[metric_name]["overall"] = {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values))
        }

    features_importance_dict = {
        role: {
            feature_name: {
                "cv": (value_list := [feature_importances_fold[role][feature_index].tolist() for feature_importances_fold in features_importance_dict]),
                "mean": float(np.mean(value_list)),
                "std": float(np.std(value_list))
            }
            for feature_index, feature_name in enumerate(features)
        }
        for role in roles
    }

    evaluation_metrics = {
        "metrics": metrics,
        "features_importance": features_importance_dict
    }

    return evaluation_metrics
