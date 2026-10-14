"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    target = "MPI_Count" 

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr ={}
    for feature in features:
        men_corr[feature] = float(men[feature].corr(men[target]), method="pearson")

    women_corr ={}
    for feature in features:
        women_corr[feature] = float(women[feature].corr(women[target]), method="pearson")

    strongest_feature = ""
    strongest_correlation = -1.0

    for feature in features:
        men_value = abs(men_corr[features])
        women_value = abs(women_corr[features])
        if pd.notna(men_value) and men_value > strongest_correlation:
            strongest_correlation = men_value
            strongest_feature = feature
        if pd.notna(women_value) and women_value > strongest_correlation:
                strongest_correlation = women_value
                strongest_feature = feature

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature,
)
