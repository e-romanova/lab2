"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    row_count = len(df)
    missing_by_column = df.isna().sum()
    adults_over_30_count = (df['Age'] > 30).sum()
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean()
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean()
    highest_fares = df["Fare"].nlargest(5)

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )

print(analyze_titanic(TitanicInput(csv_path="titanic.csv")))