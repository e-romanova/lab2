import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])

features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
target = "MRI_Count"

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, feature in enumerate(features):
    sns.scatterplot(
        data=df,
        x=feature,
        y=target,
        hue="Gender",
        ax=axes[i],
    )
    axes[i].set_title(f"{feature} vs {target}")

plt.tight_layout()
plt.show()


for gender in ["Male", "Female"]:
    group = df[df["Gender"] == gender]

    correlations = {}

    for feature in features:
        correlations[feature] = group[feature].corr(group[target], method="pearson")

    corr_df = pd.DataFrame([correlations], index=[gender])

    plt.figure(figsize=(9, 2))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Корреляции с MRI_Count: {gender}")
    plt.tight_layout()
    plt.show()