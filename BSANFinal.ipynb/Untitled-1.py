# %%
pip install pandas numpy statsmodels

# %%
import os
os.chdir("..")

print(os.listdir())

combine = pd.read_csv("2025 Combine Results.csv")

# %%
combine["Pick"] = combine["Drafted (tm/rnd/yr)"].str.extract(
    r'(\d+)(?=th pick|st pick|nd pick|rd pick)'
).astype(float)

# %%
combine = combine.replace(-9999, np.nan)
combine = combine.replace(r"^\s*$", np.nan, regex=True)

# %%
model_df = combine[[
    "Pick",
    "40yd",
    "Wt",
    "Vertical",
    "Bench"
]].copy()

# %%
for col in ["Pick", "40yd", "Wt", "Vertical", "Bench"]:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# %%
model_df = model_df.dropna()

print(model_df.shape)
print(model_df.head())

# %%
X = model_df[["40yd", "Wt", "Vertical", "Bench"]]
y = model_df["Pick"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())

# %%
model_df["predicted_pick"] = model.predict(X)
print(model_df[["Pick", "predicted_pick"]].head(10))

# %%
model_df = combine[[
    "Pick",
    "40yd",
    "Wt",
    "Vertical",
    "Bench",
    "Pos"
]].copy()

# %%
model_df = pd.get_dummies(model_df, columns=["Pos"], drop_first=True)

# %%
for col in ["Pick", "40yd", "Wt", "Vertical", "Bench"]:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# %%
model_df = model_df.dropna()

print(model_df.shape)

# %%
X = model_df.drop(columns="Pick")
y = model_df["Pick"]

# %%
X = X.astype(float)
y = y.astype(float)

# %%
import statsmodels.api as sm

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# %%
print(model.summary())

# %%
model_df["predicted_pick"] = model.fittedvalues.values

# %%
model_df[["Pick", "predicted_pick"]].head(10)

# %%
model_df["percentile"] = pd.qcut(model_df["Pick"], 10, labels=False)

# %%
def assign_group(p):
    if p == 0:
        return "Top 10%"
    elif p == 1:
        return "Next 10%"
    elif p == 2:
        return "Next 10% After That"
    else:
        return "Remaining 70%"

# %%
model_df["group"] = model_df["percentile"].apply(assign_group)

# %%
group_summary = model_df.groupby("group").apply(
    lambda x: pd.Series({
        "Count": len(x),
        "Average Actual Pick": x["Pick"].mean(),
        "Average Predicted Pick": x["predicted_pick"].mean(),
        "Mean Absolute Error": abs(x["Pick"] - x["predicted_pick"]).mean()
    })
).reset_index()

# %%
group_order = ["Top 10%", "Next 10%", "Next 10% After That", "Remaining 70%"]

group_summary["group"] = pd.Categorical(
    group_summary["group"],
    categories=group_order,
    ordered=True
)

group_summary = group_summary.sort_values("group")
print(group_summary)

# %%
rankings = pd.read_csv("2025 Player Rankings.csv", encoding="latin1")
rankings.columns = rankings.columns.str.strip()

# %%
print(rankings.head())
print(rankings["Name"].nunique())

# %%
def clean_name(name):
    name = str(name).lower()
    name = name.replace(".", "")
    name = name.replace(",", "")
    name = name.replace(" jr", "")
    name = name.replace(" sr", "")
    name = name.replace(" iii", "")
    name = name.replace(" ii", "")
    return name.strip()

combine["player_clean"] = combine["Player"].apply(clean_name)
rankings["Name_clean"] = rankings["Name"].apply(clean_name)

# %%
df = pd.merge(
    combine,
    rankings,
    left_on="player_clean",
    right_on="Name_clean",
    how="inner"
)

print(df.shape)


