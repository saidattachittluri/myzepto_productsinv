import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# LOAD ONCE & COMMIT FALLBACK CSV
try:
    df_raw = sns.load_dataset("titanic")
except Exception:
    df_raw = pd.read_csv("titanic.csv")

df_raw.to_csv("titanic.csv", index=False)

print("=== RAW DATASET PROFILE ===")
print(f"Shape: {df_raw.shape}")
print(df_raw.info())
print(df_raw.describe(include="all"))

# Report missing value percentages
missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
missing_report = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\n=== MISSING VALUE PERCENTAGES ===")
print(missing_report.apply(lambda x: f"{x:.2f}%"))

# -----------------------------------------------------------------------------
# 2. THRESHOLD-RULE CLEANING & JUSTIFICATION
# Rules: <5% -> drop rows; 5%-30% -> impute; >30% -> drop column or encode 'Missing'
# -----------------------------------------------------------------------------
df = df_raw.copy()

# deck (77.10% missing): Exceeds 30%. Imputation creates excessive noise; drop column.
df.drop(columns=["deck"], inplace=True)

# age (19.87% missing): Between 5% and 30%. Impute via median to resist outliers.
df["age"] = df["age"].fillna(df["age"].median())

# embarked (0.22% missing) & embark_town (0.22% missing): Under 5%. Drop missing rows.
df.dropna(subset=["embarked", "embark_town"], inplace=True)

# Drop redundant text columns retaining standardized equivalents
df.drop(columns=["alive", "class"], inplace=True)

print(f"\nCleaned dataset shape post-cleaning: {df.shape}")

# 3. UNIVARIATE ANALYSIS: IQR OUTLIERS & FARE SKEWNESS
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Age Distribution")
sns.boxplot(x=df["age"], ax=axes[0, 1], color="lightblue")
axes[0, 1].set_title("Age Boxplot")

sns.histplot(df["fare"], kde=True, ax=axes[1, 0], color="coral")
axes[1, 0].set_title("Fare Distribution")
sns.boxplot(x=df["fare"], ax=axes[1, 1], color="lightsalmon")
axes[1, 1].set_title("Fare Boxplot")
plt.tight_layout()
plt.savefig("plots/01_univariate_dist.png")
plt.close()

def compute_iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound

age_outliers, _, _ = compute_iqr_outliers(df["age"])
fare_outliers, _, _ = compute_iqr_outliers(df["fare"])

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print("\n=== UNIVARIATE ANALYSIS ===")
print(f"Age IQR Outliers: {age_outliers}")
print(f"Fare IQR Outliers: {fare_outliers}")
print(f"Fare Distribution Metrics -> Mean: {fare_mean:.2f}, Median: {fare_median:.2f}, Mode: {fare_mode:.2f}")
# Skewness conclusion: Mean (32.10) > Median (14.45) > Mode (8.05) -> strongly RIGHT-SKEWED.

# -----------------------------------------------------------------------------
# 4. BIVARIATE ANALYSIS: BOOLEAN MASKS & 6x6 HEATMAP
# -----------------------------------------------------------------------------
rate_female = df[df["sex"] == "female"]["survived"].mean()
rate_male = df[df["sex"] == "male"]["survived"].mean()

rate_p1 = df[df["pclass"] == 1]["survived"].mean()
rate_p2 = df[df["pclass"] == 2]["survived"].mean()
rate_p3 = df[df["pclass"] == 3]["survived"].mean()

rate_f_p1 = df[(df["sex"] == "female") & (df["pclass"] == 1)]["survived"].mean()
rate_f_p2 = df[(df["sex"] == "female") & (df["pclass"] == 2)]["survived"].mean()
rate_f_p3 = df[(df["sex"] == "female") & (df["pclass"] == 3)]["survived"].mean()
rate_m_p1 = df[(df["sex"] == "male") & (df["pclass"] == 1)]["survived"].mean()
rate_m_p2 = df[(df["sex"] == "male") & (df["pclass"] == 2)]["survived"].mean()
rate_m_p3 = df[(df["sex"] == "male") & (df["pclass"] == 3)]["survived"].mean()

print("\n=== BIVARIATE SURVIVAL RATES (BOOLEAN MASKING) ===")
print(f"Female Survival Rate: {rate_female:.4f} | Male: {rate_male:.4f}")
print(f"Pclass Survival Rates -> P1: {rate_p1:.4f} | P2: {rate_p2:.4f} | P3: {rate_p3:.4f}")
print(f"Female by Pclass -> P1: {rate_f_p1:.4f}, P2: {rate_f_p2:.4f}, P3: {rate_f_p3:.4f}")
print(f"Male by Pclass   -> P1: {rate_m_p1:.4f}, P2: {rate_m_p2:.4f}, P3: {rate_m_p3:.4f}")

# 6x6 Numeric Matrix excluding redundant boolean flags 'adult_male' and 'alone'
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("6x6 Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("plots/02_bivariate_correlations.png")
plt.close()

# 5. MULTIVARIATE DATA STORY (4 DISTINCT CHARTS)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Chart 1: Class and Gender Interaction
sns.barplot(data=df, x="pclass", y="survived", hue="sex", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("1. Survival by Pclass and Sex")

# Chart 2: Age distribution across survival and passenger class
sns.violinplot(data=df, x="pclass", y="age", hue="survived", split=True, ax=axes[0, 1], palette="muted")
axes[0, 1].set_title("2. Age Distribution vs Survival by Pclass")

# Chart 3: Fare vs Age scatter highlighting survival
sns.scatterplot(data=df, x="age", y="fare", hue="survived", alpha=0.7, ax=axes[1, 0], palette="coolwarm")
axes[1, 0].set_title("3. Fare vs Age by Survival Status")

# Chart 4: Family size interaction (sibsp + parch)
df["family_size"] = df["sibsp"] + df["parch"]
sns.pointplot(data=df, x="family_size", y="survived", ax=axes[1, 1], color="darkcyan")
axes[1, 1].set_title("4. Survival Rate across Family Size")

plt.tight_layout()
plt.savefig("plots/03_multivariate_story.png")
plt.close()

# 6. EXPLORATORY Z-SCORE STANDARDIZATION SANITY CHECK
age_mean, age_std = df["age"].mean(), df["age"].std()
fare_mean, fare_std = df["fare"].mean(), df["fare"].std()

df["age_z"] = (df["age"] - age_mean) / age_std
df["fare_z"] = (df["fare"] - fare_mean) / fare_std

print("\n=== EXPLORATORY Z-SCORE SANITY CHECK ===")
print(f"Original Age -> Mean: {age_mean:.2f}, Std: {age_std:.2f}")
print(f"Scaled Age   -> Mean: {df['age_z'].mean():.4f}, Std: {df['age_z'].std():.4f}")
print(f"Original Fare -> Mean: {fare_mean:.2f}, Std: {fare_std:.2f}")
print(f"Scaled Fare   -> Mean: {df['fare_z'].mean():.4f}, Std: {df['fare_z'].std():.4f}")