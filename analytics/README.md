# Module 2 — Analytics & Machine Learning Pipeline

## 1. Missing Value Imputation Strategy
Evaluating raw columns against the assignment's threshold criteria:
* **`deck` (77.10% missing)**: Exceeds the 30% ceiling. Imputing over three-quarters of the data would inject high-variance synthetic assumptions into downstream modeling; the column is dropped.
* **`age` (19.87% missing)**: Falls within the 5%–30% band. Imputed with the column **median** (28.0) to resist the positive skew visible in the distribution.
* **`embarked` / `embark_town` (0.22% missing)**: Below the 5% threshold (2 rows missing). Dropped to maintain clean categorical consistency without distortion.

## 2. Univariate Distribution & Outliers
* **Outliers (IQR Method)**: `age` exhibits **66 outliers** (primarily elderly travelers above the 66-year boundary). `fare` exhibits **116 outliers**, driven by lavish first-class suites exceeding $65.
* **Skewness of `fare`**:
  $$\text{Mean: } \$32.10 > \text{Median: } \$14.45 > \text{Mode: } \$8.05$$
  Because the mean is pulled heavily to the right of both the median and the mode, the fare distribution is **strongly right-skewed**.

## 3. Bivariate Relationships & Correlation Interpretation
* **Survival Rates**:
  * Gender: Females survived at **74.20%**, Males at **18.89%**.
  * Class: First Class = **62.96%**, Second Class = **47.28%**, Third Class = **24.24%**.
  * Class + Gender Interaction: First-class females had a **96.81%** survival rate; third-class males dropped to **13.54%**.
* **Off-Diagonal Correlation Extremes (6×6 Matrix)**:
  1. **`fare` vs. `pclass` ($r = -0.55$)**: Strong negative correlation confirming that higher socioeconomic classes (lower class number, 1) paid substantially higher ticket prices.
  2. **`sibsp` vs. `parch` ($r = +0.41$)**: Moderate positive correlation reflecting that family units boarded jointly across both horizontal (siblings/spouses) and vertical (parents/children) relationships.

## 4. Multivariate Data Story (Summary of 4 Charts)
1. **Survival by Class and Gender**: Confirms the "women and children first" maritime evacuation protocol, reinforced by class privilege—first-class females were prioritized over all other cohorts.
2. **Age vs. Survival by Class**: Highlights that children in Classes 1 and 2 survived at near-complete rates, whereas third-class child mortality remained elevated.
3. **Fare vs. Age Distribution**: High ticket prices cluster almost exclusively with first-class survivors; passengers paying under $20 suffered high mortality regardless of age.
4. **Family Size Impact**: Single travelers and large families (>4 members) experienced reduced survival rates, whereas small family units (2–4 members) achieved optimal survival due to collective evacuation without excessive logistical burden.

## 5. Stratified Split Justification
The target variable exhibits an imbalanced baseline: **61.6% deceased vs. 38.4% survived**. A random split risks unrepresentative partition divergence between training and test sets. Stratified sampling preserves the identical 62:38 class ratio across splits, preventing evaluation bias.

## 6. Model Comparison Table

### Classification Models
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7978 | 0.7500 | 0.7246 | 0.7371 | 0.8498 |
| **Decision Tree (depth=3)** | 0.8146 | 0.8182 | 0.6522 | 0.7258 | 0.8291 |
| **Random Forest (Tuned)** | **0.8315** | **0.8276** | **0.7273** | **0.7742** | **0.8752** |

### Regression Model (Predicting `fare`)
| Model | MAE | RMSE | $R^2$ | Adjusted $R^2$ |
| :--- | :--- | :--- | :--- | :--- |
| **Multivariate Linear Regression** | $19.42 | $34.18 | 0.3812 | 0.3598 |

* **Heteroscedasticity Analysis**: The residual plot exhibits a distinct funnel shape expanding toward higher predicted values, confirming **heteroscedasticity**. Prediction error grows larger for expensive ticket categories.

## 7. Class Imbalance Comparison
* **Baseline RF**: F1 = 0.7619
* **Balanced Class Weights**: F1 = 0.7586
* **SMOTE (Train Fold Only)**: F1 = 0.7481
* **Finding**: The baseline model achieved the highest F1 score. Because the class ratio (~62:38) is moderately balanced rather than severely skewed (e.g., 99:1), artificial synthetic oversampling generated boundary noise along the minority decision boundary, slightly degrading test precision.

## 8. Final Deployment Recommendation
We recommend deploying the **Tuned Random Forest Pipeline**. It demonstrates the strongest overall balance across all core metrics: the highest **Accuracy (83.15%)**, the best discriminative power (**ROC-AUC: 0.8752**), and the highest **F1 Score (0.7742)**. It protects against overfitting via internal bagging and regularization constraints, outperforming Logistic Regression on non-linear interactions while maintaining superior recall over the constrained Decision Tree.