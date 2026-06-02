# Employee Performance Analysis at INX Future Inc.

## Project Overview
This project aims to identify and understand performance issues at INX Future Inc., a leading data analytics and automation solutions provider with over 15 years of global experience. By utilizing data analytics and machine learning, we analyze employee data to uncover the root causes of declining performance indexes and provide actionable insights to improve service delivery and client satisfaction — while maintaining employee morale and INX's reputation as a top employer.

## Data Collection
The dataset contains **1,200 employee records** with **28 features** covering demographics, employment information, satisfaction scores, experience metrics, and performance ratings. The target variable is `PerformanceRating` with three classes — Rating 2 (Good), Rating 3 (Excellent), and Rating 4 (Outstanding). The dataset is completely clean with no missing values and no duplicate records.

## Data Preprocessing
Data preprocessing involved the following steps:
- **Outlier Detection:** Box plots were used to investigate outliers. Long-serving employees (up to 40 years experience) were identified as genuine cases and retained.
- **Data Encoding:** Binary mapping (0/1) for OverTime and Attrition. One-Hot Encoding applied to all other categorical columns, expanding the dataset from 28 to 54 columns.
- **Train/Test Split:** 80/20 split — 960 records for training, 240 for testing.
- **Dimensionality Reduction:** Correlation threshold of ±0.01 applied. 10 noise columns dropped (e.g. Gender_Male, specific R&D roles). Final dataset: 43 features.
- **Feature Scaling:** StandardScaler fitted on training data only, then applied to the test set to prevent data leakage.

## Exploratory Data Analysis (EDA)
EDA was conducted to understand data distributions, identify correlations, and visualize trends across departments and performance ratings.

Key findings:
- Rating 3 (Excellent) dominates the dataset at over 75% — a severe class imbalance requiring targeted handling
- Performance decline is **not company-wide** — it is concentrated in **Finance (avg 2.77)** and **Sales (avg 2.86)**
- **Development (3.08)** is the top-performing department at INX
- Finance has only 49 employees but 31% of them rate at Level 2 — the worst ratio of any department
- Most outstanding performers (Rating 4) come from employees who do **not** work overtime

## Feature Engineering
A correlation-based feature selection approach was applied to identify the most predictive features. Features with near-zero correlation (between -0.01 and 0.01) with the target variable were dropped as noise. The top 3 most critical factors identified were:

1. **EmpEnvironmentSatisfaction (+0.41)** — strongest positive predictor. Employees who feel good about their work environment consistently score higher ratings.
2. **EmpLastSalaryHikePercent (+0.36)** — Rating 4 employees receive salary hikes of 20–22% compared to ~14% for ratings 2 and 3.
3. **YearsSinceLastPromotion (-0.17)** — strongest negative signal. Career stagnation is a measurable drag on performance.

## Model Building
Six classification models were trained using two different imbalance-handling strategies depending on the model type:

**SMOTE (Synthetic Minority Over-Sampling):** Applied to Logistic Regression and SVM. These models are sensitive to data distribution so a balanced training set improves their decision boundaries.

**class_weight='balanced' (Cost-Sensitive Learning):** Applied to Random Forest, XGBoost, and ANN. The algorithm is penalized more heavily for misclassifying minority classes during training without modifying the dataset.

**KNN** used distance-based voting weights (`weights='distance'`) on the original scaled dataset.

## Model Evaluation
All models were evaluated using accuracy and macro F1-score. The macro F1-score was the primary metric since it treats all three performance classes equally — unlike accuracy which is skewed by the dominant Rating 3 class.

| Model | Imbalance Strategy | Accuracy | Macro F1 |
|---|---|---|---|
| Logistic Regression | SMOTE | 77.50% | 0.68 |
| SVM | SMOTE | 82.08% | 0.67 |
| KNN | Distance Weights | 76.67% | 0.50 |
| ANN | Class Weights | 81.67% | 0.71 |
| XGBoost | Class Weights | 91.67% | 0.86 |
| **Random Forest ✅** | **Class Weights** | **93.33%** | **0.88** |

**Random Forest with GridSearchCV** was selected as the best model with best parameters: `n_estimators=200`, `max_depth=None`, `min_samples_split=10`, `class_weight='balanced'`.

## Insights and Recommendations
Feature importance and correlation analysis identified three key factors driving employee performance:

1. **Invest in the work environment in Finance and Sales** — environment satisfaction is the single biggest driver of performance and these two departments have the lowest ratings.
2. **Review salary hike structures** — top performers receive 20–22% hikes while struggling employees sit at 14%. INX should check whether Finance and Sales employees are receiving comparable increases.
3. **Address promotion bottlenecks** — going years without a promotion negatively impacts performance. INX should identify employees stuck in the same role and act on it.

## Deployment
A Streamlit web application was developed to allow HR and management to:
- View the project overview and dataset summary
- Input employee details and get a live performance rating prediction with confidence score
- Explore model performance comparison charts
- View department analysis and business recommendations

## Results
Random Forest with GridSearchCV achieved **93.33% accuracy** and a **macro F1-score of 0.88** — the strongest performance across all six models. The switch from SMOTE to cost-sensitive learning eliminated synthetic noise and improved minority class precision from ~0.50 to 0.87–0.91. The model reliably identifies both struggling employees (Rating 2) and top performers (Rating 4) without defaulting to the majority class.

## Conclusion
This project successfully identified the root causes of performance decline at INX Future Inc. The drop in performance is not a company-wide problem — it is concentrated in Finance and Sales and is driven by morale and recognition issues rather than skill gaps. Targeted improvements to the work environment, salary structures, and promotion pathways in these two departments can reverse the trend while maintaining INX's reputation as a top employer.

## Getting Started
```bash
git clone https://github.com/Risper-Wacu/EMPLOYEE-PERFORMANCE-ANALYSIS.git
```




