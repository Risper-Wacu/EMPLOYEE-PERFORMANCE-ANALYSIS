# Project Summary

## 1. Requirement
The objective of this project is to analyze an employee dataset from INX Future Inc to find out why overall performance benchmarks have declined. Specifically, the business needs to identify the core features driving employee performance and build a reliable classification model to predict individual performance ratings (Ratings 2, 3, and 4). This will help management implement targeted improvements in workplace environment and compensation without impacting overall company culture.


## 2. Analysis
The dataset contains 1,200 employee records. To prepare this data for modeling, the following preprocessing steps were taken:

* **Data Transformations:** Categorical variables (such as departments and roles) were processed using One-Hot Encoding with *drop_first=True* to prevent multi-collinearity. Continuous numerical columns were normalized using *StandardScaler* to ensure scale differences wouldn't distort distance-based algorithms like SVM and KNN.
* **Feature Selection:** Variance-based filtering was used to remove 10 noise columns that had zero or near-zero variance(-0.1 to 0.1). This left 43 structural features for analysis.
* **Handling Class Imbalance:** The target variable is heavily imbalanced. I used synthetic oversampling (SMOTE) for Logistic Regression and SVM algorithms and cost-sensitive learning (*class_weight='balanced'*) was configured directly inside the models(Random Forest, XGBoost, K-NearestNeighbor and ArtificialNeuralNetworks) to control false positives and ensure balanced recall.

Six different algorithms were trained and cross-validated: Logistic Regression, SVM, KNN, a Keras Artificial Neural Network (ANN), XGBoost, and Random Forest. 

The **Random Forest** model optimized via *GridSearchCV* performed best, achieving an overall accuracy of 93.33% and a macro F1-score of 0.88. The macro F1-score confirms that the model handles minority performance classes reliably rather than just predicting the majority class.

## 3. Summary
### Key Findings
1. Environment Satisfaction (0.41) — the strongest signal in the entire dataset. Employees who feel good about where they work consistently perform better. With a correlation of 0.41, it sits clearly ahead of every other feature, making the work environment the single most important factor in predicting performance at INX.
2. Last Salary Hike Percentage (0.36) — the second strongest predictor. Employees who received meaningful salary increases tended to have higher performance ratings. 
3. Years Since Last Promotion (-0.17) — the strongest negative signal in the dataset. The longer an employee goes without a promotion, the lower their performance rating tends to be. Career stagnation is a real and measurable drag on output at INX.
4. Department Trends: A breakdown of departmental averages showed that Finance (2.77) and Sales (2.86) have the lowest performance levels, while Development (3.08) has the highest.

## Business Recommendations
1. Invest in the work environment, especially in Finance and Sales. Environment satisfaction was the single biggest driver of performance in this dataset, and Finance and Sales are the two departments with the lowest average performance ratings. Improving how employees feel about their environment results to better performance.
2. The data showed that top performers receive salary hikes between 20-22% while average and struggling employees sit around 14%. INX should check whether Finance and Sales employees are receiving comparable increases — because pay recognition and performance clearly move together.
3. Address promotion bottlenecks. The data shows that going years without a promotion negatively impacts performance. INX should identify employees who have been stuck in the same role for years, particularly in Finance and Sales, whether that means promoting them, moving them, or having an honest conversation about where they are headed.
