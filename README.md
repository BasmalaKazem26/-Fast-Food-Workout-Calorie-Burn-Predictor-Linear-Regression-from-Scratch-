
#  Fast Food & Workout Calorie Burn Predictor

An end-to-end Data Science and Machine Learning project developed in **Python**[cite: 1]. This repository focuses on exploring fast-food nutritional data, executing rigorous data engineering steps, and implementing custom machine learning algorithms **from scratch** using NumPy and pure Python mathematics[cite: 1].

---

##  Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Features](#-dataset-features)
- [Project Workflow & Implementation](#-project-workflow--implementation)
- [Tech Stack](#-tech-stack)


---

##  Project Overview
The primary objective of this project is to build a robust predictive model that analyzes the relationship between fast-food nutritional contents (calories, fats, carbs, etc.), consumer attributes, and workout duration metrics. Instead of relying solely on high-level libraries, this project highlights a deep foundational understanding of how algorithms work under the hood by implementing feature scaling and gradient descent optimization manually.

---

##  Dataset Features
The dataset (`fast_food_cheat_meal_workout_burn.csv`) contains multiple attributes related to:
* **Nutritional Information:** Total calories, fat grams, saturated fat, carbohydrates, sugar, protein, and sodium levels[cite: 1].
* **Consumer Demographics & Metrics:** Age, gender, weight, and fitness levels[cite: 1].
* **Workout & Burn Metrics:** Thermic effect of food, net caloric surplus, and specific physical activity burn times (walking, cycling, running, and HIIT burn times in minutes)[cite: 1].

---

##  Project Workflow & Implementation

### 1. Data Inspection & Cleaning
* Loaded the dataset using **Pandas** to inspect dimensions, data types, and statistical summaries (`describe()`)[cite: 1].
* Checked for missing values (`isnull().sum()`) and duplicate rows (`duplicated().sum()`) to ensure data integrity[cite: 1].

### 2. Categorical Encoding
* Identified object and categorical data types (excluding identifiers like `order_id`)[cite: 1].
* Applied **One-Hot Encoding** using `pd.get_dummies()` to transform categorical features into a machine-learning-friendly numerical format[cite: 1].

### 3. Feature Engineering & Manual Scaling
* Selected key numerical columns for standardization (calories, macros, consumer metrics, and workout durations)[cite: 1].
* Developed a custom **Standard Scaler** function (`standerd_scaler`) from scratch using pure Python logic to compute the mean, variance, and standard deviation, normalizing features to a standard normal distribution without relying on external pre-built scalers[cite: 1].

### 4. Custom Machine Learning Model (Gradient Descent Linear Regression)
* Split the processed dataset into training and testing sets using `train_test_split` (with an 80/20 ratio and fixed random state)[cite: 1].
* Engineered a custom object-oriented class `GradientDescentLinearRegression` featuring initialization, training (`fit`), and prediction (`predict`) methods[cite: 1].
* Implemented matrix dot products (`np.dot`) via **NumPy** to calculate predictions, compute gradients for weights (`dw`) and bias (`db`), and iteratively optimize parameters over defined epochs[cite: 1].

### 5. Model Evaluation
* Evaluated the predictive performance of the custom regression model on unseen test data using **Mean Squared Error (MSE)**[cite: 1].

---

##  Tech Stack

* **Programming Language:** Python[cite: 1]
* **Data Manipulation & Math:** Pandas, NumPy[cite: 1]
* **Machine Learning Utilities:** Scikit-Learn (`train_test_split`)[cite: 1]
* **Data Visualization:** Matplotlib[cite: 1]

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git)
