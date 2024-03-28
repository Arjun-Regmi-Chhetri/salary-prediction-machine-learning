
# Predicting Salaries Based on Years of Experience


This project aims to build a predictive model for estimating salaries based on an individual's years of professional experience. Leveraging a dataset containing salary information coupled with corresponding years of experience, the project employs a linear regression approach to develop and assess the model's accuracy.

## <font color="blue">Utilized Libraries</font>

- **pandas**: A powerful data manipulation tool for handling the dataset.
- **numpy**: Essential for performing numerical computations efficiently.
- **matplotlib**: Used to visualize the data distribution and model performance.
- **scikit-learn (sklearn)**: Provides convenient tools for model training, testing, and evaluation.

## Project Workflow

1. **Data Preprocessing**: The dataset is loaded from a CSV file, and any incomplete or missing entries are addressed by removing corresponding rows.

2. **Feature Extraction**: The key features, namely 'Years of Experience' and 'Salary', are extracted from the dataset.

3. **Train-Test Split**: The dataset is partitioned into training and testing subsets, with 90% of the data reserved for training and the remaining 10% for testing.

4. **Model Training**: A Linear Regression model is trained using the training data, aiming to establish a relationship between years of experience and salary.

5. **Model Testing and Prediction**: The trained model is applied to predict salaries based on years of experience. The predicted salaries are then compared against the actual salary data.

6. **Performance Evaluation**: The effectiveness of the model is assessed by evaluating the consistency between predicted salaries and actual salaries within the test set.

7. **Visualization**: Visual representations, such as scatter plots and line plots, are employed to illustrate the model's predictive performance and validate its accuracy.



## Clone the Repository

To clone the repository and access the project files, use the following command:

```bash
git clone https://github.com/Arjun-Regmi-Chhetri/salary-prediction-machine-learning
