# Video Game Sales Analysis and Classification

This Python project analyzes video game records from 1980 to 2024. It cleans the data, explores sales and release patterns, creates visualizations, and compares four classifiers that predict whether a game's recorded total sales are above the dataset median.

## Project goals

1. Explore the most frequently recorded genres and platforms.
2. Compare recorded sales across North America, Japan, PAL, and Other regions.
3. Examine how game release counts vary by year and decade.
4. Explore the relationship between critic scores and recorded total sales.
5. Compare average recorded sales across genres and platforms.
6. Classify games into lower-sales and higher-sales groups, then compare four machine learning models.

## Dataset

The project uses [Video Game Sales & Industry Data (1980–2024)](https://www.kaggle.com/datasets/bhushandivekar/video-game-sales-and-industry-data-1980-2024), downloaded from Kaggle. The raw CSV has 64,016 records and 14 fields, including title, console, genre, publisher, critic score, release date, total sales, and regional sales. Sales are measured in millions of copies.

The dataset contains substantial missing information. In the supplied copy, `total_sales` is missing for approximately 70.44% of records. The analysis uses known total sales for sales summaries and model targets. Missing sales are not treated as zero. Check the Kaggle dataset page for its current license and terms before redistributing the data.

## Data cleaning and feature engineering

The script trims text fields, converts numeric fields, parses dates, marks negative sales and critic scores outside 1–10 as missing, removes duplicate rows, and removes records without title, console, or genre. It fills missing publisher and developer values with `Unknown`.

It derives `release_year`, `release_decade`, `critic_score_category`, `regional_sales_sum`, and `na_sales_share`. Numeric predictors are median-imputed and scaled. Categorical predictors are filled with their most frequent value and one-hot encoded inside a scikit-learn pipeline.

## Exploratory analysis and visualizations

The script creates these charts in `visualizations/`:

| File | Visualization |
|---|---|
| `F1_genre_counts.png` | Top genres by number of games |
| `F2_regional_sales.png` | Recorded sales by region |
| `F3_average_sales_by_genre.png` | Genres by average recorded sales |
| `F4_games_by_year.png` | Games released by year |
| `F5_games_by_decade.png` | Games released by decade |
| `F6_critic_score_vs_sales.png` | Critic score and recorded sales |
| `F7_sales_by_score_category.png` | Average sales by critic score category |
| `F8_average_sales_by_platform.png` | Platforms by average recorded sales |
| `F9_sales_distribution.png` | Distribution of recorded total sales |
| `F10_regional_sales_correlation.png` | Correlations between regional sales |
| `F11_decision_tree_classifier.png` | First three levels of the Decision Tree classifier |
| `F12_random_forest_feature_importance.png` | Top 20 Random Forest features by importance |

## Classification task

The target is created from known `total_sales`: games with sales **above the median** are labeled High (1); games at or below the median are labeled Low (0). The predictors are critic score, release year, console, and genre. Sales columns are excluded from predictors to avoid leaking the target into the model.

The data is divided into training and test sets using an 80/20 split, `random_state=42`, and stratification. Four classifiers use the same split and preprocessing:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors (KNN) Classifier

Each model is evaluated using accuracy, macro precision, macro recall, and macro F1. The script ranks the models by macro F1 and identifies the highest-scoring model as the best fit for this comparison. It also prints the selected model's confusion matrix and classification report. Results are specific to this target definition and test split; they are not comparable to regression metrics such as MAE or RMSE.

## Requirements and setup

Python 3 and these packages are required:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Put `Video Games Sales (1980-2024) - Raw.csv` in the same directory as `Project.py`, then run:

```bash
python Project.py
```

The script displays the visualizations and writes result files to the project directory.

## Generated outputs

- `Video_Games_Sales_Cleaned.csv` — cleaned data with derived fields.
- `Video_Games_Sales_Featured.csv` — prepared data with engineered features.
- `Classification_Model_Evaluation_Results.csv` — comparison of all four classifiers.
- `Logistic_Regression_Confusion_Matrix.csv` — Logistic Regression confusion matrix.
- `Random_Forest_Confusion_Matrix.csv` — Random Forest confusion matrix.
- `Decision_Tree_Confusion_Matrix.csv` — Decision Tree confusion matrix.
- `KNN_Confusion_Matrix.csv` — KNN confusion matrix.
- `visualizations/` — ten EDA charts and two model charts.

## Limitations

- Total sales are missing for most records, so the modeling sample is a subset of the dataset and may not represent every game.
- Critic scores are also missing for many games; missing predictor values are imputed within the training pipeline.
- The High/Low label depends on the median sales threshold and answers a classification question, not a question about exact sales amounts.
- Model selection uses one train/test split. Cross-validation and parameter tuning could provide a more robust comparison.
- Recorded regional sales have differing coverage and should not be assumed to represent complete market totals.

## Repository contents

```text
.
├── Project.py
├── README.md
├── Video Games Sales (1980-2024) - Raw.csv
├── Video_Games_Sales_Cleaned.csv
├── Video_Games_Sales_Featured.csv
├── Classification_Model_Evaluation_Results.csv
└── visualizations/
```
