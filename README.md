# HDB-Price-Machine-Learning
Explore HDB Pricing by building and testing some basic Machine Learning Models

# Machine Learning Fundamentals with HDB Data:
All Train-Test is split with 70-30 ratio.

### 1. Data Cleaning and Exploration
Datasets for this project is from https://data.gov.sg/dataset/resale-flat-prices which consists of resale HDB flat prices from 1990 to 2020.
There are 5 datasets with slightly differing columns. 'remaining_lease' column is computed for all Datasets to be used. This is saved as "Resale_Flats_Dataset.csv"

Upon EDA, I found that resale flats of the more recent years were more relevant as they had more consistent average pricing. Resale flat transactions from January of 2012 onwards (inclusive) were taken and saved as "Resale_Flats_Dataset_2012_Onwards.csv". This Dataset will be used in the following Machine Learning models.

### 2. Linear Regression
There were various collinear columns in the Dataset. They were identified with a high VIF and dropped from the dataset.
Further preprocessing was done - One Hot Enconded for Linear Regression. This OHE dataset is saved as "Resale_Flats_Dataset_2012_Onwards_Non_Collinear_OHE.csv".

Default parameters were used for Linear Regression and it gave a R2 value of 82.55% and RMSE of $58,835.

### 3. Decision Tree and Random Forest
Using the One Hot Encoded Dataset, a Decision Tree was fit to the data using higher min_samples_leaf of 20 and min_samples_split of 40 to try to prevent overfitting.
Decision Tree gave a R2 of 90% with RMSE of $44,500

The Random Forest used the same dataset with n_estimators set to 200.
Random Forest gave a R2 of 91.68% with RMSE of $40,619.

Both Decision Treee and Random Forest gave better results than Linear Regression. This might be better for real world application as it is more interpertable and less trade off between time taken for model building and R2 score

### 4. Gradient Boosting
Initial model trained with default parameters did not give an ideal result, with only 71.89% for R2 and RMSE of $60,409

With RandomizedSearchCV on parameters, namely: learning_rate, n_estimators, max_depth, min_samples_leaf, min_samples_split,
with 50 random iterations, the best model chosen had a R2 of 92.13% and RMSE of $38,052.
However there are certain downsides of achieving slightly higher R2. The parameters chosen might not be the most optimal (only 50 possible combinations were tested) and the training time for these 50 parameters was long (2hrs) as compared to the other models.

Despite having the best overall result, Gradient Boosting is time consuming to build and is less intepretable than Decision Tree and Linear Regression.

# Future Improvements/Ideas
- Normalising Data
- Feature Selection
- Trying more parameters on GridSearchCV and higher fold cross validation (currently only 2)
- Trying XGBoost
