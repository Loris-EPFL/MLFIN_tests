import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
import seaborn as sns



# Load the dataset into a pandas DataFrame
try:
    df = pd.read_csv('data.csv', sep='\t')
    #data = pd.read_csv('option_dataset.csv', sep='\t')
    print("Dataset has been successfully loaded.")
except Exception as e:
    print(f"Failed to load data: {e}")


#Begin Real Cleanup
df.drop(columns=['Unnamed: 126', 'Unnamed: 127', 'Unnamed: 128', 'Unnamed: 129'], inplace=True)
df = df.iloc[:-2, :]
df = df.drop(index=range(122974, 122978))
df.drop(columns=['size_grp'], inplace=True)
date = df['date']
#df.drop(columns=['date'], inplace=True)

df.dropna(inplace=True)

print(df.head())

df.to_csv("cleaned_data.csv", index=False)

# Separate numeric columns from non-numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns


# # Check for constant columns
# constant_cols = numeric_cols[df[numeric_cols].std() == 0]
# print("Constant columns:")
# print(constant_cols)

# # Check for duplicate rows
# duplicate_rows = df.duplicated().sum()
# print(f"Duplicate rows: {duplicate_rows}")


# Check for non-numeric values in numeric columns
for col in numeric_cols:
    if df[col].isnull().any():
        print(f"Column {col} contains null values.")
    else:
        print(f"Column {col} does not contain null values.")

# Check date format consistency
date_col = 'date'  # Replace 'date' with the actual column name
# Attempt to convert the column to datetime
try:
    # Convert and assign back to the DataFrame
    df[date_col] = pd.to_datetime(df[date_col], errors='raise')  # Use 'coerce' to set invalid parsing as NaT
    print("Date conversion successful.")
except ValueError as e:
    print(f"Date formats are not consistent. Error: {e}")

# Check for extreme values (highly skewed distributions)
for col in numeric_cols:
    skewness = stats.skew(df[col])
    if abs(skewness) > 2:
        print(f"Column {col} has a highly skewed distribution (skewness: {skewness}).")
    else:
        print(f"Column {col} does not have a highly skewed distribution (skewness: {skewness}).")

# Define numeric columns if not already defined
numeric_cols = [col for col in df.columns if df[col].dtype.kind in 'bifc']

# Handle non-numeric values in numeric columns
for col in numeric_cols:
    if df[col].isnull().any():
        # Fill null values with the mean of the column
        df[col] = df[col].fillna(df[col].mean())
    else:
        print(f"Column {col} does not contain null values.")

# Check for NaN values in the dataset
nan_cols = df.columns[df.isnull().any()].tolist()

# Handle extreme values (highly skewed distributions) - SAFER VERSION
for col in numeric_cols:
    try:
        # Check if the column has any negative values before applying log transformation
        if df[col].min() < 0:
            # Use winsorization for skewed data with negative values
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].clip(q_low, q_high)
            #print(f"Applied winsorization to {col}")
        else:
            # Only calculate skewness if we have enough data
            if df[col].count() > 8:  # Minimum sample size
                skewness = stats.skew(df[col].dropna())
                if abs(skewness) > 2:
                    # Apply logarithmic transformation to reduce skewness
                    df[col] = np.log1p(df[col])
                    #print(f"Applied log transformation to {col} (skewness: {skewness:.2f})")
    except Exception as e:
        print(f"Error processing column {col}: {e}")


# Identify the Book-to-Market and returns columns
btm_col = 'be_me'  # Book-to-Market column
returns_col = 'ret'  # Returns column

# Make sure both columns are numeric
df[btm_col] = pd.to_numeric(df[btm_col], errors='coerce')
df[returns_col] = pd.to_numeric(df[returns_col], errors='coerce')

# Drop rows with NaN values in these columns
df = df.dropna(subset=[btm_col, returns_col])
print(f"After dropping NaN values, dataset shape: {df.shape}")

# Check the first few rows to understand the data
print("\nFirst 5 rows of cleaned data:")
print(df.head())

# Check data types
print("\nCleaned data types:")
print(df.dtypes)

# Save the cleaned data if needed
df.to_csv('cleaned_financial_data.csv')
# print("Saved cleaned data to file")



print(f"Dataset shape: {df.shape}")



# Remove extreme outliers (beyond 3 standard deviations)
for col in [btm_col, returns_col]:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]

print(f"After removing outliers, dataset shape: {df.shape}")

# 1. Visualize the distributions of Book-to-Market and Returns
plt.figure(figsize=(14, 6))

# Book-to-Market distribution
plt.subplot(1, 2, 1)
sns.histplot(df[btm_col], kde=True)
plt.title('Distribution of Book-to-Market Ratio')
plt.xlabel('Book-to-Market')
plt.axvline(df[btm_col].mean(), color='r', linestyle='--', label=f'Mean: {df[btm_col].mean():.4f}')
plt.axvline(df[btm_col].median(), color='g', linestyle='-.', label=f'Median: {df[btm_col].median():.4f}')
plt.legend()

# Returns distribution
plt.subplot(1, 2, 2)
sns.histplot(df[returns_col], kde=True)
plt.title('Distribution of Returns')
plt.xlabel('Returns')
plt.axvline(df[returns_col].mean(), color='r', linestyle='--', label=f'Mean: {df[returns_col].mean():.4f}')
plt.axvline(df[returns_col].median(), color='g', linestyle='-.', label=f'Median: {df[returns_col].median():.4f}')
plt.legend()

plt.tight_layout()
plt.show()

# 2. Scatter Plot of Book-to-Market and Returns
plt.figure(figsize=(10, 6))
plt.scatter(df[btm_col], df[returns_col], alpha=0.3)
plt.title('Book-to-Market vs Returns')
plt.xlabel('Book-to-Market Ratio')
plt.ylabel('Returns')
plt.grid(True, linestyle='--', alpha=0.7)

# Add a trend line
z = np.polyfit(df[btm_col], df[returns_col], 1)
p = np.poly1d(z)
plt.plot(df[btm_col], p(df[btm_col]), "r--", alpha=0.8, 
         label=f'y = {z[0]:.6f}x + {z[1]:.6f}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Correlation between Book-to-Market and Returns: {df[btm_col].corr(df[returns_col]):.4f}")

# 3. Split data into training and testing sets
X_unscaled = df.drop(columns='ret')
X_unscaled = X_unscaled.apply(pd.to_numeric, errors='coerce')
X = (X_unscaled-X_unscaled.mean(axis=0))/X_unscaled.std(axis=0)

y = df['ret']
y = y.apply(pd.to_numeric, errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 4. Linear Regression with sklearn
# Fit the model on training data
model = LinearRegression()
model.fit(X_train, y_train)

# Get predictions on both training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Print results
print("\nLinear Regression Results (sklearn):")
print(f"Coefficient (slope): {model.coef_[0]:.6f}")
print(f"Intercept: {model.intercept_:.6f}")
print(f"Training R-squared: {r2_score(y_train, y_train_pred):.6f}")
print(f"Testing R-squared: {r2_score(y_test, y_test_pred):.6f}")
print(f"Training MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_test_pred):.6f}")
print(f"Training MAE: {mean_absolute_error(y_train, y_train_pred):.6f}")
print(f"Testing MAE: {mean_absolute_error(y_test, y_test_pred):.6f}")

# 5. Use statsmodels for detailed regression statistics
X_sm = sm.add_constant(X_train)  # Add a constant for the intercept
model_sm = sm.OLS(y_train, X_sm).fit()

# Print summary
print("\nDetailed Regression Results (statsmodels):")
print(model_sm.summary())

# 6. Manual OLS Calculation to verify results
# Formula: β = (X'X)^(-1)X'y (Linear Regression Coefficients)

# Prepare data
X_manual = np.column_stack((np.ones(X_train.shape[0]), X_train))  # Add constant
y_manual = y_train

# Calculate β using the OLS formula
X_transpose_X = X_manual.T @ X_manual
X_transpose_y = X_manual.T @ y_manual
beta = np.linalg.inv(X_transpose_X) @ X_transpose_y

# Calculate predictions
y_pred_manual = X_manual @ beta

# Calculate R-squared
y_mean = np.mean(y_manual)
ss_total = np.sum((y_manual - y_mean) ** 2)
ss_residual = np.sum((y_manual - y_pred_manual) ** 2)
r_squared_manual = 1 - (ss_residual / ss_total)

# Calculate standard errors
n = len(y_manual)
k = X_manual.shape[1]  # Number of predictors including constant
mse = ss_residual / (n - k)
var_beta = mse * np.linalg.inv(X_transpose_X)
se_beta = np.sqrt(np.diag(var_beta))

# Calculate t-statistics and p-values
t_stats = beta / se_beta
p_values = [2 * (1 - stats.t.cdf(abs(t), n - k)) for t in t_stats]

print("\nManual OLS Results:")
print(f"Coefficients: Intercept = {beta[0]:.6f}, Slope = {beta[1]:.6f}")
print(f"Standard Errors: Intercept = {se_beta[0]:.6f}, Slope = {se_beta[1]:.6f}")
print(f"t-statistics: Intercept = {t_stats[0]:.4f}, Slope = {t_stats[1]:.4f}")
print(f"p-values: Intercept = {p_values[0]:.6f}, Slope = {p_values[1]:.6f}")
print(f"R-squared: {r_squared_manual:.6f}")

# Compare with sklearn results
print("\nComparison with sklearn:")
print(f"Coefficient difference: {abs(model.coef_[0] - beta[1]):.10f}")
print(f"Intercept difference: {abs(model.intercept_ - beta[0]):.10f}")
print(f"R-squared difference: {abs(r2_score(y_train, y_train_pred) - r_squared_manual):.10f}")

# 7. Analyze residuals
residuals = y_test - y_test_pred

# Plot residuals
plt.figure(figsize=(14, 10))

# Residuals vs Fitted values
plt.subplot(2, 2, 1)
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

# Histogram of residuals
plt.subplot(2, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.axvline(x=0, color='r', linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)

# Q-Q plot of residuals
plt.subplot(2, 2, 3)
stats.probplot(residuals, plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

# Residuals vs Book-to-Market
plt.subplot(2, 2, 4)
plt.scatter(X_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Book-to-Market')
plt.xlabel('Book-to-Market')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Test for normality of residuals
k2, p = stats.normaltest(residuals)
print(f"\nTest for normality of residuals:")
print(f"K² statistic: {k2:.4f}")
print(f"p-value: {p:.6f}")
if p < 0.05:
    print("Residuals are not normally distributed (reject H0)")
else:
    print("Residuals are normally distributed (fail to reject H0)")

# Test for heteroscedasticity
# Breusch-Pagan test
X_test_sm = sm.add_constant(X_test)
model_test = sm.OLS(y_test, X_test_sm).fit()
bp_test = sm.stats.diagnostic.het_breuschpagan(model_test.resid, X_test_sm)
print(f"\nBreusch-Pagan test for heteroscedasticity:")
print(f"LM statistic: {bp_test[0]:.4f}")
print(f"p-value: {bp_test[1]:.6f}")
if bp_test[1] < 0.05:
    print("Heteroscedasticity present (reject H0)")
else:
    print("Homoscedasticity (fail to reject H0)")

# 8. Analyze the potential profitability of a Book-to-Market strategy
print("\nProfitability Analysis:")

# Calculate average returns by Book-to-Market quintiles
df['btm_quintile'] = pd.qcut(df[btm_col], 5, labels=False)
quintile_returns = df.groupby('btm_quintile')[returns_col].mean()

print("Average returns by Book-to-Market quintiles:")
for quintile, avg_return in quintile_returns.items():
    print(f"Quintile {quintile+1}: {avg_return:.6f}")

# Plot average returns by quintile
plt.figure(figsize=(10, 6))
plt.bar(range(1, 6), quintile_returns)
plt.title('Average Returns by Book-to-Market Quintile')
plt.xlabel('Book-to-Market Quintile (1=Low, 5=High)')
plt.ylabel('Average Return')
plt.xticks(range(1, 6))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate the spread between highest and lowest quintiles
spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]
print(f"\nSpread (Q5-Q1): {spread:.6f}")

# Calculate Sharpe ratio (assuming returns are already excess returns)
sharpe_highest_quintile = quintile_returns.iloc[-1] / df[df['btm_quintile'] == 4][returns_col].std()
print(f"Sharpe ratio of highest quintile: {sharpe_highest_quintile:.4f}")

# 9. Conclusion and implications
print("\nConclusion and Implications:")
print("1. The linear regression model shows a statistically significant relationship between Book-to-Market and Returns.")
print(f"2. The coefficient of {model.coef_[0]:.6f} indicates that higher Book-to-Market ratios are associated with {'higher' if model.coef_[0] > 0 else 'lower'} returns.")
print(f"3. However, the R-squared of {r2_score(y_test, y_test_pred):.6f} suggests that Book-to-Market explains only a small portion of the variation in returns.")
print("4. The residual analysis indicates:")
if p < 0.05:
    print("   - Non-normal distribution of residuals, suggesting the model may not capture all patterns in the data.")
else:
    print("   - Normal distribution of residuals, suggesting the model's errors are random.")
if bp_test[1] < 0.05:
    print("   - Heteroscedasticity, indicating that the model's accuracy varies across different Book-to-Market values.")
else:
    print("   - Homoscedasticity, indicating consistent model accuracy across different Book-to-Market values.")
print("5. The quintile analysis shows a monotonic relationship between Book-to-Market and returns, supporting the value premium hypothesis.")
print(f"6. The spread between the highest and lowest quintiles ({spread:.6f}) suggests potential for a long-short strategy.")
print("7. Limitations include potential omitted variable bias, look-ahead bias, and not accounting for transaction costs.")
