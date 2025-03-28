import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BlackScholesSimulator:
    def __init__(self, seed=42):
        self.seed = seed
        self.scaler = StandardScaler()
        np.random.seed(self.seed)
    
    def black_scholes_call(self, S, X, T, r, sigma):
        """Compute the Black-Scholes call option price."""
        d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        call_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
        return call_price

    def generate_option_dataset(self):
        """Generate a dataset of call option prices based on a parameter grid."""
        # Define ranges for parameters
        S_range = np.arange(40, 70, 1)  # Stock prices from $40 to $69
        X_range = np.arange(25, 91, 1)  # Strike prices from $25 to $90
        T_range = np.arange(0.2, 2.1, 0.1)  # Time to maturity from 0.2 to 2.0 years
        r_range = np.arange(0.00, 0.06, 0.01)  # Risk-free rate from 0% to 5%
        sigma_range = np.arange(0.23, 0.76, 0.1)  # Volatility from 23% to 75%
        
        # Create a Cartesian product of all parameter combinations
        param_grid = np.array(np.meshgrid(S_range, X_range, T_range, r_range, sigma_range)).T.reshape(-1, 5)
        
        # Extract parameter columns
        S, X, T, r, sigma = param_grid.T
        
        # Compute Black-Scholes prices
        prices = self.black_scholes_call(S, X, T, r, sigma)
        
        # Add idiosyncratic noise
        
        noise = np.random.normal(0, 2, size=prices.shape)
        prices = np.maximum(prices + noise, 0)  # Ensures no negative prices

        
        # Create DataFrame
        df = pd.DataFrame({
            "Stock Price": S,
            "Strike Price": X,
            "Time to Expiration": T,
            "Risk-Free Rate": r,
            "Volatility": sigma,
            "Option Price": prices
        })
        
        # Split into training and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=self.seed)
        
        # Scale the input features
        feature_columns = ["Stock Price", "Strike Price", "Time to Expiration", "Risk-Free Rate", "Volatility"]
        train_df[feature_columns] = self.scaler.fit_transform(train_df[feature_columns])
        test_df[feature_columns] = self.scaler.transform(test_df[feature_columns])
        
        self.full_dataset = df  # Store full dataset
        return train_df, test_df
    
    def print_sample_data(self, train_df, test_df):
        """Prints sample data from training and test sets."""
        print("Training Data Sample:")
        print(train_df.head())
        print("\nTest Data Sample:")
        print(test_df.head())
    
    def export_dataset(self, filename="option_dataset.csv"):
        """Exports the full dataset to a CSV file."""
        if hasattr(self, 'full_dataset'):
            self.full_dataset.to_csv(filename, index=False, sep='\t')
            print(f"Dataset exported successfully to {filename}")
        else:
            print("Dataset has not been generated yet. Run generate_option_dataset() first.")

# Instantiate and use the class
simulator = BlackScholesSimulator()
train_df, test_df = simulator.generate_option_dataset()
simulator.print_sample_data(train_df, test_df)
simulator.export_dataset()
