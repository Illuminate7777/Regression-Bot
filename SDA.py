import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
import pandas_datareader.data as web
import tkinter as tk
from tkinter import messagebox, filedialog  # Imported filedialog
import requests
from io import BytesIO
import os  # Imported os

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stock Beta Analysis with FF5, Momentum, VIX, EPU, and Liquidity")
        self.geometry("800x800")  # Adjusted size for new layout

        # Variables to store user input
        self.stock_code = tk.StringVar()
        self.timeframe = tk.StringVar()
        self.start_date = tk.StringVar()
        self.end_date = tk.StringVar()
        self.frequency = tk.StringVar(value='monthly')  # Default to monthly

        # Variables to store user input for factors
        self.factors = {
            'Mkt-RF': tk.IntVar(value=1),  # Market factor is selected by default
            'SMB': tk.IntVar(),
            'HML': tk.IntVar(),
            'RMW': tk.IntVar(),
            'CMA': tk.IntVar(),
            'Mom': tk.IntVar(),
            'VIX': tk.IntVar(),
            'Agg_Liq': tk.IntVar(),
            'Innov_Liq': tk.IntVar(),
            'Traded_Liq': tk.IntVar(),
            'Global_Current': tk.IntVar(),
            'Global_PPP': tk.IntVar(),
            'US_TCI': tk.IntVar(),
            'US_NBPUI': tk.IntVar(),
        }

        # Create initial widgets
        self.create_initial_widgets()

    def create_initial_widgets(self):
        # Clear any existing widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Frame for factor selection checkboxes
        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(pady=10)

        tk.Label(checkbox_frame, text="Select Factors to Include in the Regression:").grid(row=0, column=0, columnspan=3, pady=5)

        # Arrange checkboxes in a grid
        factor_list = list(self.factors.keys())
        for idx, factor in enumerate(factor_list):
            row = idx // 3 + 1  # Start from row 1
            col = idx % 3
            tk.Checkbutton(checkbox_frame, text=factor, variable=self.factors[factor]).grid(row=row, column=col, sticky='w')

        # Stock ticker input
        tk.Label(self, text="Enter the stock ticker:").pack(pady=5)
        tk.Entry(self, textvariable=self.stock_code).pack()

        # Timeframe input
        tk.Label(self, text="Enter the timeframe (e.g., '1y', '5y', '10y', '52w', or year starting from 1900):").pack(pady=5)
        tk.Entry(self, textvariable=self.timeframe).pack()

        # Date range input
        tk.Label(self, text="Or set a date range:").pack(pady=5)
        date_frame = tk.Frame(self)
        date_frame.pack(pady=5)
        tk.Label(date_frame, text="Start Date (YYYY-MM):").grid(row=0, column=0)
        tk.Entry(date_frame, textvariable=self.start_date).grid(row=0, column=1)
        tk.Label(date_frame, text="End Date (YYYY-MM):").grid(row=1, column=0)
        tk.Entry(date_frame, textvariable=self.end_date).grid(row=1, column=1)

        # Frequency selection
        tk.Label(self, text="Select Frequency for Analysis:").pack(pady=5)
        freq_frame = tk.Frame(self)
        freq_frame.pack()
        tk.Radiobutton(freq_frame, text="Monthly", variable=self.frequency, value='monthly').pack(side='left', padx=5)
        tk.Radiobutton(freq_frame, text="Daily", variable=self.frequency, value='daily').pack(side='left', padx=5)

        # Get beta values button
        tk.Button(self, text="Get Beta Values", command=self.get_beta_values).pack(pady=20)

    def get_beta_values(self):
        # Get inputs
        stock_code = self.stock_code.get().upper()
        timeframe = self.timeframe.get()
        start_date = self.start_date.get()
        end_date = self.end_date.get()
        frequency = self.frequency.get()

        if not stock_code:
            messagebox.showerror("Input Error", "Please enter the stock ticker.")
            return

        # Get selected factors
        selected_factors = {factor: var.get() for factor, var in self.factors.items()}
        if not any(selected_factors.values()):
            messagebox.showerror("Input Error", "Please select at least one factor.")
            return

        # Call the analysis function to get beta values
        results = calculate_beta_with_factors(
            stock_code=stock_code,
            selected_factors=selected_factors,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency
        )

        # Check if betas were successfully calculated
        if results[0] is not None:
            # Unpack the results
            betas, t_stats, beta_market_only, yf_beta, current_price, methods_explanation, model = results
            # Store selected factors for prediction step
            self.selected_factors_for_prediction = selected_factors
            # Store the model for prediction
            self.model = model
            # Store the betas and t-stats for saving
            self.betas = betas
            self.t_stats = t_stats
            # Display the beta values and current price
            self.display_beta_values(stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, methods_explanation)
        else:
            messagebox.showerror("Error", "Failed to calculate betas.")

    def display_beta_values(self, stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, methods_explanation):
        # Clear previous widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Display beta values
        tk.Label(self, text=f"Stock: {stock_code}").pack(pady=5)
        tk.Label(self, text=f"Current Price: {current_price}").pack(pady=5)
        tk.Label(self, text=f"Beta according to yfinance: {yf_beta}").pack(pady=5)
        tk.Label(self, text="Calculated Betas with Selected Factors:").pack(pady=5)
        tk.Label(self, text=f"Market Beta: {beta_market_only:.4f}").pack()

        # Display alpha (intercept)
        if 'const' in betas:
            alpha_value = betas.get('const', 0)
            t_stat = t_stats.get('const', 0)
            tk.Label(self, text=f"Alpha (Intercept): {alpha_value:.4f} (t-stat: {t_stat:.4f})").pack()

        # Display betas and t-statistics for selected factors
        for factor in betas:
            if factor != 'const' and factor != 'Mkt-RF':
                display_name = factor.replace('_Return', '')
                display_name = display_name.replace('_', ' ')  # Replace underscores with spaces for display
                beta_value = betas.get(factor, 0)
                t_stat = t_stats.get(factor, 0)
                tk.Label(self, text=f"{display_name} Beta: {beta_value:.4f} (t-stat: {t_stat:.4f})").pack()

        # Display methods explanation (Updated)
        tk.Label(self, text="Seung Ho Jeon", font=('Helvetica', 12, 'bold')).pack(pady=10)
        tk.Label(self, text="Analysis Bot", font=('Helvetica', 12, 'bold')).pack(pady=5)

        # Button to save results to CSV
        tk.Button(self, text="Save Results to CSV", command=lambda: self.save_results_to_csv(
            stock_code, betas, t_stats)
        ).pack(pady=10)

        # Proceed to predict future prices
        tk.Button(self, text="Proceed to Predict Future Price", command=lambda: self.predict_future_price(
            stock_code, yf_beta, current_price, beta_market_only, betas, t_stats)
        ).pack(pady=20)
        # Option to return to initial analysis
        tk.Button(self, text="Return to Analysis", command=self.create_initial_widgets).pack(pady=10)

    def save_results_to_csv(self, stock_code, betas, t_stats):
        # Prepare data for CSV
        results_df = pd.DataFrame({
            'Factor': list(betas.keys()),
            'Coefficient': list(betas.values()),
            't-stat': [t_stats.get(k, np.nan) for k in betas.keys()]
        })
        # Prompt the user for a filename and location
        file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')], initialfile=f"{stock_code}_betas.csv")
        if file_path:
            try:
                results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results saved to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
        else:
            # User cancelled save dialog
            pass

    def predict_future_price(self, stock_code, yf_beta, current_price, beta_market_only, betas, t_stats):
        # Clear previous widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Ask for future period
        tk.Label(self, text="Enter a future date (YYYY-MM-DD) or period (e.g., '30d', '6m', '1y'):").pack(pady=5)
        future_period_var = tk.StringVar(value='6m')  # Default to 6 months
        tk.Entry(self, textvariable=future_period_var).pack()
        tk.Button(self, text="Predict", command=lambda: self.perform_prediction(
            stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, future_period_var.get())
        ).pack(pady=10)

    def perform_prediction(self, stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, future_period_input):
        # Use the selected factors stored earlier
        selected_factors = self.selected_factors_for_prediction
        # Calculate predicted future price with confidence intervals
        prediction_results = calculate_predicted_price(
            stock_code, betas, future_period_input, self.frequency.get(), selected_factors, self.model
        )

        # Display predicted future price
        self.display_prediction_results(stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, prediction_results)

    def display_prediction_results(self, stock_code, yf_beta, current_price, beta_market_only, betas, t_stats, prediction_results):
        # Clear previous widgets
        for widget in self.winfo_children():
            widget.destroy()

        # Display all results
        tk.Label(self, text=f"Stock: {stock_code}").pack(pady=5)
        tk.Label(self, text=f"Current Price: {current_price}").pack(pady=5)
        tk.Label(self, text=f"Beta according to yfinance: {yf_beta}").pack(pady=5)

        if prediction_results is not None:
            low, med, top = prediction_results
            tk.Label(self, text=f"Predicted Future Price (95% Confidence Interval):").pack(pady=5)
            tk.Label(self, text=f"Low: {low:.2f}").pack()
            tk.Label(self, text=f"Med: {med:.2f}").pack()
            tk.Label(self, text=f"High: {top:.2f}").pack()
        else:
            tk.Label(self, text="Unable to predict future price with the given inputs.").pack(pady=5)

        # Option to return to initial analysis
        tk.Button(self, text="Return to Analysis", command=self.create_initial_widgets).pack(pady=10)

# The rest of the code remains the same...

def calculate_beta_with_factors(stock_code, selected_factors, timeframe=None, start_date=None, end_date=None, frequency='monthly'):
    """
    Calculate betas using the FF5 model with momentum, VIX returns, EPU, and Liquidity.
    """
    methods_explanation = ""

    # Fetch stock data
    try:
        # Set the end date to today if not provided
        if end_date:
            try:
                end_date = datetime.strptime(end_date, '%Y-%m')
                end_date = end_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)  # End of month
            except ValueError:
                messagebox.showerror("Input Error", "Invalid end date format. Please use YYYY-MM.")
                return (None, None, None, None, None, None, None)
        else:
            end_date = datetime.today()

        # Determine start date
        if start_date:
            try:
                start_date = datetime.strptime(start_date, '%Y-%m')
                start_date = start_date.strftime('%Y-%m-%d')
            except ValueError:
                messagebox.showerror("Input Error", "Invalid start date format. Please use YYYY-MM.")
                return (None, None, None, None, None, None, None)
        elif timeframe:
            if timeframe == '1y':
                start_date = (end_date - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            elif timeframe == '5y':
                start_date = (end_date - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
            elif timeframe == '10y':
                start_date = (end_date - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
            elif timeframe == '52w':
                start_date = (end_date - pd.DateOffset(weeks=52)).strftime('%Y-%m-%d')
            elif timeframe.isdigit() and int(timeframe) >= 1900:
                start_date = f"{timeframe}-01-01"
            else:
                print(f"Invalid timeframe input. Using default full history for {stock_code}.")
                start_date = '1900-01-01'  # Default to full history if invalid input
        else:
            start_date = '1900-01-01'  # Default to full history

        # Download stock data
        stock_data = yf.download(stock_code, start=start_date, end=end_date.strftime('%Y-%m-%d'))
        if stock_data.empty:
            print(f"No historical data available for {stock_code}.")
            return (None, None, None, None, None, None, None)

        # Adjust start_date based on actual available data
        actual_start_date = stock_data.index[0]
        print(f"Actual available data for {stock_code} starts on {actual_start_date.strftime('%Y-%m-%d')}")
        start_date = actual_start_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error fetching data for {stock_code}: {e}")
        return (None, None, None, None, None, None, None)

    # Fetch current price and beta from yfinance
    stock = yf.Ticker(stock_code)
    try:
        info = stock.info
        yf_beta = info.get('beta', 'N/A')
    except Exception as e:
        print(f"Error fetching info for {stock_code}: {e}")
        yf_beta = 'N/A'

    # Get the current price
    history = stock.history(period='1d')
    if history.empty:
        print(f"No data available for {stock_code} today.")
        current_price = 'N/A'
    else:
        current_price = history['Close'].iloc[0]

    # Resample stock data based on frequency
    if frequency == 'monthly':
        # Aggregate daily stock data to monthly frequency
        stock_data = stock_data.resample('M').last()
        stock_data['Return_stock'] = stock_data['Close'].pct_change()
        stock_data.index = stock_data.index.to_period('M').to_timestamp()
    elif frequency == 'daily':
        # For daily data, ensure the data is at daily frequency
        stock_data['Return_stock'] = stock_data['Close'].pct_change()
    else:
        print(f"Invalid frequency input. Using default monthly frequency for {stock_code}.")
        stock_data = stock_data.resample('M').last()
        stock_data['Return_stock'] = stock_data['Close'].pct_change()
        stock_data.index = stock_data.index.to_period('M').to_timestamp()

    # Fetch Fama-French factors and Momentum
    try:
        # Fetch Fama-French 5 factors
        ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start_date, end=end_date)
        ff_factors = ff_data[0]
        ff_factors.index = ff_factors.index.to_timestamp()
        ff_factors = ff_factors / 100  # Convert percentages to decimals

        # Fetch Momentum factor
        mom_data = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=start_date, end=end_date)
        mom_factor = mom_data[0]
        mom_factor.index = mom_factor.index.to_timestamp()
        mom_factor = mom_factor / 100
        mom_factor.columns = ['Mom']

        # Combine factors
        ff_factors = ff_factors.join(mom_factor, how='inner')

        if frequency == 'daily':
            # Interpolate monthly factors to daily frequency
            daily_index = pd.date_range(start=ff_factors.index.min(), end=ff_factors.index.max(), freq='D')
            ff_factors = ff_factors.reindex(ff_factors.index.union(daily_index))
            ff_factors = ff_factors.interpolate(method='linear')
            ff_factors = ff_factors.reindex(daily_index)
    except Exception as e:
        print(f"Error fetching Fama-French factors: {e}")
        return (None, None, None, None, None, None, None)

    # Merge stock returns with factors
    data = pd.merge(stock_data[['Return_stock']], ff_factors, left_index=True, right_index=True, how='inner')

    # Handle VIX factor
    factor_names = [factor for factor in selected_factors if selected_factors[factor] == 1]

    if 'VIX' in factor_names:
        # Fetch VIX data
        vix_data = yf.download('^VIX', start=start_date, end=end_date.strftime('%Y-%m-%d'))
        if vix_data.empty:
            print("No VIX data available.")
            return (None, None, None, None, None, None, None)
        # Resample VIX data to match frequency
        if frequency == 'monthly':
            vix_data = vix_data.resample('M').last()
            vix_data.index = vix_data.index.to_period('M').to_timestamp()
        elif frequency == 'daily':
            vix_data = vix_data.asfreq('D', method='pad')
        else:
            print("Invalid frequency for VIX data.")
            return (None, None, None, None, None, None, None)
        # Use VIX levels and calculate returns
        vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX'})
        vix_data['VIX_Return'] = vix_data['VIX'].pct_change()
        vix_data = vix_data.dropna(subset=['VIX_Return'])
        # Merge VIX returns into main data frame
        data = pd.merge(data, vix_data[['VIX_Return']], left_index=True, right_index=True, how='inner')
        # Replace 'VIX' with 'VIX_Return' in factor names
        factor_names = ['VIX_Return' if factor == 'VIX' else factor for factor in factor_names]

    # Handle Liquidity factors
    liquidity_factors = []
    for liq_factor in ['Agg_Liq', 'Innov_Liq', 'Traded_Liq']:
        if liq_factor in factor_names:
            liquidity_factors.append(liq_factor)

    if liquidity_factors:
        try:
            # Read liquidity data from local CSV file
            liq_data = pd.read_csv('liq_data.csv')
            # Replace placeholder values like -99 with NaN
            liq_data.replace(-99, np.nan, inplace=True)
            # Remove rows where liquidity values are greater than 1
            liq_data = liq_data[(liq_data['Agg_Liq'] <= 1) & (liq_data['Innov_Liq'] <= 1) & (liq_data['Traded_Liq'] <= 1)]
            # Convert 'Date' column to datetime
            liq_data['Date'] = pd.to_datetime(liq_data['Date'], format='%Y%m')
            liq_data.set_index('Date', inplace=True)
            # Convert percentages to decimals
            liq_data[['Agg_Liq', 'Innov_Liq', 'Traded_Liq']] = liq_data[['Agg_Liq', 'Innov_Liq', 'Traded_Liq']] / 100

            # Handle frequency conversion if necessary
            if frequency == 'daily':
                # Interpolate monthly Liquidity data to daily frequency
                daily_index = pd.date_range(start=liq_data.index.min(), end=liq_data.index.max(), freq='D')
                liq_data = liq_data.reindex(daily_index)
                liq_data = liq_data.fillna(method='ffill')
            else:
                # Monthly data, no change needed
                pass

            # Merge Liquidity data into the main data frame
            data = pd.merge(data, liq_data[liquidity_factors], left_index=True, right_index=True, how='inner')
        except Exception as e:
            print(f"Error fetching Liquidity data: {e}")
            return (None, None, None, None, None, None, None)

    # Handle Global and US EPU factors
    epu_factors = {
        'Global_Current': {
            'url': 'https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx',
            'columns': ['GEPU_current'],
        },
        'Global_PPP': {
            'url': 'https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx',
            'columns': ['GEPU_ppp'],
        },
        'US_TCI': {
            'url': 'https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx',
            'columns': ['Three_Component_Index'],
        },
        'US_NBPUI': {
            'url': 'https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx',
            'columns': ['News_Based_Policy_Uncert_Index'],
        },
    }

    for epu_type in ['Global_Current', 'Global_PPP', 'US_TCI', 'US_NBPUI']:
        if epu_type in factor_names:
            epu_info = epu_factors[epu_type]
            epu_url = epu_info['url']
            columns_to_use = epu_info['columns']

            # Fetch EPU data
            epu_data = fetch_epu_data(epu_url, epu_type, columns_to_use)
            if epu_data is None:
                continue

            # Handle frequency
            if frequency == 'daily':
                # Interpolate monthly EPU data to daily frequency
                daily_index = pd.date_range(start=epu_data.index.min(), end=epu_data.index.max(), freq='D')
                epu_data = epu_data.reindex(daily_index)
                epu_data = epu_data.fillna(method='ffill')  # Forward-fill missing values
            else:
                # Monthly data, no change needed
                pass

            # Calculate EPU returns
            epu_column_name = epu_type + '_Return'
            epu_data[epu_column_name] = epu_data.iloc[:, 0].pct_change()
            epu_data = epu_data.dropna(subset=[epu_column_name])

            # Merge EPU returns into main data frame
            data = pd.merge(data, epu_data[[epu_column_name]], left_index=True, right_index=True, how='inner')

            # Replace EPU type in factor names
            factor_names = [epu_column_name if factor == epu_type else factor for factor in factor_names]

    data.dropna(inplace=True)

    # Check if there are enough data points
    if len(data) < 10:
        print(f"Not enough data points after merging and cleaning. Only {len(data)} observations available.")
        messagebox.showwarning("Data Warning", f"Only {len(data)} observations available. Results may not be statistically significant.")

    # Calculate excess returns (over risk-free rate)
    data['Excess_Return_stock'] = data['Return_stock'] - data['RF']

    if not factor_names:
        print("No factors selected for the regression.")
        return (None, None, None, None, None, None, None)

    # Perform regression with selected factors
    Y = data['Excess_Return_stock']
    X = pd.DataFrame()
    if selected_factors.get('Mkt-RF', 0) == 1:
        X['Mkt-RF'] = data['Mkt-RF']  # Include market factor if selected
    else:
        X['Mkt-RF'] = data['Mkt-RF']  # Always include market factor
    selected_factor_columns = [factor for factor in factor_names if factor != 'Mkt-RF']
    if selected_factor_columns:
        X = pd.concat([X, data[selected_factor_columns]], axis=1)
    X = sm.add_constant(X)
    try:
        model = sm.OLS(Y, X).fit()
    except Exception as e:
        print(f"Error in regression: {e}")
        return (None, None, None, None, None, None, None)

    # Get beta coefficients and t-statistics
    betas = model.params.to_dict()
    t_stats = model.tvalues.to_dict()

    # Perform regression with only the market factor (for reference)
    X_market_only = data[['Mkt-RF']]
    X_market_only = sm.add_constant(X_market_only)
    try:
        model_market_only = sm.OLS(Y, X_market_only).fit()
        beta_market_only = model_market_only.params['Mkt-RF']
    except Exception as e:
        print(f"Error in market-only regression: {e}")
        beta_market_only = None

    # Replace methods explanation with specified text
    methods_explanation = "Seung Ho Jeon\nAnalysis Bot"

    return betas, t_stats, beta_market_only, yf_beta, current_price, methods_explanation, model

def fetch_epu_data(epu_url, epu_type, columns_to_use):
    """
    Fetch EPU data from the specified URL and handle parsing errors gracefully.
    """
    try:
        response = requests.get(epu_url)
        if response.status_code != 200:
            print(f"Failed to download {epu_type} data.")
            return None

        # Load the Excel data
        epu_data = pd.read_excel(BytesIO(response.content), engine='openpyxl')

        # Remove the last row if it contains non-numeric data in 'Year' column
        if not pd.api.types.is_numeric_dtype(epu_data['Year'].iloc[-1]):
            epu_data = epu_data.iloc[:-1]  # Remove the last row

        # Combine 'Year' and 'Month' into a datetime index
        epu_data['Date'] = pd.to_datetime(epu_data[['Year', 'Month']].assign(DAY=1), errors='coerce')

        # Remove rows with any parsing errors (invalid dates)
        epu_data.dropna(subset=['Date'], inplace=True)

        epu_data = epu_data.set_index('Date')

        # Select the specific columns needed
        epu_data = epu_data[columns_to_use]

        return epu_data

    except Exception as e:
        print(f"Error fetching {epu_type} data: {e}")
        return None

def calculate_predicted_price(stock_code, betas, future_period_input, frequency, selected_factors, model):
    """
    Calculate predicted future price based on estimated factor returns with confidence intervals.
    """
    # Parse future end date
    future_end_date = parse_future_period(future_period_input)
    if future_end_date is None:
        print("Invalid future date or period entered.")
        return None

    # Fetch historical factor data
    try:
        # Fetch only the selected factors
        factor_names = [factor for factor in selected_factors if selected_factors[factor] == 1]

        # Fetch Fama-French 5 factors if selected
        factors_to_fetch = []
        ff_factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        for factor in ff_factor_names:
            if factor in factor_names:
                factors_to_fetch.append(factor)

        if factors_to_fetch:
            ff_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench')
            ff_factors = ff_data[0]
            ff_factors.index = ff_factors.index.to_timestamp()
            ff_factors = ff_factors / 100
            ff_factors = ff_factors[factors_to_fetch + ['RF']]  # Include RF for excess return calculation
        else:
            ff_factors = pd.DataFrame()

        # Fetch Momentum factor if selected
        if 'Mom' in factor_names:
            mom_data = web.DataReader('F-F_Momentum_Factor', 'famafrench')
            mom_factor = mom_data[0]
            mom_factor.index = mom_factor.index.to_timestamp()
            mom_factor = mom_factor / 100
            mom_factor.columns = ['Mom']
            ff_factors = ff_factors.join(mom_factor, how='inner')
        else:
            # Ensure RF is present if no FF factors are selected
            if ff_factors.empty and 'RF' not in ff_factors.columns:
                ff_factors['RF'] = 0

        # Handle VIX in prediction
        if 'VIX' in factor_names:
            # Fetch VIX data
            vix_data = yf.download('^VIX', period='max')
            vix_data = vix_data[['Close']].rename(columns={'Close': 'VIX'})
            vix_data['VIX_Return'] = vix_data['VIX'].pct_change()
            vix_data = vix_data.dropna(subset=['VIX_Return'])
            vix_data.index = pd.to_datetime(vix_data.index)
            # Resample VIX data to match frequency
            if frequency == 'monthly':
                vix_data = vix_data.resample('M').last()
                vix_data.index = vix_data.index.to_period('M').to_timestamp()
            elif frequency == 'daily':
                vix_data = vix_data.asfreq('D', method='pad')
            else:
                print("Invalid frequency for VIX data.")
                return None
            # Merge VIX returns into factors
            ff_factors = ff_factors.join(vix_data['VIX_Return'], how='inner')

        # Handle Liquidity in prediction
        liquidity_factors = []
        for liq_factor in ['Agg_Liq', 'Innov_Liq', 'Traded_Liq']:
            if liq_factor in factor_names:
                liquidity_factors.append(liq_factor)
        if liquidity_factors:
            try:
                # Read liquidity data from local CSV file
                liq_data = pd.read_csv('liq_data.csv')
                # Replace placeholder values like -99 with NaN
                liq_data.replace(-99, np.nan, inplace=True)
                # Remove rows where liquidity values are greater than 1
                liq_data = liq_data[(liq_data['Agg_Liq'] <= 1) & (liq_data['Innov_Liq'] <= 1) & (liq_data['Traded_Liq'] <= 1)]
                # Convert 'Date' column to datetime
                liq_data['Date'] = pd.to_datetime(liq_data['Date'], format='%Y%m')
                liq_data.set_index('Date', inplace=True)
                # Convert percentages to decimals
                liq_data[liquidity_factors] = liq_data[liquidity_factors] / 100

                # Handle frequency
                if frequency == 'daily':
                    # Interpolate monthly Liquidity data to daily frequency
                    daily_index = pd.date_range(start=liq_data.index.min(), end=liq_data.index.max(), freq='D')
                    liq_data = liq_data.reindex(daily_index)
                    liq_data = liq_data.fillna(method='ffill')
                else:
                    # Monthly data, no change needed
                    pass

                # Merge Liquidity data into factors
                ff_factors = ff_factors.join(liq_data[liquidity_factors], how='inner')
            except Exception as e:
                print(f"Error fetching Liquidity data: {e}")
                # Proceed without liquidity factors if data is missing
                pass

        # Handle EPU factors in prediction
        epu_factors = {
            'Global_Current': {
                'url': 'https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx',
                'columns': ['GEPU_current'],
            },
            'Global_PPP': {
                'url': 'https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx',
                'columns': ['GEPU_ppp'],
            },
            'US_TCI': {
                'url': 'https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx',
                'columns': ['Three_Component_Index'],
            },
            'US_NBPUI': {
                'url': 'https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx',
                'columns': ['News_Based_Policy_Uncert_Index'],
            },
        }

        for epu_type in ['Global_Current', 'Global_PPP', 'US_TCI', 'US_NBPUI']:
            if epu_type in factor_names:
                epu_info = epu_factors[epu_type]
                epu_url = epu_info['url']
                columns_to_use = epu_info['columns']

                # Fetch EPU data
                epu_data = fetch_epu_data(epu_url, epu_type, columns_to_use)
                if epu_data is None:
                    continue

                # Handle frequency
                if frequency == 'daily':
                    # Interpolate monthly EPU data to daily frequency
                    daily_index = pd.date_range(start=epu_data.index.min(), end=epu_data.index.max(), freq='D')
                    epu_data = epu_data.reindex(daily_index)
                    epu_data = epu_data.fillna(method='ffill')  # Forward-fill missing values
                else:
                    # Monthly data, no change needed
                    pass

                # Calculate EPU returns
                epu_column_name = epu_type + '_Return'
                epu_data[epu_column_name] = epu_data.iloc[:, 0].pct_change()
                epu_data = epu_data.dropna(subset=[epu_column_name])

                # Merge EPU returns into factors
                ff_factors = ff_factors.join(epu_data[[epu_column_name]], how='inner')

        if frequency == 'daily':
            # Interpolate factors to daily frequency
            daily_index = pd.date_range(start=ff_factors.index.min(), end=ff_factors.index.max(), freq='D')
            ff_factors = ff_factors.reindex(ff_factors.index.union(daily_index))
            ff_factors = ff_factors.interpolate(method='linear')
            ff_factors = ff_factors.reindex(daily_index)
    except Exception as e:
        print(f"Error fetching factors for prediction: {e}")
        return None

    # Estimate future factor returns based on historical averages
    estimated_factor_returns = estimate_future_factor_returns(ff_factors, future_end_date, frequency)

    if estimated_factor_returns is None:
        print("Unable to estimate future factor returns.")
        return None

    # Remove 'RF' from estimated_factor_returns if present
    if 'RF' in estimated_factor_returns.index:
        estimated_factor_returns = estimated_factor_returns.drop('RF')

    # Ensure that the constant ('const') is added to the new observation DataFrame
    # Create a DataFrame for new observations
    new_X = pd.DataFrame([estimated_factor_returns])

    # Ensure that 'const' is present in new_X
    if 'const' not in new_X.columns:
        new_X = sm.add_constant(new_X, has_constant='add')

    # Ensure that the columns in new_X match the model's exogenous variables
    missing_cols = set(model.params.index) - set(new_X.columns)
    for col in missing_cols:
        new_X[col] = 0

    # Reorder columns to match the model
    new_X = new_X[model.params.index]

    # Perform the prediction using the model
    try:
        predictions = model.get_prediction(new_X)
        prediction_summary = predictions.summary_frame(alpha=0.05)
        expected_return = prediction_summary['mean'][0]
        lower_bound = prediction_summary['obs_ci_lower'][0]
        upper_bound = prediction_summary['obs_ci_upper'][0]
    except Exception as e:
        print(f"Error calculating prediction intervals: {e}")
        return None

    # Get the current stock price
    stock = yf.Ticker(stock_code)
    stock_data = stock.history(period='1d')
    if stock_data.empty:
        print(f"No data available for {stock_code} today.")
        current_price = None
    else:
        current_price = stock_data['Close'].iloc[0]

    if current_price is None:
        print("Cannot retrieve current price for prediction.")
        return None

    # Calculate predicted future price and confidence intervals
    predicted_future_price = current_price * (1 + expected_return)
    low_price = current_price * (1 + lower_bound)
    high_price = current_price * (1 + upper_bound)

    return low_price, predicted_future_price, high_price

def parse_future_period(future_period_input):
    """
    Parse the future period input and return a datetime object.
    """
    try:
        # Try parsing as a date
        future_end_date = datetime.strptime(future_period_input, '%Y-%m-%d')
        return future_end_date
    except ValueError:
        # Parse as a period
        units = {'d': 'days', 'w': 'weeks', 'm': 'months', 'y': 'years'}
        unit = future_period_input[-1]
        if unit not in units:
            return None
        try:
            value = int(future_period_input[:-1])
            if units[unit] == 'days':
                delta = timedelta(days=value)
                future_end_date = datetime.today() + delta
            elif units[unit] == 'weeks':
                delta = timedelta(weeks=value)
                future_end_date = datetime.today() + delta
            elif units[unit] == 'months':
                future_end_date = pd.Timestamp(datetime.today()) + pd.DateOffset(months=value)
            elif units[unit] == 'years':
                future_end_date = pd.Timestamp(datetime.today()) + pd.DateOffset(years=value)
            else:
                return None
            return future_end_date
        except ValueError:
            return None

def estimate_future_factor_returns(factor_data, future_end_date, frequency):
    """
    Estimate future factor returns based on historical averages.
    """
    # Calculate average returns
    average_returns = factor_data.mean()

    # Calculate the number of periods in the future period
    if frequency == 'monthly':
        periods = (future_end_date.year - datetime.today().year) * 12 + future_end_date.month - datetime.today().month
        periods = max(periods, 1)
    elif frequency == 'daily':
        periods = (future_end_date - datetime.today()).days
        periods = max(periods, 1)
    else:
        periods = 1  # Default to 1 period

    # Estimate future returns using compounding
    try:
        estimated_returns = ((1 + average_returns) ** periods) - 1
        return estimated_returns
    except Exception as e:
        print(f"Error estimating future factor returns: {e}")
        return None

# Run the application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
