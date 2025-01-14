"""
Project: Option pricer for European options
"""

from scipy.stats import norm
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import yfinance as yf
import pandas as pd
from scipy.interpolate import make_interp_spline
import plotly.graph_objects as go

# ------------------------------------
# CLASS FOR MONTE CARLO SIMULATIONS
# ------------------------------------
class MonteCarlo:
    
    def __init__(self, spot, mu, sigma, time_horizon, num_steps, num_simulations):
        self.spot = spot
        self.mu = mu
        self.sigma = sigma
        self.time_horizon = time_horizon
        self.num_steps = num_steps
        self.num_simulations = num_simulations
        self.dt = self.time_horizon / self.num_steps  # Pas de temps

    def simulate_paths(self):
        """
        Simulate paths using Geometric Brownian Motion.
        """
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = self.spot  # Initial spot price

        for t in range(1, self.num_steps + 1):
            Z = np.random.normal(0, 1, self.num_simulations)  # Chocs aléatoires
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
            )
        return paths

    def plot_paths(self, paths):
        """
        Plot simulated price paths using Plotly.
        """
        fig = go.Figure()
        for i in range(min(10, self.num_simulations)):  # Limiter à 10 simulations pour plus de clarté
            fig.add_trace(go.Scatter(
                x=np.linspace(0, self.time_horizon, self.num_steps + 1),
                y=paths[i],
                mode="lines",
                name=f"Simulation {i+1}"
            ))
        fig.update_layout(
            title="Monte Carlo Simulations of Spot Prices",
            xaxis_title="Time (Years)",
            yaxis_title="Spot Price",
            hovermode="x unified"
        )
        return fig

    def calculate_statistics(self, paths):
        """
        Calculate statistics like mean and standard deviation at maturity.
        """
        final_prices = paths[:, -1]  # Dernière colonne = prix à maturité
        mean_price = final_prices.mean()
        std_dev = final_prices.std()
        return mean_price, std_dev

        

# ------------------------------
# CLASS FOR YFINANCE DATA
# ------------------------------
class YFinanceData:
    
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yf.Ticker(ticker)
        self.history = None

    def get_spot_price(self):
        # 1d = price historic for the last day
        self.history = self.data.history(period="1d", auto_adjust=True, actions=False) 
        # ['Close'] = get the close price
        # .iloc[-1] = get the last close price
        return self.history['Close'].iloc[-1] 

    def get_option_chain(self, date_expiration):
        try:
            return self.data.option_chain(date_expiration)
        except Exception as e:
            raise ValueError(f"Error fetching option chain: {e}")

    def get_expiration_dates(self):
        try:
            return self.data.options
        except Exception as e:
            raise ValueError(f"Error fetching expiration dates: {e}")

    def option_maturity_to_period(self, option_maturity):
        if option_maturity <= 0.25:  # ≤ 3 mois
            return "3mo"
        elif option_maturity <= 0.5:  # ≤ 6 mois
            return "6mo"
        elif option_maturity <= 1.0:  # ≤ 1 an
            return "1y"
        else:  # > 1 an
            return "2y"

    def calculate_historical_volatility(self, option_maturity=1.0):
        """
        Calculate historical volatility for the ticker with a dynamically adjusted window based on the option maturity.
        :param option_maturity: Time to maturity (in years).
        :return: Historical volatility (annualized).
        """
        # Convert option maturity to period
        period = self.option_maturity_to_period(option_maturity=option_maturity)

        # Adjust window dynamically based on the option maturity
        window = max(5, int(option_maturity * 252))  # Approximation : 1 year ≈ 252 trading days

        # Fetch historical data
        self.history = self.data.history(period=period, auto_adjust=True, actions=False)

        # Calculate log returns
        log_returns = np.log(self.history['Close'] / self.history['Close'].shift(1))

        # Calculate rolling standard deviation with the adjusted window and annualize
        return log_returns.rolling(window).std() * np.sqrt(252)

 

#------------------------------
# CLASS FOR OPTION
#------------------------------
class Option:
    def __init__(self, Spot, Strike, Maturity, Rate, Div, Sigma):
        self.Spot = Spot
        self.Strike = Strike
        self.Maturity = Maturity
        self.Rate = Rate
        self.Div = Div  # Dividend yield
        self.Sigma = Sigma

        # Calculate d1 and d2, now including Div
        self.d1 = (math.log(self.Spot / self.Strike) + (self.Rate - self.Div + (self.Sigma**2 / 2)) * self.Maturity) / (self.Sigma * math.sqrt(self.Maturity))
        self.d2 = self.d1 - self.Sigma * math.sqrt(self.Maturity)

    def option_price(self):
        call_price = (self.Spot * math.exp(-self.Div * self.Maturity) * norm.cdf(self.d1) 
                      - self.Strike * math.exp(-self.Rate * self.Maturity) * norm.cdf(self.d2))
        put_price = (self.Strike * math.exp(-self.Rate * self.Maturity) * norm.cdf(-self.d2) 
                     - self.Spot * math.exp(-self.Div * self.Maturity) * norm.cdf(-self.d1))
        return call_price, put_price
        
    def option_payoff(self, option_position, spot=None):
        """
        Calcul du payoff pour un call et un put en fonction de la position (Long ou Short).
        :param option_position: "Long" ou "Short"
        :param spot: (Optional) Spot Price à utiliser pour le calcul. Si None, utilise self.Spot.
        :return: call_payoff, put_payoff
        """
        # Utilise self.Spot si spot n'est pas fourni
        spot = spot if spot is not None else self.Spot

        # Calcul du payoff
        if option_position == "Long":
            call_payoff = max(spot - self.Strike, 0)
            put_payoff = max(self.Strike - spot, 0)
        elif option_position == "Short":
            call_payoff = -max(spot - self.Strike, 0)
            put_payoff = -max(self.Strike - spot, 0)

        return call_payoff, put_payoff

    def option_net_profit(self, option_position, call_price=None, put_price=None, payoff=None, spot=None):
        """
        Calcul du profit net pour un call et un put en fonction de la position (Long ou Short).
        :param option_position: "Long" ou "Short"
        :param call_price: (Optional) Prix pré-calculé de l'option Call.
        :param put_price: (Optional) Prix pré-calculé de l'option Put.
        :param payoff: (Optional) Payoff pré-calculé pour l'option (si fourni).
        :param spot: (Optional) Spot Price à utiliser pour le calcul.
        :return: call_net_profit, put_net_profit
        """
        # Si le payoff n'est pas fourni, on le calcule avec option_payoff
        if payoff is None:
            call_payoff, put_payoff = self.option_payoff(option_position, spot=spot)
        else:
            call_payoff, put_payoff = payoff

        # Si le prix du call n'est pas fourni, on le calcule
        if call_price is None:
            call_price, _ = self.option_price()

        # Si le prix du put n'est pas fourni, on le calcule
        if put_price is None:
            _, put_price = self.option_price()

        # Calcul des profits nets
        if option_position == "Long":
            call_net_profit = call_payoff - call_price
            put_net_profit = put_payoff - put_price
        elif option_position == "Short":
            call_net_profit = call_price + call_payoff
            put_net_profit = put_price + put_payoff

        return call_net_profit, put_net_profit

    def option_greeks(self, option_position):
        # delta, theta, vega, rho, gamma
        delta_call = norm.cdf(self.d1)
        delta_put = norm.cdf(self.d1) - 1
        gamma = norm.pdf(self.d1) / (self.Spot * self.Sigma * math.sqrt(self.Maturity))
        theta_call = -((self.Spot * norm.pdf(self.d1) * self.Sigma)) / (2 * math.sqrt(self.Maturity)) - (self.Rate * self.Strike * math.exp(-self.Rate*self.Maturity) * norm.cdf(self.d2))
        theta_put = -((self.Spot * norm.pdf(self.d1) * self.Sigma)) / (2 * math.sqrt(self.Maturity)) + (self.Rate * self.Strike * math.exp(-self.Rate*self.Maturity) * norm.cdf(-self.d2))
        vega = self.Spot * norm.pdf(self.d1) * math.sqrt(self.Maturity)
        rho_call = self.Maturity * self.Strike * math.exp(-self.Rate*self.Maturity) * norm.cdf(self.d2)
        rho_put = -self.Maturity * self.Strike * math.exp(-self.Rate*self.Maturity) * norm.cdf(-self.d2)
        
        # Adjust for position
        if option_position == "Short":
            delta_call *= -1
            delta_put *= -1
            theta_call *= -1
            theta_put *= -1
            rho_call *= -1
            rho_put *= -1
        
        return {
            "Delta Call": delta_call,
            "Delta Put": delta_put,
            "Gamma": gamma,
            "Theta Call": theta_call,
            "Theta Put": theta_put,
            "Vega": vega,
            "Rho Call": rho_call,
            "Rho Put": rho_put
        }

    def plot_greeks(self, option_position):
        """
        Plots the Greeks (Delta, Gamma, Theta, Vega, Rho) as a function of the spot price.
        Takes into account the position (Long or Short) of the option.
        """

        # Generate values for the spot
        spot_values = np.arange(50, 150, 5)

        # Initialize lists for each greek
        greeks_storage = {
            "Delta Call": [],
            "Delta Put": [],
            "Gamma": [],
            "Theta Call": [],
            "Theta Put": [],
            "Vega": [],
            "Rho Call": [],
            "Rho Put": []
        }

        # Calculus for each spot value
        for spot_value in spot_values:
            # Set self.Spot with each spot value
            self.Spot = spot_value

            # Re-calculate d1, d2 & the greeks for each spot value
            self.d1 = (math.log(self.Spot / self.Strike) + (self.Rate + (self.Sigma**2 / 2)) * self.Maturity) / (self.Sigma * math.sqrt(self.Maturity))
            self.d2 = self.d1 - self.Sigma * math.sqrt(self.Maturity)

            # Call method to obtain results
            greeks = self.option_greeks(option_position)  # Pass the option position

            # Add greek values to the corresponding lists
            for key in greeks_storage:
                greeks_storage[key].append(greeks[key])

        # Plot each greek
        for greek_name, values in greeks_storage.items():
            plt.figure()
            plt.plot(spot_values, values, label=greek_name)
            plt.axvline(x=self.Strike, color='red', linestyle='--', label='Strike')
            plt.title(f"{greek_name} vs Spot Price ({option_position})")
            plt.xlabel("Spot Price (S)")
            plt.ylabel(f"{greek_name}")
            plt.legend()
            plt.grid()
            plt.show()
 
    def calculate_pnl(self, purchase_price_call, purchase_price_put, option_position):
        call_price, put_price = self.option_price()
        if option_position == "Long":
            pnl_call = call_price - purchase_price_call
            pnl_put = put_price - purchase_price_put
        elif option_position == "Short":
            pnl_call = purchase_price_call - call_price
            pnl_put = purchase_price_put - put_price
        return pnl_call, pnl_put
    
    def implied_vol(self, mkt_call_price, mkt_put_price, tolerance=1e-6, max_iter=2000):
        
        # Initialize an estimation for sigma
        sigma_call = 0.2
        sigma_put = 0.2
    
        # Initialize differences and flags for convergence
        diff_call = float('inf')
        diff_put = float('inf')
        found_call = False
        found_put = False
    
        for test in range(max_iter):
            # Create temporary options
            temp_call_option = Option(self.Spot, self.Strike, self.Maturity, self.Rate, self.Div, sigma_call)
            temp_put_option = Option(self.Spot, self.Strike, self.Maturity, self.Rate, self.Div, sigma_put)
    
            # Calculate theoretical prices
            theor_temp_call_price, _ = temp_call_option.option_price()
            _, theor_temp_put_price = temp_put_option.option_price()
    
            # Calculate differences
            diff_call = theor_temp_call_price - mkt_call_price
            diff_put = theor_temp_put_price - mkt_put_price
    
            # Check for convergence
            if abs(diff_call) <= tolerance:
                found_call = True
            if abs(diff_put) <= tolerance:
                found_put = True
    
            # If both converged, return the results
            if found_call and found_put:
                return sigma_call, sigma_put
    
            # Calculate Vega (same for call and put)
            vega_temp_option = temp_call_option.option_greeks()["Vega"]
    
            # Prevent division by zero or instability
            if vega_temp_option < 1e-6:
                vega_temp_option = 1e-6
    
            # Adjust sigma only if not yet converged
            if not found_call:
                sigma_call -= diff_call / vega_temp_option
            if not found_put:
                sigma_put -= diff_put / vega_temp_option
    
        # Handle non-convergence
        if not found_call and not found_put:
            raise ValueError("Neither implied volatility converged within the maximum iterations.")
        elif not found_call:
            return None, sigma_put
        elif not found_put:
            return sigma_call, None
 
    def implied_vol_bisection(self, mkt_call_price, mkt_put_price, tolerance=1e-6, max_iter=500):
        """
        Compute implied volatilities for call and put options using the bisection method.
        :param mkt_call_price: Market price of the call option
        :param mkt_put_price: Market price of the put option
        :param tolerance: Convergence threshold
        :param max_iter: Maximum iterations
        :return: Implied vol for call and put options
        """
        def bisection(price_target, option_type):
            if price_target < 0.001:  # Prix très bas
                return None  # Retourner N/A directement
            lower_bound = 0.001  # Minimum volatility
            upper_bound = 500   # Maximum volatility

            for _ in range(max_iter):
                mid_vol = (lower_bound + upper_bound) / 2.0
                temp_option = Option(self.Spot, self.Strike, self.Maturity, self.Rate, self.Div, mid_vol)
                
                # Prix théorique
                if option_type == "Call":
                    theor_price, _ = temp_option.option_price()
                else:
                    _, theor_price = temp_option.option_price()

                # Ajuster les bornes
                if theor_price > price_target:
                    upper_bound = mid_vol
                else:
                    lower_bound = mid_vol

                # Vérifier convergence
                if abs(theor_price - price_target) < tolerance:
                    return mid_vol
            return None  # Si pas de convergence

        # Calculer pour call et put
        imp_vol_call = bisection(mkt_call_price, "Call")
        imp_vol_put = bisection(mkt_put_price, "Put")

        return imp_vol_call, imp_vol_put   

    def delta_hedging(self, option_qty, option_position):
        """
        Calculate hedging positions and quantities for Call and Put options.
        Returns a DataFrame for better scalability.
        """
        # Vérification de l'input
        if option_position not in ["Long", "Short"]:
            raise ValueError("option_position must be 'Long' or 'Short'")
        
        # Calcul des deltas
        delta_call = norm.cdf(self.d1)
        delta_put = norm.cdf(self.d1) - 1
        
        # Calcul des hedging quantities
        call_hedging_qty = abs(delta_call * option_qty)
        put_hedging_qty = abs(delta_put * option_qty)
        
        # Calcul des hedging costs
        call_hedging_cost = call_hedging_qty * self.Spot
        put_hedging_cost = put_hedging_qty * self.Spot
        
        # Logique de couverture
        if option_position == "Long":
            hedge_data = {
                "Option Type": ["Call", "Put"],
                "Hedging Position": ["Short shares", "Long shares"],
                "Hedging Quantity": [call_hedging_qty, put_hedging_qty],
                "Hedging Cost": [call_hedging_cost, put_hedging_cost]
            }
        elif option_position == "Short":
            hedge_data = {
                "Option Type": ["Call", "Put"],
                "Hedging Position": ["Long shares", "Short shares"],
                "Hedging Quantity": [call_hedging_qty, put_hedging_qty],
                "Hedging Cost": [call_hedging_cost, put_hedging_cost]
            }

        # Retourner un DataFrame
        return pd.DataFrame(hedge_data)

    def plot_dynamic_hedging(self, option_qty):
        """
        Plot dynamic delta-hedging (quantity and cost) as a function of the Spot Price.
        :param option_qty: Quantity of options held
        """
        # Initialiser les listes de stockage
        spot_values = np.linspace(self.Spot * 0.5, self.Spot * 1.5, 100)
        call_hedge_qty, call_hedge_cost = [], []
        put_hedge_qty, put_hedge_cost = [], []

        for spot in spot_values:
            # Recalculer d1 pour chaque spot
            d1 = (math.log(spot / self.Strike) + (self.Rate + (self.Sigma**2 / 2)) * self.Maturity) / (self.Sigma * math.sqrt(self.Maturity))
            delta_call = norm.cdf(d1)
            delta_put = norm.cdf(d1) - 1

            # Calculer les quantités et les coûts
            call_qty = delta_call * option_qty
            put_qty = -delta_put * option_qty
            call_cost = abs(call_qty) * spot
            put_cost = abs(put_qty) * spot

            # Stocker les valeurs
            call_hedge_qty.append(call_qty)
            call_hedge_cost.append(call_cost)
            put_hedge_qty.append(put_qty)
            put_hedge_cost.append(put_cost)

        # Graphique interactif pour Call
        fig_call = go.Figure()
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=call_hedge_qty, mode="lines",
            name="Call Hedging Quantity",
            hovertemplate="<b>Spot:</b> %{x:.2f}<br>"
                          "<b>Hedging Qty:</b> %{y:.2f}"
        ))
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=call_hedge_cost, mode="lines",
            name="Call Hedging Cost",
            hovertemplate="<b>Spot:</b> %{x:.2f}<br>"
                          "<b>Hedging Cost:</b> %{y:.2f}"
        ))
        fig_call.update_layout(
            title="Dynamic Hedging for Call Options",
            xaxis_title="Spot Price",
            yaxis_title="Hedging Quantity / Cost",
            hovermode="x unified"
        )

        # Graphique interactif pour Put
        fig_put = go.Figure()
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=put_hedge_qty, mode="lines",
            name="Put Hedging Quantity",
            hovertemplate="<b>Spot:</b> %{x:.2f}<br>"
                          "<b>Hedging Qty:</b> %{y:.2f}"
        ))
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=put_hedge_cost, mode="lines",
            name="Put Hedging Cost",
            hovertemplate="<b>Spot:</b> %{x:.2f}<br>"
                          "<b>Hedging Cost:</b> %{y:.2f}"
        ))
        fig_put.update_layout(
            title="Dynamic Hedging for Put Options",
            xaxis_title="Spot Price",
            yaxis_title="Hedging Quantity / Cost",
            hovermode="x unified"
        )

        return fig_call, fig_put
    
    def gamma_hedge(self, MonteCarlo, transaction_fee):
        
        # Initialiser les colonnes sous forme de dictionnaire
        def initialize_results():
            return {
                "t": [],
                "Spot": [],
                "Delta": [],
                "Gamma": [],
                "Initial Delta Hedge Position": [],
                "Delta Hedge Quantity": [],
                "Delta Hedge Cost": [],
                "Gamma Hedge Position": [],
                "Gamma Hedge Quantity": [],
                "Gamma Hedge Cost": [],
                "Gamma PnL": [],
                "Transaction Cost": [],
                "Net Gamma PnL": []
            }
        
        # Initialisation des résultats pour Call et Put
        gamma_hedge_results_call = initialize_results()
        gamma_hedge_results_put = initialize_results()
            
        # ------------------------------    
        # Step 1 : Initialize for t=O
        # ------------------------------ 
        
        # Create an initial option (same for call and put)
        initial_option = Option(Spot, Strike, Maturity, Rate, Div, Sigma)

        # Col t
        t = 0
        gamma_hedge_results_call['t'].append(t)
        gamma_hedge_results_put['t'].append(t)

        # Col "Spot"
        initial_spot = self.Spot
        gamma_hedge_results_call['Spot'].append(initial_spot)
        gamma_hedge_results_put['Spot'].append(initial_spot)

        # Get initial greek data
        initial_greek_data = initial_option.option_greeks(option_position)
            
        # Col "Delta"
        initial_delta_call = initial_greek_data["Delta Call"]
        gamma_hedge_results_call['Delta'].append(initial_delta_call)
        initial_delta_put = initial_greek_data["Delta Put"]
        gamma_hedge_results_put['Delta'].append(initial_delta_put)

        # Col "Gamma" (same for call and put)
        initial_gamma = initial_greek_data["Gamma"]
        gamma_hedge_results_call['Gamma'].append(initial_gamma)
        gamma_hedge_results_put['Gamma'].append(initial_gamma)

        # Get initial delta hedging data
        initial_delta_hedging_data = initial_option.delta_hedging(option_qty, option_position)
        
        # Col "Initial Delta Hedge Position"
        initial_delta_call_hedge_position = initial_delta_hedging_data['Hedging Position'][0]
        gamma_hedge_results_call['Initial Delta Hedge Position'].append(initial_delta_call_hedge_position)

        initial_delta_put_hedge_position = initial_delta_hedging_data['Hedging Position'][1]
        gamma_hedge_results_put['Initial Delta Hedge Position'].append(initial_delta_put_hedge_position)

        # Col "Delta Hedge Quantity"
        initial_delta_call_hedge_qty = initial_delta_hedging_data["Hedging Quantity"][0]
        gamma_hedge_results_call['Delta Hedge Quantity'].append(initial_delta_call_hedge_qty)

        initial_delta_put_hedge_qty = initial_delta_hedging_data["Hedging Quantity"][1]
        gamma_hedge_results_put['Delta Hedge Quantity'].append(initial_delta_put_hedge_qty)

        # Col "Delta Hedge Cost"
        if initial_delta_call_hedge_position == "Short shares":  # --> will be positive to show we gain money from short position
            initial_delta_call_hedge_cost = initial_delta_hedging_data["Hedging Cost"][0]
        else:
            initial_delta_call_hedge_cost = (-1) * initial_delta_hedging_data["Hedging Cost"][0]
        gamma_hedge_results_call['Delta Hedge Cost'].append(initial_delta_call_hedge_cost)     

        if initial_delta_put_hedge_position == "Short shares":     
            initial_delta_put_hedge_cost = initial_delta_hedging_data["Hedging Cost"][1]
        else:
            initial_delta_put_hedge_cost = (-1) * initial_delta_hedging_data["Hedging Cost"][1]
        gamma_hedge_results_put['Delta Hedge Cost'].append(initial_delta_put_hedge_cost)
        
        # Col "Gamma Hedge Position"
        initial_gamma_call_hedge_position = "-"
        initial_gamma_put_hedge_position = "-"
        gamma_hedge_results_call['Gamma Hedge Position'].append(initial_gamma_call_hedge_position)
        gamma_hedge_results_put['Gamma Hedge Position'].append(initial_gamma_put_hedge_position)

        # Col "Gamma Hedge Quantity"
        initial_gamma_call_hedge_qty = 0
        initial_gamma_put_hedge_qty = 0
        gamma_hedge_results_call['Gamma Hedge Quantity'].append(initial_gamma_call_hedge_qty)
        gamma_hedge_results_put['Gamma Hedge Quantity'].append(initial_gamma_put_hedge_qty)
        
        # Col "Gamma Hedge Cost"
        initial_gamma_call_hedge_cost = 0
        initial_gamma_put_hedge_cost = 0
        gamma_hedge_results_call['Gamma Hedge Cost'].append(initial_gamma_call_hedge_cost)
        gamma_hedge_results_put['Gamma Hedge Cost'].append(initial_gamma_put_hedge_cost)
        
        # Col "Gamma PnL"
        initial_gamma_call_pnl = 0
        initial_gamma_put_pnl = 0
        gamma_hedge_results_call['Gamma PnL'].append(initial_gamma_call_pnl)
        gamma_hedge_results_put['Gamma PnL'].append(initial_gamma_put_pnl)
        
        # Col "Transaction Cost"
        initial_transaction_fee = 0
        gamma_hedge_results_call['Transaction Cost'].append(initial_transaction_fee)
        gamma_hedge_results_put['Transaction Cost'].append(initial_transaction_fee)
        
        # Col "Net Gamma PnL"
        initial_net_gamma_call_pnl = 0
        initial_net_gamma_put_pnl = 0
        gamma_hedge_results_call['Net Gamma PnL'].append(initial_net_gamma_call_pnl)
        gamma_hedge_results_put['Net Gamma PnL'].append(initial_net_gamma_put_pnl)
        
        # Create spots with Monte Carlo (same for call and put)
        sim_spot = MonteCarlo.simulate_paths()

        # Initialize previous variables
        previous_spot = initial_spot

        previous_delta_call = initial_delta_call    
        previous_delta_put = initial_delta_put

        previous_delta_call_hedge_position = initial_delta_call_hedge_position
        previous_delta_put_hedge_position = initial_delta_put_hedge_position

        previous_delta_call_hedge_qty = initial_delta_call_hedge_qty
        previous_delta_put_hedge_qty = initial_delta_put_hedge_qty
       
        previous_gamma_call_hedge_cost = initial_gamma_call_hedge_cost
        previous_gamma_put_hedge_cost = initial_gamma_put_hedge_cost

        previous_gamma_call_pnl = initial_gamma_call_pnl
        previous_gamma_put_pnl = initial_gamma_put_pnl

        # ------------------------------    
        # Step 2 : Initialize from t=1
        # ------------------------------ 
        for t in range(1, num_steps +1):  
            
            # Col "t"
            gamma_hedge_results_call['t'].append(t)
            gamma_hedge_results_put['t'].append(t)

            # Col "Spot"
            current_spot = sim_spot[0, t]  # Utiliser la première simulation
            gamma_hedge_results_call['Spot'].append(current_spot)
            gamma_hedge_results_put['Spot'].append(current_spot)
                
            # Creer une option temporaire avec le spot de monte carlo (même pour call et put)
            current_option = Option(current_spot, Strike, Maturity, Rate, Div, Sigma)
             
            # Get greek data (same for call and put)
            current_greek_data = current_option.option_greeks(option_position)
            
            # Col "Delta"
            current_delta_call = current_greek_data["Delta Call"]
            gamma_hedge_results_call['Delta'].append(current_delta_call)

            current_delta_put = current_greek_data["Delta Put"]
            gamma_hedge_results_put['Delta'].append(current_delta_put)
           
            # Col "Gamma"
            current_gamma_call = current_greek_data["Gamma"]
            gamma_hedge_results_call['Gamma'].append(current_gamma_call)

            current_gamma_put = current_greek_data["Gamma"]
            gamma_hedge_results_put['Gamma'].append(current_gamma_put)

             # Get delta hedging data
            current_delta_hedging_data = current_option.delta_hedging(option_qty, option_position)
            
             # Col "Initial Delta Hedge Position"
            current_delta_call_hedge_position = current_delta_hedging_data['Hedging Position'][0]
            gamma_hedge_results_call['Initial Delta Hedge Position'].append(current_delta_call_hedge_position)

            current_delta_put_hedge_position = current_delta_hedging_data['Hedging Position'][1]
            gamma_hedge_results_put['Initial Delta Hedge Position'].append(current_delta_put_hedge_position)

            # Col "Delta Hedge Quantity" 
            current_delta_call_hedge_qty = current_delta_hedging_data["Hedging Quantity"][0]
            gamma_hedge_results_call['Delta Hedge Quantity'].append(current_delta_call_hedge_qty)

            current_delta_put_hedge_qty = current_delta_hedging_data["Hedging Quantity"][1]
            gamma_hedge_results_put['Delta Hedge Quantity'].append(current_delta_put_hedge_qty)

            # Col "Delta Hedge Cost" 
            if current_delta_call_hedge_position == "Short shares": # --> will be positive to show we gain money from short position
                current_delta_call_hedge_cost = current_delta_hedging_data["Hedging Cost"][0]
            else:
                current_delta_call_hedge_cost = (-1) * current_delta_hedging_data["Hedging Cost"][0]
            gamma_hedge_results_call['Delta Hedge Cost'].append(current_delta_call_hedge_cost)

            if current_delta_put_hedge_position == "Short shares":
                current_delta_put_hedge_cost = current_delta_hedging_data["Hedging Cost"][1]
            else:
                current_delta_put_hedge_cost = (-1) * current_delta_hedging_data["Hedging Cost"][1]
            gamma_hedge_results_put['Delta Hedge Cost'].append(current_delta_put_hedge_cost)

            # Col "Gamma Hedge Position"

            # Call
            current_delta_diff_call = current_delta_call - previous_delta_call

            if current_delta_diff_call > 0:  # Delta augmente (besoin de vendre)
                current_gamma_call_hedge_position = "Short shares"
            else:  # Delta diminue (besoin d'acheter)
                current_gamma_call_hedge_position = "Long shares"
            gamma_hedge_results_call['Gamma Hedge Position'].append(current_gamma_call_hedge_position)

            # Put
            current_delta_diff_put = current_delta_put - previous_delta_put

            if current_delta_diff_put > 0:  # Delta augmente (besoin de vendre)
                current_gamma_put_hedge_position = "Short shares"
            else:  # Delta diminue (besoin d'acheter)
                current_gamma_put_hedge_position = "Long shares"
            gamma_hedge_results_put['Gamma Hedge Position'].append(current_gamma_put_hedge_position)

            # Col "Gamma Hedge Quantity"
            current_gamma_call_hedge_qty = abs(current_delta_call_hedge_qty - previous_delta_call_hedge_qty)
            gamma_hedge_results_call['Gamma Hedge Quantity'].append(current_gamma_call_hedge_qty)

            current_gamma_put_hedge_qty = abs(current_delta_put_hedge_qty - previous_delta_put_hedge_qty)
            gamma_hedge_results_put['Gamma Hedge Quantity'].append(current_gamma_put_hedge_qty)

            # Col "Gamma Hedge Cost"
            if current_gamma_call_hedge_position == "Short shares":
                current_gamma_call_hedge_cost = current_gamma_call_hedge_qty * current_spot
            else:
                current_gamma_call_hedge_cost = (-1) * current_gamma_call_hedge_qty * current_spot
            gamma_hedge_results_call['Gamma Hedge Cost'].append(current_gamma_call_hedge_cost)

            if current_gamma_put_hedge_position == "Short shares":
                current_gamma_put_hedge_cost = current_gamma_put_hedge_qty * current_spot
            else:
                current_gamma_put_hedge_cost = (-1) * current_gamma_put_hedge_qty * current_spot
            gamma_hedge_results_put['Gamma Hedge Cost'].append(current_gamma_put_hedge_cost)

            # Col "Gamma PnL"
            # Since "Gamma Hedge Cost" is already positive/negative for short/long position, we just add the transaction cost
            current_gamma_call_pnl = current_gamma_call_hedge_cost + previous_gamma_call_pnl
            gamma_hedge_results_call['Gamma PnL'].append(current_gamma_call_pnl)

            current_gamma_put_pnl = current_gamma_put_hedge_cost + previous_gamma_put_pnl
            gamma_hedge_results_put['Gamma PnL'].append(current_gamma_put_pnl)

            current_transaction_fee = 0  # Valeur par défaut
            # Col "Transaction Cost"
            if current_gamma_call_hedge_qty != 0:
                if current_gamma_call_hedge_position == "Short shares":
                    current_transaction_fee = (-1) * current_gamma_call_hedge_cost * transaction_fee
                else:
                    current_transaction_fee = current_gamma_call_hedge_cost * transaction_fee
            gamma_hedge_results_call['Transaction Cost'].append(current_transaction_fee)

            if current_gamma_put_hedge_qty != 0:
                if current_gamma_put_hedge_position == "Short shares":
                    current_transaction_fee = (-1) * current_gamma_put_hedge_cost * transaction_fee
                else:
                    current_transaction_fee = current_gamma_put_hedge_cost * transaction_fee
            gamma_hedge_results_put['Transaction Cost'].append(current_transaction_fee)
                    
            # Col "Net Gamma PnL"
            current_net_gamma_call_pnl = current_gamma_call_pnl + current_transaction_fee
            gamma_hedge_results_call['Net Gamma PnL'].append(current_net_gamma_call_pnl)

            current_net_gamma_put_pnl = current_gamma_put_pnl + current_transaction_fee
            gamma_hedge_results_put['Net Gamma PnL'].append(current_net_gamma_put_pnl)
             
            # Actualize previous variables
            previous_spot = current_spot

            previous_delta_call = current_delta_call
            previous_delta_put = current_delta_put

            previous_delta_call_hedge_position = current_gamma_call_hedge_position
            previous_delta_put_hedge_position = current_gamma_put_hedge_position

            previous_delta_call_hedge_qty = current_delta_call_hedge_qty
            previous_delta_put_hedge_qty = current_delta_put_hedge_qty

            previous_gamma_call_hedge_cost = current_gamma_call_hedge_cost
            previous_gamma_put_hedge_cost = current_gamma_put_hedge_cost

            previous_gamma_call_pnl = current_gamma_call_pnl
            previous_gamma_put_pnl = current_gamma_put_pnl

        # Convert into df
        df_gamma_hedge_results_call = pd.DataFrame(gamma_hedge_results_call)
        df_gamma_hedge_results_put = pd.DataFrame(gamma_hedge_results_put)
        
        return df_gamma_hedge_results_call, df_gamma_hedge_results_put

    def plot_delta_hedging_from_gamma_hedge(self, df_call, df_put, strike):
        """
        Plot hedging costs based on real data from the hedging results tables and add Strike Price annotation.
        
        :param df_call: DataFrame containing hedging results for Call options.
        :param df_put: DataFrame containing hedging results for Put options.
        :param strike: Strike price of the option.
        :return: Figures for Call and Put dynamic hedging.
        """
        # Graphique pour les Calls
        fig_call = go.Figure()
        fig_call.add_trace(go.Scatter(
            x=df_call["Spot"], y=df_call["Delta Hedge Cost"], mode="lines",
            name="Call Hedging Cost",
            line=dict(color="red"),
            hovertemplate="<b>Spot:</b> %{x:.2f}<br><b>Hedging Cost:</b> %{y:.2f}"
        ))
        fig_call.add_vline(
            x=strike, 
            line_dash="dot", 
            line_color="blue", 
            annotation_text=f"Strike: {strike}",
            annotation_position="top left"  # Annotation à gauche
        )
        fig_call.update_layout(
            title="Dynamic Hedging for Call Options",
            xaxis_title="Spot Price",
            yaxis_title="Hedging Cost",
            hovermode="x unified",
            template="plotly_dark"
        )

        # Graphique pour les Puts
        fig_put = go.Figure()
        fig_put.add_trace(go.Scatter(
            x=df_put["Spot"], y=df_put["Delta Hedge Cost"], mode="lines",
            name="Put Hedging Cost",
            line=dict(color="red"),
            hovertemplate="<b>Spot:</b> %{x:.2f}<br><b>Hedging Cost:</b> %{y:.2f}"
        ))
        fig_put.add_vline(
            x=strike, 
            line_dash="dot", 
            line_color="blue", 
            annotation_text=f"Strike: {strike}",
            annotation_position="top left"  # Annotation à gauche
        )
        fig_put.update_layout(
            title="Dynamic Hedging for Put Options",
            xaxis_title="Spot Price",
            yaxis_title="Hedging Cost",
            hovermode="x unified",
            template="plotly_dark"
        )

        return fig_call, fig_put

    def call_spread(self, option_position, K1, K2, call_K1_price=None, call_K2_price=None):
        """
        Long Call Spread Strategy: Buy a call option (K1) and sell another call option (K2).
        Short Call Spread Strategy: Sell a call option (K1) and buy another call option (K2).
        :param option_position: "Long" or "Short"
        :param K1: Strike price of the lower call option.
        :param K2: Strike price of the higher call option.
        :param call_K1_price: (Optional) Prix pré-calculé pour K1.
        :param call_K2_price: (Optional) Prix pré-calculé pour K2.
        :return: DataFrame with detailed results for the strategy.
        """
        # Validation des strikes
        if K1 >= K2:
            raise ValueError("K1 (lower strike) must be less than K2 (higher strike) for a Call Spread.")

        # Création des deux options
        call_K1 = Option(self.Spot, K1, self.Maturity, self.Rate, self.Div, self.Sigma)
        call_K2 = Option(self.Spot, K2, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if call_K1_price is None or call_K2_price is None:
            call_K1_price, _ = call_K1.option_price()
            call_K2_price, _ = call_K2.option_price()

        # Long Call K1 et Short Call K2
        if option_position == "Long":
            call_K1_position = "Long"
            call_K2_position = "Short"
            call_spread_position = "Long"
        elif option_position == "Short":
            call_K1_position = "Short"
            call_K2_position = "Long"
            call_spread_position = "Short"

        # Calculs du payoff et profit pour les options individuelles
        call_K1_payoff, _ = call_K1.option_payoff(call_K1_position)
        call_K2_payoff, _ = call_K2.option_payoff(call_K2_position)
        call_K1_profit, _ = call_K1.option_net_profit(call_K1_position, call_price=call_K1_price)
        call_K2_profit, _ = call_K2.option_net_profit(call_K2_position, call_price=call_K2_price)

        # Calculs prix, payoff et profit du Call Spread global
        call_spread_price = call_K1_price - call_K2_price
        call_spread_payoff = call_K1_payoff + call_K2_payoff
        call_spread_profit = call_K1_profit + call_K2_profit

        # Création d'une table des résultats combinée pour les options individuelles et globales
        df_call_spread = pd.DataFrame({
            "Option": [f"{call_K1_position} Call K1", f"{call_K2_position} Call K2", f"{call_spread_position} Call Spread"],
            "Spot": [self.Spot, self.Spot, self.Spot],
            "Strike": [K1, K2, f"{K1}-{K2}"],
            "Payoff": [f"{call_K1_payoff:.2f}", f"{call_K2_payoff:.2f}", f"{call_spread_payoff:.2f}"],
            "Price": [f"{call_K1_price:.2f}", f"{call_K2_price:.2f}", f"{call_spread_price:.2f}"],
            "Profit": [f"{call_K1_profit:.2f}", f"{call_K2_profit:.2f}", f"{call_spread_profit:.2f}"]
        })

        return df_call_spread

    def plot_call(self, option_position, spot_values, call_price=None):
        if call_price is None:
            call_price, _ = self.option_price()

        call_payoffs, call_profits = [], []

        for spot in spot_values:
            temp_option = Option(spot, option.Strike, option.Maturity, option.Rate, option.Div, option.Sigma)
            call_payoff, _ = temp_option.option_payoff(option_position)
            call_net_profit, _ = option.option_net_profit(option_position, call_price=call_price, spot=spot)
            call_payoffs.append(call_payoff)
            call_profits.append(call_net_profit)

        profit_area_call = [max(0, p) for p in call_profits]
        loss_area_call = [min(0, p) for p in call_profits]

        fig_call = go.Figure()

        # Add Call Payoff Line
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=call_payoffs, mode="lines",
            name="Call Payoff", line=dict(color="blue")
        ))

        # Add Call Profit Line
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=call_profits, mode="lines",
            name="Call Profit", line=dict(color="green")
        ))

        # Add Profit and Loss Areas
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=profit_area_call, fill='tozeroy', mode='none',
            fillcolor='rgba(0, 255, 0, 0.2)', name="Profit Area"
        ))
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=loss_area_call, fill='tozeroy', mode='none',
            fillcolor='rgba(255, 0, 0, 0.2)', name="Loss Area"
        ))

        # Add Premium Line
        fig_call.add_trace(go.Scatter(
            x=spot_values, y=[-call_price] * len(spot_values) if option_position == "Long" else [call_price] * len(spot_values),
            mode="lines", name="Call Premium", line=dict(color="purple", dash="dot")
        ))

        # Add Strike Price Line
        fig_call.add_vline(
            x=option.Strike, line_dash="dash", line_color="blue",
            annotation_text=f"Strike: {option.Strike}",
            annotation_position = "top left",
            annotation_font=dict(color="blue")
        )

        # Add Spot Price Line
        fig_call.add_vline(
            x=option.Spot, line_dash="dot", line_color="green",
            annotation_text=f"Spot: {option.Spot}",
            annotation_position="top right" if self.Spot == self.Strike else "top left",
            annotation_font=dict(color="green")
        )

        # Add Breakeven Point Line
        breakeven_call = option.Strike + call_price
        fig_call.add_vline(
            x=breakeven_call, line_dash="dot", line_color="orange",
            annotation_text=f"BE: {breakeven_call:.2f}",
            annotation_position="top right" ,
            annotation_font=dict(color="orange")
        )

        # Add Horizontal Line at y=0
        fig_call.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)

        # Annotate Call Premium
        fig_call.add_annotation(
            x=spot_values[-1], y=(-call_price if option_position == "Long" else call_price),
            text=f"Call Premium: {call_price:.2f}", showarrow=False,
            font=dict(color="purple", size=12), yshift=(-10 if option_position == "Long" else 10)
        )

        fig_call.update_layout(
            title=f"{option_position} Call Option Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Payoff / Profit",
            hovermode="x unified", template="plotly_dark"
        )

        return fig_call

    def plot_put(self, option_position, spot_values, put_price=None):
        if put_price is None:
            _, put_price = self.option_price()

        put_payoffs, put_profits = [], []

        for spot in spot_values:
            temp_option = Option(spot, option.Strike, option.Maturity, option.Rate, option.Div, option.Sigma)
            _, put_payoff = temp_option.option_payoff(option_position)
            _, put_net_profit = option.option_net_profit(option_position, put_price=put_price, spot=spot)
            put_payoffs.append(put_payoff)
            put_profits.append(put_net_profit)

        profit_area_put = [max(0, p) for p in put_profits]
        loss_area_put = [min(0, p) for p in put_profits]

        fig_put = go.Figure()

        # Add Put Payoff Line
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=put_payoffs, mode="lines",
            name="Put Payoff", line=dict(color="red")
        ))

        # Add Put Profit Line
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=put_profits, mode="lines",
            name="Put Profit", line=dict(color="orange")
        ))

        # Add Profit and Loss Areas
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=profit_area_put, fill='tozeroy', mode='none',
            fillcolor='rgba(0, 255, 0, 0.2)', name="Profit Area"
        ))
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=loss_area_put, fill='tozeroy', mode='none',
            fillcolor='rgba(255, 0, 0, 0.2)', name="Loss Area"
        ))

        # Add Premium Line
        fig_put.add_trace(go.Scatter(
            x=spot_values, y=[-put_price] * len(spot_values) if option_position == "Long" else [put_price] * len(spot_values),
            mode="lines", name="Put Premium", line=dict(color="purple", dash="dot")
        ))

        # Add Strike Price Line
        fig_put.add_vline(
            x=option.Strike, line_dash="dash", line_color="red",
            annotation_text=f"Strike: {option.Strike}",
            annotation_position="top right",
            annotation_font=dict(color="red")
        )

        # Add Spot Price Line
        fig_put.add_vline(
            x=option.Spot, line_dash="dot", line_color="green",
            annotation_text=f"Spot: {option.Spot}",
            annotation_position="top left" if self.Spot == self.Strike else "top right",
            annotation_font=dict(color="green")
        )

        # Add Breakeven Point Line
        breakeven_put = option.Strike - put_price
        fig_put.add_vline(
            x=breakeven_put, line_dash="dot", line_color="orange",
            annotation_text=f"BE: {breakeven_put:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="orange")
        )

        # Add Horizontal Line at y=0
        fig_put.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)

        # Annotate Put Premium
        fig_put.add_annotation(
            x=spot_values[-1], y=(-put_price if option_position == "Long" else put_price),
            text=f"Put Premium: {put_price:.2f}", showarrow=False,
            font=dict(color="purple", size=12), yshift=(-10 if option_position == "Long" else 10)
        )

        fig_put.update_layout(
            title=f"{option_position} Put Option Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Payoff / Profit",
            hovermode="x unified", template="plotly_dark"
        )

        return fig_put

    def plot_call_spread(self, option_position, spot, K1, K2, maturity, rate, div, sigma):
        # Création des options pour K1 et K2
        option_K1 = Option(spot, K1, maturity, rate, div, sigma)
        option_K2 = Option(spot, K2, maturity, rate, div, sigma)

        # Long Call K1 et Short Call K2
        if option_position == "Long":
            call_K1_position = "Long"
            call_K2_position = "Short"
        elif option_position == "Short":
            call_K1_position = "Short"
            call_K2_position = "Long"

        # Calcul des prix des options
        price_K1, _ = option_K1.option_price()
        price_K2, _ = option_K2.option_price()
        call_spread_price =  price_K1 - price_K2

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les résultats pour les deux options et la stratégie
        profit_K1, profit_K2, spread_profit = [], [], []

        for s in spot_values:
            temp_K1 = Option(s, K1, maturity, rate, div, sigma)
            temp_K2 = Option(s, K2, maturity, rate, div, sigma)

            p_K1, _ = temp_K1.option_net_profit(call_K1_position, call_price=price_K1, spot=s)
            profit_K1.append(p_K1)

            p_K2, _ = temp_K2.option_net_profit(call_K2_position, call_price=price_K2, spot=s)
            profit_K2.append(p_K2)

            total_profit = p_K1 + p_K2
            spread_profit.append(total_profit)

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour K1, K2 et Call Spread
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K1, mode="lines", name=f"Profit {call_K1_position} Call K1", line=dict(color="blue", width=1 )))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K2, mode="lines", name=f"Profit {call_K2_position} Call K2", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=spread_profit, mode="lines", name=f"Profit {option_position} Call Spread", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in spread_profit]
        loss_area = [min(0, p) for p in spread_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes des strikes, breakeven et des prix
        breakeven = K1 + (price_K1 - price_K2)

        # Ligne pour le Spot Price
        fig.add_vline(x=spot, line_dash="dot", line_color="green", annotation_text=f"Spot: {spot}", annotation_font=dict(color="green"))

        # Ligne pour K1
        fig.add_vline(x=K1, line_dash="dash", line_color="blue", opacity=0.5,
                    annotation_text=f"K1: {K1}", annotation_position="top right", annotation_font=dict(color="blue"))

        # Ligne pour K2
        fig.add_vline(x=K2, line_dash="dash", line_color="pink", opacity=0.5,
                    annotation_text=f"K2: {K2}", annotation_position="top right", annotation_font=dict(color="pink"))

        # Ligne pour Breakevens
        fig.add_vline(x=breakeven, line_dash="dot", line_color="orange", annotation_text=f"BE: {breakeven:.2f}",
                    annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix
        fig.add_hline(y=-price_K1 if call_K1_position == "Long" else price_K1, line_dash="dot", line_color="blue",
                    annotation_text=f"{call_K1_position} Call K1 Price: {price_K1:.2f}", annotation_position="right", annotation_font=dict(color="blue"))

        fig.add_hline(y=-price_K2 if call_K2_position == "Long" else price_K2, line_dash="dot", line_color="pink",
                    annotation_text=f"{call_K2_position} Call K2 Price: {price_K2:.2f}", annotation_position="right", annotation_font=dict(color="pink"))

        fig.add_hline(y=-call_spread_price if option_position == "Long" else call_spread_price, line_dash="dot", line_color="green",
                    annotation_text=f"{option_position} Call Spread Price: {call_spread_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Call Spread Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Profit / Payoff",
            hovermode="x unified", template="plotly_dark"
        )

        return fig

    def put_spread(self, option_position, K1, K2, put_K1_price=None, put_K2_price=None):
        """
        Long Put Spread Strategy: Buy a put option (K2) and sell another put option (K1).
        Short Put Spread Strategy: Sell a put option (K2) and buy another put option (K1).
        :param option_position: "Long" or "Short"
        :param K1: Strike price of the lower put option.
        :param K2: Strike price of the higher put option.
        :param put_K1_price: (Optional) Prix pré-calculé pour K1.
        :param put_K2_price: (Optional) Prix pré-calculé pour K2.
        :return: DataFrame with detailed results for the strategy.
        """
        # Validation des strikes
        if K1 >= K2:
            raise ValueError("K1 (lower strike) must be less than K2 (higher strike) for a Put Spread.")

        # Création des deux options
        put_K1 = Option(self.Spot, K1, self.Maturity, self.Rate, self.Div, self.Sigma)
        put_K2 = Option(self.Spot, K2, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if put_K1_price is None or put_K2_price is None:
            _, put_K1_price = put_K1.option_price()
            _, put_K2_price = put_K2.option_price()

        # Long Put K2 et Short Put K1
        if option_position == "Long":
            put_K1_position = "Short"
            put_K2_position = "Long"
            put_spread_position = "Long"
        elif option_position == "Short":
            put_K1_position = "Long"
            put_K2_position = "Short"
            put_spread_position = "Short"

        # Calculs du payoff et profit pour les options individuelles
        _, put_K1_payoff = put_K1.option_payoff(put_K1_position)
        _, put_K2_payoff = put_K2.option_payoff(put_K2_position)
        _, put_K1_profit = put_K1.option_net_profit(put_K1_position, put_price=put_K1_price)
        _, put_K2_profit = put_K2.option_net_profit(put_K2_position, put_price=put_K2_price)

        # Calculs prix, payoff et profit du Put Spread global
        put_spread_price = put_K2_price - put_K1_price  # K2 est plus cher que K1
        put_spread_payoff = put_K1_payoff + put_K2_payoff
        put_spread_profit = put_K1_profit + put_K2_profit

        # Création d'une table des résultats combinée pour les options individuelles et globales
        df_put_spread = pd.DataFrame({
            "Option": [f"{put_K1_position} Put K1", f"{put_K2_position} Put K2", f"{put_spread_position} Put Spread"],
            "Spot": [self.Spot, self.Spot, self.Spot],
            "Strike": [K1, K2, f"{K1}-{K2}"],
            "Payoff": [f"{put_K1_payoff:.2f}", f"{put_K2_payoff:.2f}", f"{put_spread_payoff:.2f}"],
            "Price": [f"{put_K1_price:.2f}", f"{put_K2_price:.2f}", f"{put_spread_price:.2f}"],
            "Profit": [f"{put_K1_profit:.2f}", f"{put_K2_profit:.2f}", f"{put_spread_profit:.2f}"]
        })

        return df_put_spread

    def plot_put_spread(self, option_position, spot, K1, K2, maturity, rate, div, sigma):
        """
        Génère un graphique interactif pour analyser les profits et payoffs d'un Put Spread.
        """
        # Création des options pour K1 et K2
        option_K1 = Option(spot, K1, maturity, rate, div, sigma)
        option_K2 = Option(spot, K2, maturity, rate, div, sigma)

        # Long Put K2 et Short Put K1
        if option_position == "Long":
            put_K1_position = "Short"
            put_K2_position = "Long"
        elif option_position == "Short":
            put_K1_position = "Long"
            put_K2_position = "Short"

        # Calcul des prix des options
        _, price_K1 = option_K1.option_price()
        _, price_K2 = option_K2.option_price()
        put_spread_price = price_K2 - price_K1  # K2 est plus cher que K1 pour un Put Spread

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les résultats pour les deux options et la stratégie
        profit_K1, profit_K2, spread_profit = [], [], []

        for s in spot_values:
            temp_K1 = Option(s, K1, maturity, rate, div, sigma)
            temp_K2 = Option(s, K2, maturity, rate, div, sigma)

            _, p_K1 = temp_K1.option_net_profit(put_K1_position, put_price=price_K1, spot=s)
            profit_K1.append(p_K1)

            _, p_K2 = temp_K2.option_net_profit(put_K2_position, put_price=price_K2, spot=s)
            profit_K2.append(p_K2)

            total_profit = p_K1 + p_K2
            spread_profit.append(total_profit)

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour K1, K2 et Put Spread
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K1, mode="lines", name=f"Profit {put_K1_position} Put K1", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K2, mode="lines", name=f"Profit {put_K2_position} Put K2", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=spread_profit, mode="lines", name=f"Profit {option_position} Put Spread", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in spread_profit]
        loss_area = [min(0, p) for p in spread_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes des strikes, breakeven et des prix
        breakeven = K2 - (price_K2 - price_K1)  # Logique correcte pour le Breakeven du Put Spread

        # Ligne pour le Spot Price
        fig.add_vline(x=spot, line_dash="dot", line_color="green", annotation_text=f"Spot: {spot}", annotation_font=dict(color="green"))

        # Ligne pour K1
        fig.add_vline(x=K1, line_dash="dash", line_color="blue", opacity=0.5,
                    annotation_text=f"K1: {K1}", annotation_position="top right", annotation_font=dict(color="blue"))

        # Ligne pour K2
        fig.add_vline(x=K2, line_dash="dash", line_color="pink", opacity=0.5,
                    annotation_text=f"K2: {K2}", annotation_position="top right", annotation_font=dict(color="pink"))

        # Ligne pour Breakevens
        fig.add_vline(x=breakeven, line_dash="dot", line_color="orange", annotation_text=f"BE: {breakeven:.2f}",
                    annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix
        fig.add_hline(y=-price_K1 if put_K1_position == "Long" else price_K1, line_dash="dot", line_color="blue",
                    annotation_text=f"{put_K1_position} Put K1 Price: {price_K1:.2f}", annotation_position="right", annotation_font=dict(color="blue"))

        fig.add_hline(y=-price_K2 if put_K2_position == "Long" else price_K2, line_dash="dot", line_color="pink",
                    annotation_text=f"{put_K2_position} Put K2 Price: {price_K2:.2f}", annotation_position="right", annotation_font=dict(color="pink"))

        fig.add_hline(y=-put_spread_price if option_position == "Long" else put_spread_price, line_dash="dot", line_color="green",
                    annotation_text=f"{option_position} Put Spread Price: {put_spread_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Put Spread Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Profit / Payoff",
            hovermode="x unified", template="plotly_dark"
        )

        return fig

    def strangle(self, option_position, K_put, K_call, put_price=None, call_price=None):
        """
        Strangle Strategy: Buy or sell a put option and a call option with different strikes.
        :param option_position: "Long" or "Short"
        :param K_put: Strike price of the put option (lower strike).
        :param K_call: Strike price of the call option (higher strike).
        :param put_price: (Optional) Prix pré-calculé pour le put.
        :param call_price: (Optional) Prix pré-calculé pour le call.
        :return: DataFrame with detailed results for the strategy.
        """
        # Validation des strikes
        if K_put >= K_call:
            raise ValueError("K_put (lower strike) must be less than K_call (higher strike) for a Strangle.")

        # Création des deux options
        put_option = Option(self.Spot, K_put, self.Maturity, self.Rate, self.Div, self.Sigma)
        call_option = Option(self.Spot, K_call, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if put_price is None or call_price is None:
            _, put_price = put_option.option_price()
            call_price, _ = call_option.option_price()

        # Définir les positions pour Long ou Short Strangle
        if option_position == "Long":
            put_position = "Long"
            call_position = "Long"
            strangle_position = "Long"
            price_sign = 1
            payoff_sign = 1
        elif option_position == "Short":
            put_position = "Short"
            call_position = "Short"
            strangle_position = "Short"
            price_sign = -1
            payoff_sign = -1

        # Calculs du payoff et profit pour les options individuelles
        _, put_payoff = put_option.option_payoff(put_position)
        call_payoff, _ = call_option.option_payoff(call_position)
        _, put_profit = put_option.option_net_profit(put_position, put_price=put_price)
        call_profit, _ = call_option.option_net_profit(call_position, call_price=call_price)

        # Calculs prix, payoff et profit du Strangle global
        strangle_price = price_sign * (put_price + call_price)
        strangle_payoff = payoff_sign * (put_payoff + call_payoff)
        strangle_profit = put_profit + call_profit

        # Création d'une table des résultats combinée pour les options individuelles et globales
        df_strangle = pd.DataFrame({
            "Option": [f"{put_position} Put", f"{call_position} Call", f"{strangle_position} Strangle"],
            "Spot": [self.Spot, self.Spot, self.Spot],
            "Strike": [K_put, K_call, f"{K_put}-{K_call}"],
            "Payoff": [f"{put_payoff:.2f}", f"{call_payoff:.2f}", f"{strangle_payoff:.2f}"],
            "Price": [f"{put_price:.2f}", f"{call_price:.2f}", f"{strangle_price:.2f}"],
            "Profit": [f"{put_profit:.2f}", f"{call_profit:.2f}", f"{strangle_profit:.2f}"]
        })

        return df_strangle

    def plot_strangle(self, option_position, spot, K_put, K_call, maturity, rate, div, sigma):
        """
        Génère un graphique interactif pour analyser les profits et payoffs d'un Strangle.
        """
        # Création des options
        put_option = Option(spot, K_put, maturity, rate, div, sigma)
        call_option = Option(spot, K_call, maturity, rate, div, sigma)

        # Définir les positions pour Long ou Short Strangle
        if option_position == "Long":
            put_position = "Long"
            call_position = "Long"
        elif option_position == "Short":
            put_position = "Short"
            call_position = "Short"

        # Calcul des prix des options
        _, put_price = put_option.option_price()
        call_price, _ = call_option.option_price()
        strangle_price = put_price + call_price

        # Calcul des breakevens
        if option_position == "Long":
            breakeven_lower = K_put - strangle_price
            breakeven_upper = K_call + strangle_price
        elif option_position == "Short":
            breakeven_lower = K_put + strangle_price
            breakeven_upper = K_call - strangle_price

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les résultats pour les deux options et la stratégie
        profit_put, profit_call, strangle_profit = [], [], []

        for s in spot_values:
            temp_put = Option(s, K_put, maturity, rate, div, sigma)
            temp_call = Option(s, K_call, maturity, rate, div, sigma)

            _, p_put = temp_put.option_net_profit(put_position, put_price=put_price, spot=s)
            profit_put.append(p_put)

            p_call, _ = temp_call.option_net_profit(call_position, call_price=call_price, spot=s)
            profit_call.append(p_call)

            total_profit = p_put + p_call
            strangle_profit.append(total_profit)

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour le put, le call et le strangle
        fig.add_trace(go.Scatter(x=spot_values, y=profit_put, mode="lines", name=f"Profit {put_position} Put", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_call, mode="lines", name=f"Profit {call_position} Call", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=strangle_profit, mode="lines", name=f"Profit {option_position} Strangle", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in strangle_profit]
        loss_area = [min(0, p) for p in strangle_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes des strikes
        fig.add_vline(x=K_put, line_dash="dash", line_color="blue", annotation_text=f"K_put: {K_put}", annotation_position="top right", annotation_font=dict(color="blue"))
        fig.add_vline(x=K_call, line_dash="dash", line_color="pink", annotation_text=f"K_call: {K_call}", annotation_position="top right", annotation_font=dict(color="pink"))

        # Ajouter les breakeven points
        fig.add_vline(x=breakeven_lower, line_dash="dot", line_color="orange",
                    annotation_text=f"BE Lower: {breakeven_lower:.2f}", annotation_position="bottom left", annotation_font=dict(color="orange"))
        fig.add_vline(x=breakeven_upper, line_dash="dot", line_color="orange",
                    annotation_text=f"BE Upper: {breakeven_upper:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix
        fig.add_hline(y=-put_price if put_position == "Long" else put_price, line_dash="dot", line_color="blue",
                    annotation_text=f"{put_position} Put Price: {put_price:.2f}", annotation_position="right", annotation_font=dict(color="blue"))

        fig.add_hline(y=-call_price if call_position == "Long" else call_price, line_dash="dot", line_color="pink",
                    annotation_text=f"{call_position} Call Price: {call_price:.2f}", annotation_position="right", annotation_font=dict(color="pink"))

        fig.add_hline(y=-strangle_price if option_position == "Long" else strangle_price, line_dash="dot", line_color="green",
                    annotation_text=f"{option_position} Strangle Price: {strangle_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Strangle Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Profit / Payoff",
            hovermode="x unified", template="plotly_dark"
        )

        return fig

    def straddle(self, option_position, strike, put_price=None, call_price=None):
        """
        Calcule les résultats d'un Straddle (Long ou Short).
        Long Straddle: Acheter un call et un put avec le même strike et la même maturité.
        Short Straddle: Vendre un call et un put avec le même strike et la même maturité.
        :param option_position: "Long" ou "Short".
        :param strike: Prix d'exercice des deux options (call et put).
        :param put_price: (Optionnel) Prix pré-calculé pour le put.
        :param call_price: (Optionnel) Prix pré-calculé pour le call.
        :return: DataFrame contenant les détails du Straddle.
        """
        # Création des options
        put_option = Option(self.Spot, strike, self.Maturity, self.Rate, self.Div, self.Sigma)
        call_option = Option(self.Spot, strike, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if put_price is None or call_price is None:
            _, put_price = put_option.option_price()
            call_price, _ = call_option.option_price()

        # Définir les positions pour Long ou Short Straddle
        if option_position == "Long":
            put_position = "Long"
            call_position = "Long"
            straddle_position = "Long"
        elif option_position == "Short":
            put_position = "Short"
            call_position = "Short"
            straddle_position = "Short"

        # Calculs des payoffs et profits pour les options individuelles
        _, put_payoff = put_option.option_payoff(put_position)
        call_payoff, _ = call_option.option_payoff(call_position)
        _, put_profit = put_option.option_net_profit(put_position, put_price=put_price)
        call_profit, _ = call_option.option_net_profit(call_position, call_price=call_price)

        # Calculs du Straddle global
        straddle_price = put_price + call_price
        straddle_payoff = put_payoff + call_payoff
        straddle_profit = put_profit + call_profit

        # Résultats sous forme de DataFrame
        df_straddle = pd.DataFrame({
            "Option": [f"{put_position} Put", f"{call_position} Call", f"{straddle_position} Straddle"],
            "Spot": [self.Spot, self.Spot, self.Spot],
            "Strike": [strike, strike, strike],
            "Payoff": [f"{put_payoff:.2f}", f"{call_payoff:.2f}", f"{straddle_payoff:.2f}"],
            "Price": [f"{put_price:.2f}", f"{call_price:.2f}", f"{straddle_price:.2f}"],
            "Profit": [f"{put_profit:.2f}", f"{call_profit:.2f}", f"{straddle_profit:.2f}"]
        })

        return df_straddle

    def plot_straddle(self, option_position, spot, strike, maturity, rate, div, sigma):
        """
        Génère un graphique interactif pour analyser les profits et payoffs d'un Straddle.
        """
        # Création des options
        put_option = Option(spot, strike, maturity, rate, div, sigma)
        call_option = Option(spot, strike, maturity, rate, div, sigma)

        # Définir les positions pour Long ou Short Straddle
        if option_position == "Long":
            put_position = "Long"
            call_position = "Long"
        elif option_position == "Short":
            put_position = "Short"
            call_position = "Short"

        # Calcul des prix des options
        _, put_price = put_option.option_price()
        call_price, _ = call_option.option_price()
        straddle_price = put_price + call_price

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les résultats pour les deux options et la stratégie
        profit_put, profit_call, straddle_profit = [], [], []

        for s in spot_values:
            temp_put = Option(s, strike, maturity, rate, div, sigma)
            temp_call = Option(s, strike, maturity, rate, div, sigma)

            _, p_put = temp_put.option_net_profit(put_position, put_price=put_price, spot=s)
            profit_put.append(p_put)

            p_call, _ = temp_call.option_net_profit(call_position, call_price=call_price, spot=s)
            profit_call.append(p_call)

            total_profit = p_put + p_call
            straddle_profit.append(total_profit)

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour le put, le call et le straddle
        fig.add_trace(go.Scatter(x=spot_values, y=profit_put, mode="lines", name=f"Profit {put_position} Put", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_call, mode="lines", name=f"Profit {call_position} Call", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=straddle_profit, mode="lines", name=f"Profit {option_position} Straddle", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in straddle_profit]
        loss_area = [min(0, p) for p in straddle_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes du strike
        fig.add_vline(x=strike, line_dash="dash", line_color="purple",
                    annotation_text=f"Strike: {strike}", annotation_position="top right", annotation_font=dict(color="purple"))

        # Ajouter les breakeven points
        #if option_position == "Long":
        breakeven_lower = strike - straddle_price
        breakeven_upper = strike + straddle_price
        #elif option_position == "Short":
            #breakeven_upper = strike + straddle_price
            #breakeven_lower = strike - straddle_price

        fig.add_vline(x=breakeven_lower, line_dash="dot", line_color="orange",
                    annotation_text=f"BE Lower: {breakeven_lower:.2f}", annotation_position="bottom left", annotation_font=dict(color="orange"))
        fig.add_vline(x=breakeven_upper, line_dash="dot", line_color="orange",
                    annotation_text=f"BE Upper: {breakeven_upper:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix
        fig.add_hline(y=-put_price if put_position == "Long" else put_price, line_dash="dot", line_color="blue",
                    annotation_text=f"{put_position} Put Price: {put_price:.2f}", annotation_position="right", annotation_font=dict(color="blue"))

        fig.add_hline(y=-call_price if call_position == "Long" else call_price, line_dash="dot", line_color="pink",
                    annotation_text=f"{call_position} Call Price: {call_price:.2f}", annotation_position="right", annotation_font=dict(color="pink"))

        fig.add_hline(y=-straddle_price if option_position == "Long" else straddle_price, line_dash="dot", line_color="green",
                    annotation_text=f"{option_position} Straddle Price: {straddle_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Straddle Payoff & Profit Analysis",
            xaxis_title="Spot Price", yaxis_title="Profit / Payoff",
            hovermode="x unified", template="plotly_dark"
        )

        return fig

    def call_butterfly(self, option_position, K1, K2, K3, call_K1_price=None, call_K2_price=None, call_K3_price=None):
        """
        Long Call Butterfly Strategy: Buy a call option (K1), sell two call options (K2), and buy a call option (K3).
        Short Call Butterfly Strategy: Sell a call option (K1), buy two call options (K2), and sell a call option (K3).
        """
        # Validation des strikes
        if not (K1 < K2 < K3):
            raise ValueError("Strike prices must satisfy K1 < K2 < K3 for a Call Butterfly.")

        # Création des options
        call_K1 = Option(self.Spot, K1, self.Maturity, self.Rate, self.Div, self.Sigma)
        call_K2 = Option(self.Spot, K2, self.Maturity, self.Rate, self.Div, self.Sigma)
        call_K3 = Option(self.Spot, K3, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if call_K1_price is None or call_K2_price is None or call_K3_price is None:
            call_K1_price, _ = call_K1.option_price()
            call_K2_price, _ = call_K2.option_price()
            call_K3_price, _ = call_K3.option_price()

        # Définir les facteurs pour Long ou Short Butterfly
        if option_position == "Long":
            positions = ["Long", "Short", "Long"]
        elif option_position == "Short":
            positions = ["Short", "Long", "Short"]

        factors = [1, 2, 1]

        # Calcul des payoffs et profits pour chaque option
        call_K1_payoff = call_K1.option_payoff(option_position=positions[0])[0] * factors[0]
        call_K2_payoff = call_K2.option_payoff(option_position=positions[1])[0] * factors[1]
        call_K3_payoff = call_K3.option_payoff(option_position=positions[2])[0] * factors[2]

        call_K1_profit = call_K1.option_net_profit(option_position=positions[0], call_price=call_K1_price)[0] * factors[0]
        call_K2_profit = call_K2.option_net_profit(option_position=positions[1], call_price=call_K2_price)[0] * factors[1]
        call_K3_profit = call_K3.option_net_profit(option_position=positions[2], call_price=call_K3_price)[0] * factors[2]

        call_K1_price = call_K1_price * factors[0]
        call_K2_price = call_K2_price * factors[1]
        call_K3_price = call_K3_price * factors[2]

        # Résultats globaux
        butterfly_price = call_K1_price - call_K2_price + call_K3_price
        butterfly_payoff = call_K1_payoff + call_K2_payoff + call_K3_payoff
        butterfly_profit = call_K1_profit + call_K2_profit + call_K3_profit

        # DataFrame de résultats
        df_call_butterfly = pd.DataFrame({
            "Option": [f"{positions[0]} Call K1", f"2 * {positions[1]} Call K2", f"{positions[2]} Call K3", f"{option_position} Call Butterfly"],
            "Spot": [self.Spot, self.Spot, self.Spot, self.Spot],
            "Strike": [K1, K2, K3, f"{K1}-{K2}-{K3}"],
            "Payoff": [f"{call_K1_payoff:.2f}", f"{call_K2_payoff:.2f}", f"{call_K3_payoff:.2f}", f"{butterfly_payoff:.2f}"],
            "Price": [f"{call_K1_price:.2f}", f"{call_K2_price:.2f}", f"{call_K3_price:.2f}", f"{butterfly_price:.2f}"],
            "Profit": [f"{call_K1_profit:.2f}", f"{call_K2_profit:.2f}", f"{call_K3_profit:.2f}", f"{butterfly_profit:.2f}"]
        })

        return df_call_butterfly

    def plot_call_butterfly(self, option_position, spot, K1, K2, K3, maturity, rate, div, sigma):
        """
        Génère un graphique interactif pour analyser les profits et payoffs d'un Call Butterfly.
        """
        # Création des options
        option_K1 = Option(spot, K1, maturity, rate, div, sigma)
        option_K2 = Option(spot, K2, maturity, rate, div, sigma)
        option_K3 = Option(spot, K3, maturity, rate, div, sigma)

        # Définir les positions pour Long ou Short Butterfly
        if option_position == "Long":
            positions = ["Long", "Short", "Long"]
        elif option_position == "Short":
            positions = ["Short", "Long", "Short"]

        factors = [1, 2, 1]

        # Calcul des prix des options
        price_K1, _ = option_K1.option_price()
        price_K2, _ = option_K2.option_price()
        price_K3, _ = option_K3.option_price()

        # Prix total du Call Butterfly
        butterfly_price = price_K1 - factors[1] * price_K2 + price_K3

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les profits pour les options et la stratégie
        profit_K1, profit_K2, profit_K3, butterfly_profit = [], [], [], []

        for s in spot_values:
            temp_K1 = Option(s, K1, maturity, rate, div, sigma)
            temp_K2 = Option(s, K2, maturity, rate, div, sigma)
            temp_K3 = Option(s, K3, maturity, rate, div, sigma)

            # Calculer les profits pour chaque option
            p_K1 = temp_K1.option_net_profit(option_position=positions[0], call_price=price_K1, spot=s)[0] * factors[0]
            p_K2 = temp_K2.option_net_profit(option_position=positions[1], call_price=price_K2, spot=s)[0] * factors[1]
            p_K3 = temp_K3.option_net_profit(option_position=positions[2], call_price=price_K3, spot=s)[0] * factors[2]

            # Calculer le profit total
            total_profit = p_K1 + p_K2 + p_K3

            # Ajouter aux listes
            profit_K1.append(p_K1)
            profit_K2.append(p_K2)
            profit_K3.append(p_K3)
            butterfly_profit.append(total_profit)

        # Calcul des breakevens
        breakeven_low = K1 + butterfly_price
        breakeven_high = K3 - butterfly_price

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour K1, K2, K3 et Butterfly
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K1, mode="lines", name=f"Profit {positions[0]} Call K1", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K2, mode="lines", name=f"Profit 2 * {positions[1]} Call K2", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K3, mode="lines", name=f"Profit {positions[2]} Call K3", line=dict(color="purple", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=butterfly_profit, mode="lines", name=f"Profit {option_position} Call Butterfly", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in butterfly_profit]
        loss_area = [min(0, p) for p in butterfly_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes des strikes et des breakevens
        fig.add_vline(x=K1, line_dash="dash", line_color="blue", annotation_text=f"K1: {K1}", annotation_position="top right", annotation_font=dict(color="blue"))
        fig.add_vline(x=K2, line_dash="dash", line_color="pink", annotation_text=f"K2: {K2}", annotation_position="top right", annotation_font=dict(color="pink"))
        fig.add_vline(x=K3, line_dash="dash", line_color="purple", annotation_text=f"K3: {K3}", annotation_position="top right", annotation_font=dict(color="purple"))
        fig.add_vline(x=breakeven_low, line_dash="dot", line_color="orange", annotation_text=f"BE Low: {breakeven_low:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))
        fig.add_vline(x=breakeven_high, line_dash="dot", line_color="orange", annotation_text=f"BE High: {breakeven_high:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix des options et le Butterfly
        fig.add_hline(y=-price_K1 if positions[0] == "Long" else price_K1, line_dash="dot", line_color="blue",
                    annotation_text=f"{positions[0]} Call K1 Price: {price_K1:.2f}", annotation_position="right", annotation_font=dict(color="blue"))
        fig.add_hline(y=-price_K2*factors[1] if positions[1] == "Long" else price_K2*factors[1], line_dash="dot", line_color="pink",
                    annotation_text=f"{positions[1]} 2 * Call K2 Price: {price_K2:.2f}", annotation_position="right", annotation_font=dict(color="pink"))
        fig.add_hline(y=-price_K3 if positions[2] == "Long" else price_K3, line_dash="dot", line_color="purple",
                    annotation_text=f"{positions[2]} Call K3 Price: {price_K3:.2f}", annotation_position="right", annotation_font=dict(color="purple"))
        fig.add_hline(y=-butterfly_price if option_position == "Long" else butterfly_price, line_dash="dot", line_color="green",
                    annotation_text=f"{option_position} Butterfly Price: {butterfly_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Call Butterfly Payoff & Profit Analysis",
            xaxis_title="Spot Price",
            yaxis_title="Profit / Payoff",
            hovermode="x unified",
            template="plotly_dark"
        )

        return fig

    def put_butterfly(self, option_position, K1, K2, K3, put_K1_price=None, put_K2_price=None, put_K3_price=None):
        """
        Long Put Butterfly Strategy: Buy a put option (K1), sell two put options (K2), and buy a put option (K3).
        Short Put Butterfly Strategy: Sell a put option (K1), buy two put options (K2), and sell a put option (K3).
        """
        # Validation des strikes
        if not (K1 < K2 < K3):
            raise ValueError("Strike prices must satisfy K1 < K2 < K3 for a Put Butterfly.")

        # Création des options
        put_K1 = Option(self.Spot, K1, self.Maturity, self.Rate, self.Div, self.Sigma)
        put_K2 = Option(self.Spot, K2, self.Maturity, self.Rate, self.Div, self.Sigma)
        put_K3 = Option(self.Spot, K3, self.Maturity, self.Rate, self.Div, self.Sigma)

        # Calcul des prix si non fournis
        if put_K1_price is None or put_K2_price is None or put_K3_price is None:
            _, put_K1_price = put_K1.option_price()
            _, put_K2_price = put_K2.option_price()
            _, put_K3_price = put_K3.option_price()

        # Définir les facteurs pour Long ou Short Butterfly
        if option_position == "Long":
            positions = ["Long", "Short", "Long"]
        elif option_position == "Short":
            positions = ["Short", "Long", "Short"]

        factors = [1, 2, 1]

        # Calcul des payoffs et profits pour chaque option
        put_K1_payoff = put_K1.option_payoff(option_position=positions[0])[1] * factors[0]
        put_K2_payoff = put_K2.option_payoff(option_position=positions[1])[1] * factors[1]
        put_K3_payoff = put_K3.option_payoff(option_position=positions[2])[1] * factors[2]

        put_K1_profit = put_K1.option_net_profit(option_position=positions[0], put_price=put_K1_price)[1] * factors[0]
        put_K2_profit = put_K2.option_net_profit(option_position=positions[1], put_price=put_K2_price)[1] * factors[1]
        put_K3_profit = put_K3.option_net_profit(option_position=positions[2], put_price=put_K3_price)[1] * factors[2]

        put_K1_price = put_K1_price * factors[0]
        put_K2_price = put_K2_price * factors[1]
        put_K3_price = put_K3_price * factors[2]

        # Résultats globaux
        butterfly_price = put_K1_price - put_K2_price + put_K3_price
        butterfly_payoff = put_K1_payoff + put_K2_payoff + put_K3_payoff
        butterfly_profit = put_K1_profit + put_K2_profit + put_K3_profit

        # DataFrame de résultats
        df_put_butterfly = pd.DataFrame({
            "Option": [f"{positions[0]} Put K1", f"2 * {positions[1]} Put K2", f"{positions[2]} Put K3", f"{option_position} Put Butterfly"],
            "Spot": [self.Spot, self.Spot, self.Spot, self.Spot],
            "Strike": [K1, K2, K3, f"{K1}-{K2}-{K3}"],
            "Payoff": [f"{put_K1_payoff:.2f}", f"{put_K2_payoff:.2f}", f"{put_K3_payoff:.2f}", f"{butterfly_payoff:.2f}"],
            "Price": [f"{put_K1_price:.2f}", f"{put_K2_price:.2f}", f"{put_K3_price:.2f}", f"{butterfly_price:.2f}"],
            "Profit": [f"{put_K1_profit:.2f}", f"{put_K2_profit:.2f}", f"{put_K3_profit:.2f}", f"{butterfly_profit:.2f}"]
        })

        return df_put_butterfly

    def plot_put_butterfly(self, option_position, spot, K1, K2, K3, maturity, rate, div, sigma):
        """
        Génère un graphique interactif pour analyser les profits et payoffs d'un Put Butterfly.
        """
        # Création des options
        option_K1 = Option(spot, K1, maturity, rate, div, sigma)
        option_K2 = Option(spot, K2, maturity, rate, div, sigma)
        option_K3 = Option(spot, K3, maturity, rate, div, sigma)

        # Définir les positions pour Long ou Short Butterfly
        if option_position == "Long":
            positions = ["Long", "Short", "Long"]
        elif option_position == "Short":
            positions = ["Short", "Long", "Short"]

        factors = [1, 2, 1]

        # Calcul des prix des options
        _, price_K1 = option_K1.option_price()
        _, price_K2 = option_K2.option_price()
        _, price_K3 = option_K3.option_price()

        # Prix total du Put Butterfly
        butterfly_price = price_K1 - factors[1] * price_K2 + price_K3

        # Définir les valeurs de spot pour les graphiques
        spot_values = np.linspace(spot * 0.5, spot * 1.5, 1000)

        # Calculer les profits pour les options et la stratégie
        profit_K1, profit_K2, profit_K3, butterfly_profit = [], [], [], []

        for s in spot_values:
            temp_K1 = Option(s, K1, maturity, rate, div, sigma)
            temp_K2 = Option(s, K2, maturity, rate, div, sigma)
            temp_K3 = Option(s, K3, maturity, rate, div, sigma)

            # Calculer les profits pour chaque option
            p_K1 = temp_K1.option_net_profit(option_position=positions[0], put_price=price_K1, spot=s)[1] * factors[0]
            p_K2 = temp_K2.option_net_profit(option_position=positions[1], put_price=price_K2, spot=s)[1] * factors[1]
            p_K3 = temp_K3.option_net_profit(option_position=positions[2], put_price=price_K3, spot=s)[1] * factors[2]

            # Calculer le profit total
            total_profit = p_K1 + p_K2 + p_K3

            # Ajouter aux listes
            profit_K1.append(p_K1)
            profit_K2.append(p_K2)
            profit_K3.append(p_K3)
            butterfly_profit.append(total_profit)

        # Calcul des breakevens
        breakeven_low = K1 + butterfly_price
        breakeven_high = K3 - butterfly_price

        # Création du graphique interactif
        fig = go.Figure()

        # Ajouter les courbes de profit pour K1, K2, K3 et Butterfly
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K1, mode="lines", name=f"Profit {positions[0]} Put K1", line=dict(color="blue", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K2, mode="lines", name=f"Profit 2 * {positions[1]} Put K2", line=dict(color="pink", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=profit_K3, mode="lines", name=f"Profit {positions[2]} Put K3", line=dict(color="purple", width=1)))
        fig.add_trace(go.Scatter(x=spot_values, y=butterfly_profit, mode="lines", name=f"Profit {option_position} Put Butterfly", line=dict(color="green", width=5)))

        # Ajouter les zones de profit et de perte
        profit_area = [max(0, p) for p in butterfly_profit]
        loss_area = [min(0, p) for p in butterfly_profit]
        fig.add_trace(go.Scatter(x=spot_values, y=profit_area, fill="tozeroy", mode="none", fillcolor="rgba(0, 255, 0, 0.2)", name="Profit Area"))
        fig.add_trace(go.Scatter(x=spot_values, y=loss_area, fill="tozeroy", mode="none", fillcolor="rgba(255, 0, 0, 0.2)", name="Loss Area"))

        # Ajouter les lignes des strikes et des breakevens
        fig.add_vline(x=K1, line_dash="dash", line_color="blue", annotation_text=f"K1: {K1}", annotation_position="top right", annotation_font=dict(color="blue"))
        fig.add_vline(x=K2, line_dash="dash", line_color="pink", annotation_text=f"K2: {K2}", annotation_position="top right", annotation_font=dict(color="pink"))
        fig.add_vline(x=K3, line_dash="dash", line_color="purple", annotation_text=f"K3: {K3}", annotation_position="top right", annotation_font=dict(color="purple"))
        fig.add_vline(x=breakeven_low, line_dash="dot", line_color="orange", annotation_text=f"BE Low: {breakeven_low:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))
        fig.add_vline(x=breakeven_high, line_dash="dot", line_color="orange", annotation_text=f"BE High: {breakeven_high:.2f}", annotation_position="bottom right", annotation_font=dict(color="orange"))

        # Lignes horizontales pour les prix des options et le Butterfly
        fig.add_hline(y=-price_K1 if positions[0] == "Long" else price_K1, line_dash="dot", line_color="blue",
                        annotation_text=f"{positions[0]} Put K1 Price: {price_K1:.2f}", annotation_position="right", annotation_font=dict(color="blue"))
        fig.add_hline(y=-price_K2*factors[1] if positions[1] == "Long" else price_K2*factors[1], line_dash="dot", line_color="pink",
                        annotation_text=f"{positions[1]} 2 * Put K2 Price: {price_K2:.2f}", annotation_position="right", annotation_font=dict(color="pink"))
        fig.add_hline(y=-price_K3 if positions[2] == "Long" else price_K3, line_dash="dot", line_color="purple",
                        annotation_text=f"{positions[2]} Put K3 Price: {price_K3:.2f}", annotation_position="right", annotation_font=dict(color="purple"))
        fig.add_hline(y=-butterfly_price if option_position == "Long" else butterfly_price, line_dash="dot", line_color="green",
                        annotation_text=f"{option_position} Butterfly Price: {butterfly_price:.2f}", annotation_position="right", annotation_font=dict(color="green"))

        # Ligne horizontale à y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=3)

        # Configurer le graphique
        fig.update_layout(
            title=f"{option_position} Put Butterfly Payoff & Profit Analysis",
            xaxis_title="Spot Price",
            yaxis_title="Profit / Payoff",
            hovermode="x unified",
            template="plotly_dark"
        )

        return fig



# ------------------------------
# CREATE STREAMLIT INTERFACE
# ------------------------------

# Création des onglets
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Pricing", "Greeks", "Volatility", "Monte Carlo", "Hedging", "PnL", "Option Strategy"])

# -----------------------------------
# Inputs globaux (toujours visibles)
# -----------------------------------
with st.sidebar:
    # Section Contact dans un bloc coloré
    st.markdown("""
    <div style="background-color: #f0f4f7; padding: 6px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #000000; text-align: center; margin: 0 0 0px 0;">Contact Me</h3>
        <p style="color: #000000; margin: 0px; line-height: 1.5;"><strong>Raphael EL-BAZE</strong></p>
        <p style="color: #000000; margin: 0px; line-height: 1.5;">📧 <a href="mailto:raphael.elbaze.pro@gmail.com" style="color: #0072b1;">raphael.elbaze.pro@gmail.com</a></p>
        <p style="color: #000000; margin: 0px; line-height: 1.5;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" alt="LinkedIn" style="width: 16px; vertical-align: middle;"> 
            <a href="https://www.linkedin.com/in/raphael-el-baze/" target="_blank" style="color: #0072b1;">https://www.linkedin.com/in/raphael-el-baze/</a>
        </p>
        <p style="color: #000000; margin: 0px; line-height: 1.8;">📞 +33 (0)6 68 86 91 30</p>
    </div>
    """, unsafe_allow_html=True)

    # Trait de séparation
    st.markdown("<hr>", unsafe_allow_html=True)

# Message pour les inputs à gauche
st.sidebar.markdown("""
<div style="background-color: #eaf4f4; padding: 10px; border: 1px solid #ccc; border-radius: 5px; color: #333; font-size: 14px;">
    <strong>Note:</strong> These inputs affect all tabs except <strong>Option Strategy</strong>.
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Global Inputs")
input_mode = st.sidebar.radio("Choose Mode", ["Manual Inputs", "Fetch Data from API (yfinance)"], key="input_mode")

if input_mode == "Manual Inputs":
    #st.info(f"You are currently using the Manual Inputs mode.") 
    option_position = st.sidebar.selectbox(
    "Choose Option Position", 
    ["Long", "Short"], 
    index=0, 
    key="global_option_position", 
    help="Position type: 'Long' means buying the option, while 'Short' means selling it."
    )
    Spot = st.sidebar.number_input(
        "Spot Price (S)", 
        min_value=1.0, 
        value=100.0, 
        step=1.0, 
        key="spot_manual", 
        help="The current price of the underlying asset in the market."
    )
    Strike = st.sidebar.number_input(
        "Strike Price (K)", 
        min_value=1.0, 
        value=100.0, 
        step=1.0, 
        key="strike_manual", 
        help="The agreed price at which the underlying asset can be bought or sold."
    )
    Maturity = st.sidebar.number_input(
        "Time to Maturity (years)", 
        min_value=0.01, 
        value=1.0, 
        step=0.01, 
        help="The time remaining until the option expires, expressed in years."
    )
    Rate = st.sidebar.number_input(
        "Risk-Free Rate (r)", 
        min_value=0.0, 
        value=0.05, 
        step=0.01, 
        help="The annualized risk-free interest rate, often based on government bonds."
    )
    Div = st.sidebar.number_input(
        "Dividend Yield (q)", 
        min_value=0.0, 
        value=0.02, 
        step=0.01, 
        help="The annualized dividend yield of the underlying asset."
    )
    Sigma = st.sidebar.number_input(
        "Volatility (σ)", 
        min_value=0.01, 
        value=0.2, 
        step=0.01, 
        help="The expected volatility of the underlying asset, expressed as a percentage."
    )
else:
    #st.info(f"You are currently using the API mode.") 
    st.sidebar.header("Fetch Market Data")
    option_position = st.sidebar.selectbox(
        "Choose Option Position", 
        ["Long", "Short"], 
        index=0, 
        help="Position type: 'Long' means buying the option, while 'Short' means selling it."
    )
    popular_tickers = ["AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]
    selected_ticker = st.sidebar.selectbox(
        "Choose a Stock Ticker", 
        popular_tickers, 
        help="Select the ticker symbol of the stock you want to analyze."
    )

    # Fetch Data from API (yfinance)
    yf_data = YFinanceData(selected_ticker)
    Spot = yf_data.get_spot_price()
    st.sidebar.number_input(
        "Spot Price (S)", 
        value=Spot, 
        step=0.01, 
        help="The current price of the underlying asset in the market, retrieved from the API."
    )

    expirations = yf_data.get_expiration_dates()
    date_expiration = st.sidebar.selectbox(
        "Expiration Date", 
        expirations, 
        help="Available expiration dates for the selected option, based on market data."
    )

    # Calcule la maturité automatiquement
    calculated_maturity = (pd.to_datetime(date_expiration) - pd.Timestamp.now()).days / 365

    # Ajuste la maturité si elle est inférieure au minimum accepté
    if calculated_maturity < 0.01:
        st.sidebar.warning("The selected expiration date is too close. The maturity has been set to a minimum of 0.01 years to ensure stability in calculations.")
        calculated_maturity = 0.01

    # L'utilisateur peut ajuster la maturité si nécessaire
    Maturity = st.sidebar.number_input(
        "Time to Maturity (years)",
        min_value=0.01,  # Minimum d'un jour
        value=calculated_maturity,  # Valeur préremplie
        step=0.01,
        help="The time remaining until the option expires, calculated automatically from the selected expiration date."
    )

    if date_expiration:
        # Obtenir la chaîne d'options
        option_chain = yf_data.get_option_chain(date_expiration)
        # Liste des strikes disponibles
        strikes = option_chain.calls['strike'].tolist()
        # Trouver l'indice du strike le plus proche du spot
        closest_strike_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - Spot))
        # Utiliser cet indice pour définir le strike par défaut
        Strike = st.sidebar.selectbox(
            "Strike Price", 
            strikes, 
            index=closest_strike_index,  # Définit l'indice du strike le plus proche comme valeur par défaut
            help="Available strike prices for the selected expiration date, retrieved from the option chain."
        )

    Rate = st.sidebar.number_input(
        "Risk-Free Rate (r)", 
        min_value=0.0, 
        value=0.05, 
        step=0.01, 
        help="The annualized risk-free interest rate, often based on government bonds."
    )
    Div = st.sidebar.number_input(
        "Dividend Yield (q)", 
        min_value=0.0, 
        value=0.02, 
        step=0.01, 
        help="The annualized dividend yield of the underlying asset."
    )
    Sigma = st.sidebar.number_input(
        "Volatility (σ)", 
        min_value=0.01, 
        value=0.2, 
        step=0.01, 
        help="The expected volatility of the underlying asset, expressed as a percentage."
    )

    # Get last call and put prices from API
    mkt_call_price = option_chain.calls.loc[option_chain.calls['strike'] == Strike, 'lastPrice'].values
    mkt_put_price = option_chain.puts.loc[option_chain.puts['strike'] == Strike, 'lastPrice'].values
    mkt_call_price = mkt_call_price[0] if len(mkt_call_price) > 0 else None
    mkt_put_price = mkt_put_price[0] if len(mkt_put_price) > 0 else None

# Create Option
option = Option(Spot, Strike, Maturity, Rate, Div, Sigma)
call_price, put_price = option.option_price()

# ------------------------------
# Tab1: Pricing
# ------------------------------
with tab1:
    # Titre principal
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
            Black-Scholes Option Pricer for European Options
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Ajout d'une définition pliable des options de base
    with st.expander("What are Call, Put, and Position Types?"):
        st.markdown("""
        ### Option Types
        - **Call Option**: Gives the holder the right (but not the obligation) to buy the underlying asset at the strike price.
        - **Put Option**: Gives the holder the right (but not the obligation) to sell the underlying asset at the strike price.

        ### Exercise Styles
        - **American Option**: Can be exercised at any time before or at expiration.
        - **European Option**: Can only be exercised at expiration.

        ### Position Types
        - **Long Call**: Buying a call option, betting on the price of the underlying asset increasing.
        - **Short Call**: Selling a call option, betting on the price of the underlying asset decreasing or staying the same.
        - **Long Put**: Buying a put option, betting on the price of the underlying asset decreasing.
        - **Short Put**: Selling a put option, betting on the price of the underlying asset increasing or staying the same.
        """)

    # Ajout d'une définition pliable du modèle de Black-Scholes
    with st.expander("What is the Black-Scholes Model?"):
        st.markdown("""
        The Black-Scholes model is used to calculate the theoretical price of European-style options. 
        It assumes the following:
        - The underlying asset follows a lognormal distribution with constant volatility.
        - The risk-free rate and volatility remain constant over the life of the option.
        - There are no transaction costs or taxes.
        - Dividends are continuously paid at a constant rate (q).
        - The options can only be exercised at expiration.

        The key formula components include:
        - Spot Price (S): The current price of the underlying asset.
        - Strike Price (K): The agreed price at which the underlying asset can be bought or sold.
        - Risk-Free Rate (r): The risk-free interest rate.
        - Time to Maturity (T): The time remaining until the option expires.
        - Volatility (σ): The standard deviation of the asset's returns.
        """)

    def display_price_card(color, title, subtitle, price, option_type=None):
        """
        Affiche une carte HTML pour un prix.
        Si le prix n'existe pas, affiche un message personnalisé.
        """
        if isinstance(price, (int, float)):
            formatted_price = f"${price:.2f}"
        else:
            formatted_price = f"The {option_type.lower()} does not exist for this strike"
        
        return f"""
        <div style="background-color:{color}; padding:10px; border-radius:10px; text-align:center;">
            <h3 style="color:black; margin:5px 0;">{title}</h3>
            <p style="color:black; margin:0; font-size:14px;">({subtitle})</p>
            <h1 style="color:black; font-size:22px; margin:5px 0;">{formatted_price}</h1>
        </div>
        """
    
    # Create Option
    # option = Option(Spot, Strike, Maturity, Rate, Div, Sigma)
    # call_price, put_price = option.option_price()

    if input_mode == "Manual Inputs":
        # Display Theoretical Prices
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                display_price_card("#90EE90", "Theoretical Call Price", "Black-Scholes Model", call_price),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                display_price_card("#FFC0CB", "Theoretical Put Price", "Black-Scholes Model", put_price),
                unsafe_allow_html=True
            )

    else:
        # Display Prices
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                display_price_card("#90EE90", "Theoretical Call Price", "Black-Scholes Model", call_price),
                unsafe_allow_html=True
            )
            st.markdown(
                display_price_card("#ADD8E6", "Market Call Price", "From API", mkt_call_price, option_type="Call"),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                display_price_card("#FFC0CB", "Theoretical Put Price", "Black-Scholes Model", put_price),
                unsafe_allow_html=True
            )
            st.markdown(
                display_price_card("#FFDEAD", "Market Put Price", "From API", mkt_put_price, option_type="Put"),
                unsafe_allow_html=True
            )

    # Spacer for better layout
    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # Calculate Payoff and Net Profit
    call_payoff, put_payoff = option.option_payoff(option_position)
    call_profit, put_profit = option.option_net_profit(option_position, call_price=call_price, put_price=put_price)

    # Display Payoff & Profit Table
    data = {
        "Option position": [f"{option_position} Call", f"{option_position} Put"],
        "Spot": [Spot, Spot],
        "Strike": [Strike, Strike],
        "Payoff": [f"{call_payoff:.2f}", f"{put_payoff:.2f}"],
        "Price": [f"{call_price:.2f}", f"{put_price:.2f}"],
        "Profit": [f"{call_profit:.2f}", f"{put_profit:.2f}"]
    }

    st.subheader("Payoff & Profit Table")

    # Ajout d'une définition pliable de payoff et profit
    with st.expander("What are payoff and profit?"):
        st.markdown("""
        ### Payoff Definitions
        - **Long Call**: The payoff is `max(Spot - Strike, 0)`. The holder exercises the option only if the spot price is higher than the strike price, as they would profit by buying the underlying asset at a lower price than the market price.
        - **Short Call**: The payoff is `-max(Spot - Strike, 0)`. The writer incurs a loss if the spot price exceeds the strike price, as they are obligated to sell the asset at a lower price than the market price.
        - **Long Put**: The payoff is `max(Strike - Spot, 0)`. The holder exercises the option only if the spot price is lower than the strike price, as they would profit by selling the asset at a higher price than the market price.
        - **Short Put**: The payoff is `-max(Strike - Spot, 0)`. The writer incurs a loss if the spot price is below the strike price, as they are obligated to buy the asset at a higher price than the market price.

        ### Why Can Payoff Be Zero?
        A payoff can be zero when the option holder chooses not to exercise the option. For example:
        - **Long Call**: If the spot price is below the strike price, the holder would not exercise, as buying at the strike price would cost more than the market price.
        - **Long Put**: If the spot price is above the strike price, the holder would not exercise, as selling at the strike price would bring in less than the market price.

        ### Profit Definition
        - **Profit**: Calculated as `Payoff - Option Price`. It reflects the net return after accounting for the initial cost of entering the option position.
        """)
    
    # Afficher le dataframe
    st.dataframe(pd.DataFrame(data))

    # Spacer for better layout
    st.markdown("<div style='margin-bottom:40px;'></div>", unsafe_allow_html=True)

    # Display Payoff & Profit Graphs
    spot_values = np.linspace(min(Spot, Strike) * 0.5, max(Spot, Strike) * 1.5, 1000)
    fig_call = option.plot_call(option_position, spot_values, call_price=call_price)
    fig_put = option.plot_put(option_position, spot_values, put_price=put_price)

    st.subheader("Payoff & Profit Graphs")

    # Ajouter un encart dépliant pour les graphiques de Payoff et Profit
    with st.expander("Payoff & Profit Graphs"):
        st.subheader("Visual Representation of Payoff and Profit")
        st.markdown("""
        These graphs show how the payoff and profit vary with changes in the spot price of the underlying asset:
        - **Payoff**: Indicates the value of the option at expiration based on the spot price.
        - **Profit**: Represents the net return from the option after accounting for its cost.
        """)
    
    # Graphiques de Payoff et Profit
    spot_values = np.linspace(min(Spot, Strike) * 0.5, max(Spot, Strike) * 1.5, 1000)
    fig_call = option.plot_call(option_position, spot_values, call_price=call_price)
    fig_put = option.plot_put(option_position, spot_values, put_price=put_price)

    st.plotly_chart(fig_call, use_container_width=True)
    st.plotly_chart(fig_put, use_container_width=True)

# ------------------------------
# Tab2: Greeks
# ------------------------------
with tab2:
    st.markdown(
    """
    <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
        Option Greeks
    </h3>
    """,
    unsafe_allow_html=True,
    )   

   # Ajouter un menu dépliant pour les définitions et utilités des Greeks
    with st.expander("What are Greeks and why are they useful?"):
        st.markdown("""
        ### What are Greeks?
        Greeks are sensitivities that measure how the option's price reacts to various factors. These are crucial for understanding and managing the risks associated with options.

        ### Why are Greeks useful?
        - **Delta**: Helps to hedge portfolios by measuring sensitivity to price changes of the underlying asset.
        - **Gamma**: Indicates the stability of Delta and the risk of large price movements.
        - **Theta**: Measures the impact of time decay on the option's value.
        - **Vega**: Shows the sensitivity to volatility changes of the underlying asset.
        - **Rho**: Indicates the sensitivity to changes in the risk-free interest rate.

        ### Definitions of Each Greek:
        - **Delta**: Measures the change in the option's price for a 1-unit change in the underlying asset's price.
          - Call options: Delta is positive (between 0 and 1).
          - Put options: Delta is negative (between -1 and 0).
        - **Gamma**: Measures the rate of change of Delta with respect to the underlying asset's price. Gamma is highest near the strike price.
        - **Theta**: Represents the rate of time decay of the option's price. It is usually negative because options lose value as expiration approaches.
        - **Vega**: Measures the sensitivity of the option's price to a 1% change in the volatility of the underlying asset.
        - **Rho**: Measures the change in the option's price for a 1% change in the risk-free interest rate.
        """)

    if all(v is not None for v in [Spot, Strike, Maturity, Rate, Div, Sigma]):
        # Greeks pour le mode "Manual Inputs"
        greeks = option.option_greeks(option_position)

        # Table des Greeks
        st.subheader("Greeks Table (with B&S Model)")

        greeks_df = pd.DataFrame({
            "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
            "Call": [
                f"{greeks['Delta Call']:.2f}",
                f"{greeks['Gamma']:.2f}",
                f"{greeks['Theta Call']:.2f}",
                f"{greeks['Vega']:.2f}",
                f"{greeks['Rho Call']:.2f}"
            ],
            "Put": [
                f"{greeks['Delta Put']:.2f}",
                f"{greeks['Gamma']:.2f}",
                f"{greeks['Theta Put']:.2f}",
                f"{greeks['Vega']:.2f}",
                f"{greeks['Rho Put']:.2f}"
            ]
        })

        # Afficher le DataFrame
        st.dataframe(greeks_df)

        # Calcul et affichage des Greeks pour le mode "Fetch Data from API"
        if input_mode == "Fetch Data from API":
            st.subheader("Market Greeks (From API Data)")
            market_option = Option(Spot, Strike, Maturity, Rate, Div, Sigma)
            market_greeks = market_option.option_greeks(option_position)


            # Créer un DataFrame pour les Greeks du marché
            market_greeks_df = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                "Call (Market)": [
                    f"{market_greeks['Delta Call']:.2f}",
                    f"{market_greeks['Gamma']:.2f}",
                    f"{market_greeks['Theta Call']:.2f}",
                    f"{market_greeks['Vega']:.2f}",
                    f"{market_greeks['Rho Call']:.2f}"
                ],
                "Put (Market)": [
                    f"{market_greeks['Delta Put']:.2f}",
                    f"{market_greeks['Gamma']:.2f}",
                    f"{market_greeks['Theta Put']:.2f}",
                    f"{market_greeks['Vega']:.2f}",
                    f"{market_greeks['Rho Put']:.2f}"
                ]
            })

            # Afficher le DataFrame des Greeks du marché
            st.dataframe(market_greeks_df)

        if st.button("Plot Greeks", key="plot_greeks"):
            spot_values = np.linspace(Spot * 0.5, Spot * 1.5, 100)

            # Initialiser les listes pour les Greeks
            delta_call, delta_put = [], []
            gamma = []
            theta_call, theta_put = [], []
            vega = []
            rho_call, rho_put = [], []

            # Calcul des Greeks pour chaque valeur de spot
            for s in spot_values:
                temp_option = Option(s, Strike, Maturity, Rate, Div, Sigma)
                greeks_temp = temp_option.option_greeks(option_position)

                delta_call.append(greeks_temp["Delta Call"])
                delta_put.append(greeks_temp["Delta Put"])
                gamma.append(greeks_temp["Gamma"])
                theta_call.append(greeks_temp["Theta Call"])
                theta_put.append(greeks_temp["Theta Put"])
                vega.append(greeks_temp["Vega"])
                rho_call.append(greeks_temp["Rho Call"])
                rho_put.append(greeks_temp["Rho Put"])

            # Fonction pour tracer un graphique avec lignes Spot et Strike
            def plot_greek(title, y_values_list, y_label, trace_names):
                fig = go.Figure()
                for y_values, name in zip(y_values_list, trace_names):
                    fig.add_trace(go.Scatter(
                        x=spot_values,
                        y=y_values,
                        mode='lines',
                        name=name
                    ))
                # Ligne pour le Spot avec valeur à droite
                fig.add_vline(
                    x=Spot,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Spot: {Spot:.2f}",
                    annotation_position="top right"
                )
                # Ligne pour le Strike avec valeur à gauche
                fig.add_vline(
                    x=Strike,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Strike: {Strike:.2f}",
                    annotation_position="top left"
                )
                # Ligne horizontale y=0
                fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
                fig.update_layout(
                    title=title,
                    xaxis_title="Spot Price (S)",
                    yaxis_title=y_label
                )
                return fig

            # Tracer Delta
            # Ajouter une description avant chaque Greek
            with st.expander("Delta Dynamics"):
                st.markdown("""
                **Delta** measures the sensitivity of the option price to changes in the spot price:
                - **OTM (Out of the Money)**:
                    - For **Call OTM**, Delta is close to 0 because the option is unlikely to become profitable.
                    - For **Put OTM**, Delta is also close to 0 as the option is unlikely to be exercised.
                - **ATM (At the Money)**:
                    - For **Call ATM**, Delta is approximately 0.5 because there's a 50% probability of ending ITM.
                    - For **Put ATM**, Delta is approximately -0.5 due to the symmetric probability of exercise in the opposite direction.
                - **ITM (In the Money)**:
                    - For **Call ITM**, Delta approaches 1 because the option behaves almost like the underlying asset.
                    - For **Put ITM**, Delta approaches -1, as its value decreases inversely to the spot price.

                **Long vs Short Positions**:
                - For **Long positions**, Delta is positive for Calls and negative for Puts.
                - For **Short positions**, Delta flips in sign (negative for Calls, positive for Puts).

                **Why?**  
                Delta reflects the probability of exercise and sensitivity to the spot price. ITM options closely follow the asset's price, while OTM options are less sensitive as they are unlikely to be exercised.
                """)            
                fig = plot_greek(
                f"Delta vs Spot Price - {option_position} Position",
                [delta_call, delta_put],
                "Delta",
                ["Delta Call", "Delta Put"]
            )
            st.plotly_chart(fig)

            # Tracer Gamma
            # Ajouter les explications pour Gamma
            with st.expander("Gamma Dynamics"):
                st.markdown("""
                **Gamma** measures the rate of change of Delta:
                - **ATM (At the Money)**:
                    - Gamma is highest because Delta changes rapidly with small spot price variations.
                - **OTM/ITM**:
                    - Gamma is close to 0 because Delta becomes stable:
                        - **OTM**: Low probability of exercise, so Delta is nearly constant.
                        - **ITM**: The option behaves almost like the underlying asset, making Delta stable.

                **Why?**  
                Gamma is high near the strike price because the probability of exercise is most sensitive to small changes in the spot price. In OTM/ITM scenarios, this sensitivity diminishes.
                """)            
                fig = plot_greek(
                f"Gamma vs Spot Price - {option_position} Position",
                [gamma],
                "Gamma",
                ["Gamma"]
            )
            st.plotly_chart(fig)

            # Tracer Theta
            # Ajouter les explications pour Theta
            with st.expander("Theta Dynamics"):
                st.markdown("""
                **Theta** measures the time decay of the option price:
                - **ATM (At the Money)**:
                    - Theta is most negative because time decay has the largest impact on options with significant time value.
                - **OTM**:
                    - Theta is less negative because OTM options have little time value to lose.
                - **ITM**:
                    - Theta is also less negative because ITM options rely more on intrinsic value than time value.

                **Long vs Short Positions**:
                - For **Long positions**, Theta is **negative** because time decay reduces the option's value.
                - For **Short positions**, Theta is **positive** because the seller benefits from time decay.

                **Why?**  
                Theta reflects the loss of time value. ATM options are most affected because they have the highest time value, unlike OTM or ITM options.
                """)
                fig = plot_greek(
                    f"Theta vs Spot Price - {option_position} Position",
                    [theta_call, theta_put],
                    "Theta",
                    ["Theta Call", "Theta Put"]
            )
            st.plotly_chart(fig)

            # Tracer Vega
            # Ajouter les explications pour Vega
            with st.expander("Vega Dynamics"):
                st.markdown("""
                **Vega** measures the sensitivity of the option price to changes in volatility:
                - **ATM (At the Money)**:
                    - Vega is highest because volatility significantly impacts the probability of the option ending ITM.
                - **OTM/ITM**:
                    - Vega is lower:
                        - **OTM**: Volatility has little impact on the low probability of exercise.
                        - **ITM**: The option is already profitable, so volatility is less relevant.

                **Why?**  
                Vega is highest for ATM options because their value is highly dependent on the probability of exercise, which is directly affected by volatility. OTM/ITM options are less sensitive to volatility changes.
                """)            
                fig = plot_greek(
                f"Vega vs Spot Price - {option_position} Position",
                [vega],
                "Vega",
                ["Vega"]
            )
            st.plotly_chart(fig)

            # Tracer Rho
            # Ajouter les explications pour Rho
            with st.expander("Rho Dynamics"):
                st.markdown("""
                **Rho** measures the sensitivity of the option price to changes in interest rates:
                - **Call Options**:
                    - Rho is positive because higher interest rates reduce the present value of the strike price, making Calls more valuable.
                - **Put Options**:
                    - Rho is negative because higher interest rates increase the present value of the strike price, making Puts less valuable.

                - **OTM/ATM**:
                    - Rho has a stronger impact because these options are highly sensitive to interest rate changes.
                - **ITM**:
                    - Rho has a smaller impact because the option's intrinsic value dominates.

                **Why?**  
                Rho reflects the effect of interest rates on the present value of the strike price. OTM/ATM options are more sensitive to this change, whereas ITM options are less affected.
                """)            
                fig = plot_greek(
                f"Rho vs Spot Price - {option_position} Position",
                [rho_call, rho_put],
                "Rho",
                ["Rho Call", "Rho Put"]
            )
            st.plotly_chart(fig)

    else:
        st.error("Les données de l'option ne sont pas complètes. Veuillez vérifier vos paramètres.")

# ------------------------------
# Tab3: Volatility Skew/Smile
# ------------------------------
with tab3:
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
            Volatility
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Ajouter une explication générale sur la volatilité
    with st.expander("What is Volatility?"):
        st.markdown("""
        - **Volatility** measures the degree of variation in the price of an asset over time.
        - **Historical Volatility**: Calculated from past price movements of the underlying asset. It reflects the actual volatility observed over a specific period.
        - **Implied Volatility (IV)**: Forward-looking and derived from option prices. It represents the market's expectation of future volatility over the life of the option.
        """)

    # Ajouter une explication sur la volatilité implicite
    with st.expander("What is Implied Volatility?"):
        st.markdown("""
        **Implied Volatility (IV)** is the volatility that, when input into the Black-Scholes model, makes the theoretical option price equal to the market price. 
        - IV reflects the market's collective expectations of future volatility.
        - Unlike historical volatility, which is calculated from past data, IV is **inferred** by solving the Black-Scholes formula in reverse using observed market prices.

        **Why can't we observe IV directly?**
        - IV represents expectations about future uncertainty, which cannot be directly measured like past price movements.
        - To determine IV, traders input the market price into the Black-Scholes model and solve for the volatility that matches this price.

        **Why is IV important in option pricing?**
        - IV ensures that the Black-Scholes theoretical price aligns with the actual market price.
        
        **Overpricing and Underpricing**:
        - If the market price of an option is **higher** than its theoretical value, IV will be **higher** to justify this price.
        - If the market price is **lower**, IV will be **lower**.

        **Examples**:
        - A Call option priced at 5 on the market but theoretically valued at 3 (with 20% volatility) will require IV to be higher (e.g., 30%) to match the 5 price.
        - Conversely, if the market price is 2 and the theoretical price is 3, the IV will drop below 20%.

        **Using IV to compare options**:
        - A **higher IV** suggests the market expects significant future price movements (higher uncertainty).
        - A **lower IV** suggests calmer price expectations.
        
        **Comparing IV to historical volatility provides insights**:
        - **IV > historical volatility**: The option may be **overpriced**, as the market anticipates more risk.
        - **IV < historical volatility**: The option may be **underpriced**, as the market expects lower future volatility.
        """)

    # Vérifier si l'API a été utilisée pour éviter NameError
    if input_mode == "Fetch Data from API (yfinance)" and 'option_chain' in locals():

        # ------------------------------
        # 1. Récupération des IVs depuis Yahoo Finance
        # ------------------------------
        yf_iv_calls = option_chain.calls[['strike', 'impliedVolatility', 'lastPrice']]
        yf_iv_puts = option_chain.puts[['strike', 'impliedVolatility', 'lastPrice']]

        # Définir une plage autour du strike ATM
        atm_range = 0.2  # +/- 7% autour du spot
        lower_bound = Spot * (1 - atm_range)
        upper_bound = Spot * (1 + atm_range)

        # Filtrer les strikes autour du spot
        df_calls_yf = yf_iv_calls[(yf_iv_calls['strike'] >= lower_bound) & (yf_iv_calls['strike'] <= upper_bound)]
        df_puts_yf = yf_iv_puts[(yf_iv_puts['strike'] >= lower_bound) & (yf_iv_puts['strike'] <= upper_bound)]

        # Strikes communs uniquement (à la fois dans les calls et les puts)
        common_strikes = set(df_calls_yf['strike']).intersection(set(df_puts_yf['strike']))
        common_strikes = sorted(common_strikes)  # Tri pour un affichage ordonné

        # Initialisation des colonnes
        mkt_call_prices, mkt_put_prices, call_iv, put_iv, avg_iv, call_iv_theo, put_iv_theo, avg_iv_theo = [], [], [], [], [], [], [], []

        # 1. Calculer la volatilité historique estimée
        vol_estimated = []
        yf_data = YFinanceData(selected_ticker)  # Exemple avec AAPL, récupère la donnée du ticker actuel

        # Appelle une fois la méthode de calcul pour récupérer la vol historique
        vol = yf_data.calculate_historical_volatility(option_maturity=calculated_maturity)

        # On stocke directement la volatilité historique ajustée (même valeur pour tous les strikes)
        vol_estimated_array = [vol.iloc[-1]] * len(common_strikes)  # Même valeur pour chaque strike

        # Initialisation des listes pour les prix théoriques et les gains/pertes
        theoretical_prices = []
        market_avg_prices = []  # Prix moyen marché (call et put)
        profit_potential = []

        # ------------------------------
        # 2. Calcul des IV réelles et théoriques
        # ------------------------------
        for idx, strike in enumerate(common_strikes):
            # Récupérer les IV et prix réels
            call_iv_val = df_calls_yf[df_calls_yf['strike'] == strike]['impliedVolatility'].values
            put_iv_val = df_puts_yf[df_puts_yf['strike'] == strike]['impliedVolatility'].values
            mkt_call_price = df_calls_yf[df_calls_yf['strike'] == strike]['lastPrice'].values
            mkt_put_price = df_puts_yf[df_puts_yf['strike'] == strike]['lastPrice'].values

            # IV réelles (avec arrondi)
            call_iv.append(round(call_iv_val[0], 2) if len(call_iv_val) > 0 else None)
            put_iv.append(round(put_iv_val[0], 2) if len(put_iv_val) > 0 else None)
            mkt_call_prices.append(round(mkt_call_price[0], 2) if len(mkt_call_price) > 0 else None)
            mkt_put_prices.append(round(mkt_put_price[0], 2) if len(mkt_put_price) > 0 else None)

            # Moyenne des IV réelles (avec arrondi)
            valid_iv = list(filter(None, [call_iv[-1], put_iv[-1]]))
            avg_iv.append(round(sum(valid_iv) / len(valid_iv), 2) if valid_iv else None)

            # Calcul IV théoriques via bisection
            if len(mkt_call_price) > 0 and len(mkt_put_price) > 0:
                temp_option = Option(Spot, strike, Maturity, Rate, Div, 0.2)  # Sigma initiale
                imp_vol_call, imp_vol_put = temp_option.implied_vol_bisection(mkt_call_price[0], mkt_put_price[0])
            else:
                imp_vol_call, imp_vol_put = None, None

            # Ajouter les IV théoriques (avec arrondi)
            call_iv_theo.append(round(imp_vol_call, 2) if imp_vol_call else None)
            put_iv_theo.append(round(imp_vol_put, 2) if imp_vol_put else None)

            # Moyenne des IV théoriques (avec arrondi)
            valid_iv_theo = list(filter(None, [call_iv_theo[-1], put_iv_theo[-1]]))
            avg_iv_theo.append(round(sum(valid_iv_theo) / len(valid_iv_theo), 2) if valid_iv_theo else None)

            # Créer une instance de l'option avec la vol historique ajustée
            adjusted_vol = vol_estimated_array[idx]  # Récupère la vol pour ce strike
            option = Option(Spot, strike, Maturity, Rate, Div, adjusted_vol)  # Utilisation de la vol historique
            call_price, put_price = option.option_price()  # Calcul des prix d'options
            theoretical_price = round((call_price + put_price) / 2, 2)  # Moyenne des prix call et put avec arrondi
            theoretical_prices.append(theoretical_price)

            # Calcul du prix moyen du marché (call et put, avec arrondi)
            market_avg_price = (
                round((mkt_call_price[0] + mkt_put_price[0]) / 2, 2)
                if len(mkt_call_price) > 0 and len(mkt_put_price) > 0
                else None
            )
            market_avg_prices.append(market_avg_price)

            # Calcul du profit potentiel basé sur la stratégie (avec arrondi)
            if avg_iv[-1] > vol_estimated_array[idx]:  # Surévalué (Sell Option)
                profit = round(market_avg_price - theoretical_price, 2) if market_avg_price else None
            else:  # Sous-évalué (Buy Option)
                profit = round(theoretical_price - market_avg_price, 2) if market_avg_price else None

            profit_potential.append(profit)


        # ------------------------------
        # 3. Création d'un DataFrame unique pour comparer les IV de yf et celles qu'on a calculées avec la méthode bissection
        # ------------------------------
        df_combined = pd.DataFrame({
            "Strike": common_strikes,
            "Mkt Call Price": mkt_call_prices,
            "IV Call (yf)": call_iv,
            "Mkt Put Price": mkt_put_prices,
            "IV Put (yf)": put_iv,
            "Average IV (yf)": avg_iv,
            "IV Call (Theo)": call_iv_theo,
            "IV Put (Theo)": put_iv_theo,
            "Average IV (Theo)": avg_iv_theo
        })

        # ------------------------------
        # 4. Affichage du DataFrame et du graphique
        # ------------------------------
        st.subheader("Implied Volatility Comparison (yfinance vs bisection)")

        # Explication des méthodes
        with st.expander("What Are We Doing Here?"):
            st.markdown("""
            **Objective**:  
            - To compare the **implied volatilities (IV)** provided by yfinance with those calculated using the **bisection method**.

            **Yahoo Finance IV**:  
            - Directly provided for each strike and expiration.
            - Represents the market's expectations of future volatility.

            **Bisection Method IV**:  
            - Recalculates the implied volatility by solving the Black-Scholes formula backward.
            - Uses the **market price** of options (calls and puts) to find the IV that matches the observed price.

            **Why Compare Them?**  
            - To validate our bisection method and ensure we can replicate market IVs.
            - Discrepancies can reveal potential issues in pricing, input assumptions, or inefficiencies in market data.
            """)

        st.dataframe(df_combined)

        # Graphique interactif
        st.subheader("Volatility Smile & Skew")

        with st.expander("What is a Volatility Smile or Skew?"):
            st.markdown("""
            **Volatility Smile/Skew** refers to a graph that represents **implied volatility (y-axis)** as a function of **strike prices (x-axis)**, for both **put and call options** with the same expiration date (not across different maturities).

            **Volatility Smile** refers to the pattern where implied volatility is higher for deep OTM and ITM options, and lower for ATM options. It creates a "smile-like" curve when plotted.

            **Volatility Skew** refers to the asymmetry in implied volatility between options with different strikes but the same expiration date. It often reflects market bias or expectations.

            **Why Do Smile and Skew Exist?**:
            - The Black-Scholes model assumes constant volatility, but real markets deviate due to factors like:
                - **Crash Risk**: Higher IV for OTM Puts reflects market fear of a significant downside move.
                - **Demand Imbalance**: Traders may buy more OTM Calls or Puts, increasing their IV.

            **Examples of Smile/Skew**:
            - **Equities**:
                - A **negative skew** is common because investors often buy **OTM Puts** as protection against market crashes.
                - This demand increases the price and implied volatility (IV) of puts compared to calls, leading to higher IV for **Puts OTM** than **Calls OTM**.
                - Example: The **S&P 500** often exhibits a negative skew due to the asymmetric risk of sudden market drops.

            - **Commodities**:
                - A **positive skew** is often observed because traders hedge against **supply shocks** that can cause sharp price increases.
                - These shocks, such as hurricanes affecting oil production or droughts impacting agriculture, make **Calls OTM** more expensive, resulting in higher IV for calls than puts.
                - Example: In the **oil market**, geopolitical tensions or natural disasters often increase the demand for **OTM Calls**, causing a positive skew.

            - **FX (Foreign Exchange)**:
                - A **volatility smile** is common in currency markets due to the symmetrical nature of currency pairs.
                - Traders hedge against extreme price movements in both directions (up or down), leading to increased IV for **both OTM Calls and Puts**, especially during times of political or economic uncertainty.
                - Example: For the **EUR/USD pair**, a smile might appear around significant events like central bank decisions, where both sides of the market anticipate high volatility.

            **Interpreting Smile and Skew**:
            - **High IV for OTM Puts**: Indicates fear of significant downside risk (e.g., stock market crash).
            - **High IV for OTM Calls**: Suggests expectations of upside potential (e.g., speculative tech stock rally).
            - **Flat Smile**: Reflects a stable perception of future risks.
            """)

        fig = go.Figure()

        # IV réelle (Yahoo Finance)
        fig.add_trace(go.Scatter(
            x=df_combined["Strike"],
            y=df_combined["Average IV (yf)"],
            mode='markers+lines',
            name='IV (yf)',
            marker=dict(color='purple')
        ))

        # IV théorique (Bisection)
        fig.add_trace(go.Scatter(
            x=df_combined["Strike"],
            y=df_combined["Average IV (Theo)"],
            mode='markers+lines',
            name='IV (Theo)',
            marker=dict(color='cyan')
        ))

        # Ajouter le Spot ATM
        fig.add_vline(x=Spot, line_dash="dash", line_color="orange", annotation_text="ATM Spot")

        fig.update_layout(
            title="Volatility Smile/Skew Graph",
            xaxis_title="Strike Prices",
            yaxis_title="Implied Volatility",
            hovermode="x unified",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # ------------------------------
        # 5. # Trading Strategies
        # ------------------------------
        st.subheader("Historical vs Implied Volatility: Detecting Arbitrage Opportunities")

        with st.expander("Volatility Analysis and Arbitrage Strategy: Detailed Explanation"):
            st.markdown("""
            ### **Introduction to Volatility**
            - **Historical Volatility (HV):** 
            - Calculated from past price movements of the underlying asset.
            - Represents the realized variation in price over a specified period.
            - In our case, we adapted the historical volatility to match the **time-to-maturity (TTM)** of the option for greater coherence.

            - **Implied Volatility (IV):**
            - Forward-looking and derived from market option prices.
            - Represents the market's expectation of future volatility for a specific option (depends on strike and maturity).
            - IV is calculated by solving the Black-Scholes model backward using market prices of options.

            ---

            ### **Steps Implemented in the Application**

            #### 1. **Estimating Historical Volatility (HV)**
            - To make the historical volatility relevant to the option's time-to-maturity:
            1. **Period Matching:** We used a dynamic adjustment of the historical data period (`3mo`, `6mo`, `1y`, `2y`) based on the option's maturity.
            2. **Rolling Window:** Applied a rolling window (e.g., 30 days) to compute the standard deviation of logarithmic returns.
            3. **Annualization:** Converted the rolling standard deviation into an annualized figure to align with financial conventions.

            - This approach ensures the **HV is tailored to the specific option's maturity**, increasing its comparability with implied volatility.

            #### 2. **Implied Volatility (IV) Retrieval**
            - **Market Data:** 
            - Retrieved from Yahoo Finance's API, providing IV values for each strike and maturity.
            - The IV values are specific to each option, reflecting market sentiment for that particular strike and expiration.

            - **Theoretical IV Calculation:**
            - For validation, we implemented a bisection method to compute the theoretical IV based on observed market prices.

            #### 3. **Arbitrage Opportunity Detection**
            - We designed a strategy to compare **average implied volatility (IV)** with **estimated historical volatility (HV):**
            - **Overpriced Options (IV > HV):** Suggests the option is too expensive relative to historical volatility → **Sell the option**.
            - **Underpriced Options (IV < HV):** Suggests the option is undervalued relative to historical volatility → **Buy the option**.

            - **Visualization:**
            - A comparison table shows key metrics:
                - Strike Price
                - Average IV (from Yahoo Finance)
                - Estimated HV
                - Evaluation: "Overpriced" or "Underpriced"
                - Suggested Strategy: "Sell Option" or "Buy Option"
            - Graphs illustrate:
                - **Volatility Smile/Skew**: Plots IV values by strike prices.
                - **IV vs. HV Comparison**: Highlights arbitrage opportunities.

            ---

            ### **Results Observed**
            - **Differences Across Stocks:**
            - The strategy shows variability depending on the stock (e.g., TSLA vs. AAPL).
            - For some tickers and expirations (e.g., AAPL), the strategy predominantly suggests selling options, while others (e.g., TSLA) reveal a mix of "buy" and "sell" signals.

            ---

            ### **Key Takeaways**
            1. **Dynamic HV Adaptation:** By matching HV to the option's time-to-maturity, the comparison between HV and IV becomes more consistent and reliable.
            2. **Arbitrage Opportunities:** Comparing HV with IV provides insights into whether an option is mispriced relative to historical market data.
            3. **Practical Application:** The tool assists traders in making informed decisions to exploit market inefficiencies.

            ---

            ### **Next Steps**
            - Extend the analysis by incorporating **volatility surfaces** (IV variations across strikes and maturities).
            - Include additional filters (e.g., liquidity or bid-ask spreads) to refine trade recommendations.
            - Allow user-defined rolling windows for HV calculations.
            """)

        # ------------------------------
        # Création du DataFrame pour l'arbitrage
        # ------------------------------
        avg_iv_array = np.array(avg_iv)

        # Calcul des différences entre prix théorique et prix de marché moyen
        price_difference = [
            theoretical - market if theoretical is not None and market is not None else None
            for theoretical, market in zip(theoretical_prices, market_avg_prices)
        ]

        df_iv_arbitrage_opportunity = pd.DataFrame({
            "Strike": common_strikes,
            "Market Average Price": market_avg_prices,  # Prix moyen marché (call et put)
            "Theoretical Average Price": theoretical_prices,  # Prix théorique moyen (call et put)
            "Average IV (yf)": avg_iv_array,  # IV moyenne depuis yfinance
            "Estimated Vol": vol_estimated_array,  # Volatilité historique ajustée
            "Vol Difference": avg_iv_array - vol_estimated_array,  # Différence entre IV moyenne et HV
            "Evaluation": np.where(avg_iv_array > vol_estimated_array, "Overpriced", "Underpriced"),  # Sur ou sous-évalué
            "Profit Potential (price difference)": profit_potential,  # Gains ou pertes potentiels
            "Strategy": np.where(avg_iv_array > vol_estimated_array, "Sell Option", "Buy Option")  # Stratégie recommandée
        })        
        
        # Afficher le DataFrame
        st.dataframe(df_iv_arbitrage_opportunity)

        # Graphique interactif
        fig = go.Figure()

        # IV réelle (Yahoo Finance)
        fig.add_trace(go.Scatter(
            x=df_iv_arbitrage_opportunity["Strike"],
            y=df_iv_arbitrage_opportunity["Average IV (yf)"],
            mode='markers+lines',
            name='IV (yf)',
            marker=dict(color='purple')
        ))

        # Volatilité estimée (historique)
        fig.add_trace(go.Scatter(
            x=df_iv_arbitrage_opportunity["Strike"],
            y=df_iv_arbitrage_opportunity["Estimated Vol"],
            mode='markers+lines',
            name='Estimated Vol',
            marker=dict(color='orange')
        ))

        # Ajouter le Spot ATM
        fig.add_vline(x=Spot, line_dash="dash", line_color="blue", annotation_text="ATM Spot")

        fig.update_layout(
            title="Implied Volatility vs Estimated Volatility",
            xaxis_title="Strike Prices",
            yaxis_title="Volatility",
            hovermode="x unified",
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        # ------------------------------
        # Résumé des performances de la stratégie d'arbitrage
        # ------------------------------

        # Filtrer les valeurs valides pour le calcul de la moyenne
        valid_profits = [profit for profit in profit_potential if profit is not None]

        # Calcul des statistiques globales
        average_profit = np.mean(valid_profits) if valid_profits else 0
        num_buy_positions = sum(df_iv_arbitrage_opportunity["Strategy"] == "Buy Option")
        num_sell_positions = sum(df_iv_arbitrage_opportunity["Strategy"] == "Sell Option")

        # Création du DataFrame récapitulatif
        df_summary = pd.DataFrame({
            "Metric": ["Average Profit", "Number of Buy Positions", "Number of Sell Positions"],
            "Value": [average_profit, num_buy_positions, num_sell_positions]
        })

        # Afficher le DataFrame récapitulatif
        st.subheader("Arbitrage Strategy Summary")
        st.dataframe(df_summary)

    else:
        st.warning("Please fetch data from API (yfinance) to use this functionality.")

# ------------------------------
# Tab4: Monte Carlo
# ------------------------------
with tab4:
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
            Monte Carlo Simulator
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Explanations
    with st.expander("What is Monte Carlo Simulation and How Does It Work?"):
        st.markdown("""
        ### **Introduction to Monte Carlo Simulation**
        Monte Carlo simulations are used to model the potential evolution of spot prices based on the **Geometric Brownian Motion (GBM)** framework. This method generates multiple random paths to predict how the price of an asset might behave under different market conditions.

        **Why Use Monte Carlo Simulations?**
        - To estimate the potential future behavior of spot prices.
        - To calculate statistical metrics like the mean price and standard deviation at maturity.
        - To visualize the range of possible outcomes and assess risk.

        ---

        ### **Key Parameters**
        - **Drift (µ):**
            - Represents the expected return or trend of the asset over time.
            - A positive drift indicates an upward trend, while a negative drift indicates a downward trend.
        - **Volatility (σ):**
            - Reflects the uncertainty or risk of the asset's price movements.
            - Higher volatility leads to larger and more frequent price fluctuations.
        - **Time Horizon:**
            - The total time (in years) over which the simulation runs.
            - This corresponds to the time to maturity of the option or investment.
        - **Number of Steps:**
            - Defines how finely the time horizon is divided.
            - A higher number of steps increases the precision of the simulated paths but requires more computation.
        - **Number of Simulations:**
            - Specifies how many independent paths are generated.
            - A larger number of simulations provides a better approximation of the price distribution.

        ---

        ### **The Monte Carlo Process**
        1. **Initialization:**
            - Start with the current spot price (initial condition).
        2. **Simulation:**
            - For each step:
                - Generate a random shock using a normal distribution.
                - Compute the new price using the **Geometric Brownian Motion formula**:
                \[
                S_{t+1} = S_t \cdot e^{\left( \mu - \frac{\sigma^2}{2} \right) \Delta t + \sigma \sqrt{\Delta t} \cdot Z}
                \]
                - \( S_t \): Price at the current step.
                - \( \Delta t \): Time increment per step.
                - \( Z \): Random variable from a standard normal distribution.
        3. **Statistics:**
            - At the end of the simulation, calculate:
                - **Mean Price at Maturity:** Average final price across all paths.
                - **Standard Deviation at Maturity:** Measure of price dispersion at maturity.
        4. **Visualization:**
            - Plot the simulated paths to observe price variations over time.

        ---

        ### **Practical Insights**
        - **Drift and Volatility:**
            - A higher drift results in a more upward-biased simulation.
            - A higher volatility creates more uncertainty, leading to wider price ranges.
        - **Number of Steps and Simulations:**
            - Increasing the number of steps smoothens the paths but increases computation time.
            - More simulations improve the accuracy of the statistical metrics.

        ---

        ### **Applications**
        - Option Pricing: To estimate the expected payoff of options under uncertain price movements.
        - Risk Analysis: To assess the range of possible outcomes and potential losses.
        - Portfolio Management: To predict how an asset might perform over a specific horizon.

        **Example:**
        - For a stock with:
            - Initial price (\( S_0 \)): $100
            - Drift (\( µ \)): 5%
            - Volatility (\( σ \)): 20%
            - Time to Maturity: 1 year
            - 100 steps and 1000 simulations:
        - The Monte Carlo simulation will generate 1000 potential paths for the stock's price over one year.
        - This helps to estimate the likely price distribution and assess the risk-return tradeoff.
        """)

    # Inputs utilisateur pour Monte Carlo avec des clés uniques
    st.subheader("Monte Carlo Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        mu = st.number_input(
            "Drift (µ)", 
            min_value=-1.0, 
            max_value=1.0, 
            value=0.05, 
            step=0.01, 
            key="mc_mu", 
            help=(
                "The drift represents the expected rate of return of the asset. "
                "A positive drift means the asset is expected to grow on average, "
                "while a negative drift implies a decline."
            )
        )

    with col2:
        num_steps = st.number_input(
            "Number of Steps", 
            min_value=1, 
            value=100, 
            step=1, 
            key="mc_num_steps", 
            help=(
                "The number of time steps used to simulate each path. "
                "A higher number provides more granularity in the simulation, "
                "but increases computational cost."
            )
        )

    with col3:
        num_simulations = st.number_input(
            "Number of Simulations", 
            min_value=1, 
            value=1000, 
            step=100, 
            key="mc_num_simulations", 
            help=(
                "The total number of paths to simulate. "
                "More simulations provide better statistical reliability, "
                "but increase computational cost."
            )
        )
    # Vérifier que tous les paramètres sont valides
    if all([Spot, mu, Sigma, Maturity, num_steps, num_simulations]):
        # Instancier la classe Monte Carlo
        simulation = MonteCarlo(Spot, mu, Sigma, Maturity, num_steps, num_simulations)

        # Simuler les trajectoires
        paths = simulation.simulate_paths()

        # Tracer les trajectoires simulées
        st.subheader("Simulated Price Paths")
        fig_paths = simulation.plot_paths(paths)
        st.plotly_chart(fig_paths)

        # Calculer et afficher les statistiques
        st.subheader("Statistics at Maturity")
        mean_price, std_dev = simulation.calculate_statistics(paths)
        col1, col2 = st.columns(2)
        col1.metric("Average Price at Maturity", f"${mean_price:.2f}")
        col2.metric("Standard Deviation at Maturity", f"${std_dev:.2f}")

        # Ajouter des options interactives pour analyser les résultats
        st.subheader("Explore Simulated Prices")
        selected_simulation = st.slider(
            "Select Simulation Path to View Details", 
            min_value=1, 
            max_value=min(num_simulations, 10), 
            value=1, 
            step=1
        )
        st.line_chart(paths[selected_simulation - 1])
    else:
        st.warning("Please fill all parameters to simulate the paths.")

# ------------------------------
# Tab5: Hedging
# ------------------------------
with tab5:
    # Titre principal
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
            Delta & Gamma Hedging
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Explanations
    with st.expander("What are Delta Hedging and Gamma Hedging?"):
        st.markdown("""
        ## **Delta Hedging and Gamma Hedging**
        In this section, we focus on managing option risks through **Delta Hedging** and **Gamma Hedging**. These techniques aim to neutralize sensitivities to changes in the underlying asset's price.

        ### **1. Delta Hedging**
        - **What is Delta?**
        - Delta measures the sensitivity of an option's price to changes in the underlying asset's price.
        - For example, a delta of 0.5 means the option price will increase by 0.50 if the underlying price increases by 1.

        - **Purpose of Delta Hedging:**
        - Neutralize the exposure to the underlying asset's price movements by dynamically adjusting the position in the underlying asset.
        
        - **How it Works:**
        - **For a Call Option:**
            - If you are **long a call option** (positive delta):
            - Hedge by **shorting shares** of the underlying asset in proportion to the delta.
            - If you are **short a call option** (negative delta):
            - Hedge by **buying shares** of the underlying asset in proportion to the delta.

        - **For a Put Option:**
            - If you are **long a put option** (negative delta):
            - Hedge by **buying shares** of the underlying asset in proportion to the delta.
            - If you are **short a put option** (positive delta):
            - Hedge by **shorting shares** of the underlying asset in proportion to the delta.

        - **Dynamic Adjustment:**
        - As the delta changes with time and price movements, the hedging position is rebalanced dynamically.

        ### **2. Gamma Hedging**
        - **What is Gamma?**
        - Gamma measures the sensitivity of Delta to changes in the underlying asset's price.
        - A high Gamma means Delta changes significantly for small price movements.

        - **Purpose of Gamma Hedging:**
        - Neutralize the risk from rapid changes in Delta by using additional instruments (like other options).

        - **How it Works:**
        - Delta is managed at every step using Delta Hedging, but Gamma helps predict and manage changes in Delta for smoother hedging adjustments.
        - Typically, Gamma Hedging involves buying or selling options to offset Delta variability.

        ### **Rules for Delta and Gamma Hedging**
        #### **Case 1: Long Call (Delta > 0, bullish)**
        - **At t=0:**
        - Initial Delta (Δ₀) > 0.
        - Hedge: Short `Δ₀ × Qty` shares of the underlying.
        - **At t=1:**
        - **If Spot Increases (S₁ > S₀):**
            - Delta increases (Δ₁ > Δ₀) → Closer to 1.
            - Hedge: Short `(Δ₁ - Δ₀) × Qty` additional shares.
        - **If Spot Decreases (S₁ < S₀):**
            - Delta decreases (Δ₁ < Δ₀) → Closer to 0.
            - Hedge: Long `(Δ₁ - Δ₀) × Qty` shares.

        #### **Case 2: Short Call (Delta < 0, bearish)**
        - **At t=0:**
        - Initial Delta (Δ₀) < 0.
        - Hedge: Long `Δ₀ × Qty` shares of the underlying.
        - **At t=1:**
        - **If Spot Increases (S₁ > S₀):**
            - Delta decreases (Δ₁ < Δ₀) → Closer to -1.
            - Hedge: Long `(Δ₁ - Δ₀) × Qty` additional shares.
        - **If Spot Decreases (S₁ < S₀):**
            - Delta increases (Δ₁ > Δ₀) → Closer to 0.
            - Hedge: Short `(Δ₁ - Δ₀) × Qty` shares.

        #### **Case 3: Long Put (Delta < 0, bearish)**
        - **At t=0:**
        - Initial Delta (Δ₀) < 0.
        - Hedge: Long `Δ₀ × Qty` shares of the underlying.
        - **At t=1:**
        - **If Spot Increases (S₁ > S₀):**
            - Delta increases (Δ₁ > Δ₀) → Closer to 0.
            - Hedge: Short `(Δ₁ - Δ₀) × Qty` shares.
        - **If Spot Decreases (S₁ < S₀):**
            - Delta decreases (Δ₁ < Δ₀) → Closer to -1.
            - Hedge: Long `(Δ₁ - Δ₀) × Qty` additional shares.

        #### **Case 4: Short Put (Delta > 0, bullish)**
        - **At t=0:**
        - Initial Delta (Δ₀) > 0.
        - Hedge: Short `Δ₀ × Qty` shares of the underlying.
        - **At t=1:**
        - **If Spot Increases (S₁ > S₀):**
            - Delta decreases (Δ₁ < Δ₀) → Closer to 0.
            - Hedge: Long `(Δ₁ - Δ₀) × Qty` shares.
        - **If Spot Decreases (S₁ < S₀):**
            - Delta increases (Δ₁ > Δ₀) → Closer to 1.
            - Hedge: Short `(Δ₁ - Δ₀) × Qty` additional shares.

        ### **What We Do in Tab 5**
        - **Delta Hedging:** Dynamically calculates the required hedging position for call and put options based on Delta changes.
        - **Gamma Hedging:** Adjusts Delta hedging positions dynamically over multiple steps to account for Gamma.
        - **Visualization:**
        - A table provides detailed results for hedging quantities, costs, and profit/loss (PnL) over time.
        - Charts show how the net Gamma PnL evolves dynamically.

        ### **About the Charts**
        - The **Dynamic Delta Hedging Analysis** graphs show the relationship between the **Spot Price** and the hedging metrics:
        1. **Call Option:**
            - **Blue Line:** Hedging quantity required to offset Delta for a call option.
            - **Red Line:** Hedging cost associated with the hedging quantity.
        2. **Put Option:**
            - **Blue Line:** Hedging quantity required to offset Delta for a put option.
            - **Red Line:** Hedging cost associated with the hedging quantity.
        - The vertical dashed lines represent:
            - **Spot Price (yellow):** Current price of the underlying.
            - **Strike Price (purple):** Exercise price of the option.
        - These charts help visualize the dynamics of hedging positions as the spot price changes.
        """)

    # Input utilisateur
    st.subheader("Hedging Parameters")
    col1, col2 = st.columns(2)
    with col1:
        option_qty = st.number_input(
            "Option Quantity", 
            min_value=1, 
            value=10, 
            step=1, 
            help=("The total number of options in your portfolio."
                  "This value is used to determine the hedging quantities and the associated costs for maintaining a balanced risk exposure.")
        )
    with col2:
        transaction_fee = st.number_input(
            "Transaction Fee (%)", 
            min_value=0.0, 
            value=0.01, 
            step=0.001,
            help=("The percentage cost associated with buying or selling shares to hedge the option. "
                  "This fee represents the transaction cost incurred for each trade in the hedging process. "
                  "For example, a transaction fee of 0.01% means that for every USD 1,000 traded, USD 0.10 is charged as a fee.")
        )  # en pourcentage

    # Créer un objet Monte Carlo
    monte_carlo = MonteCarlo(Spot, mu, Sigma, Maturity, num_steps, num_simulations)

    if st.button("Run Gamma Hedging"):
        # Lancer le calcul de gamma hedge
        gamma_hedge_results_call, gamma_hedge_results_put = option.gamma_hedge(monte_carlo, transaction_fee)

        # Affichage des résultats Call
        st.subheader(f"Gamma Hedging Results -  {option_position} Call Option")
        st.dataframe(gamma_hedge_results_call.style.format({
            "Spot": "{:.2f}",
            "Delta": "{:.4f}",
            "Gamma": "{:.4f}",
            "Delta Hedge Quantity": "{:.2f}",
            "Delta Hedge Cost": "{:.2f}",
            "Gamma Hedge Quantity": "{:.2f}",
            "Gamma Hedge Cost": "{:.2f}",
            "Gamma PnL": "{:.2f}",
            "Transaction Cost": "{:.2f}",
            "Net Gamma PnL": "{:.2f}"
        }))

        # Affichage des résultats Put
        st.subheader(f"Gamma Hedging Results - {option_position} Put Option")
        # Annotation on the results
        st.markdown(
            """
            **Note:**
            - The results related to **Delta Hedging** (quantity and cost) differ between Call and Put options because of the opposing values of Delta.
            - However, the results for **Gamma Hedging** (quantity and cost) are identical for Long Call and Long Put positions because Gamma depends solely on the sensitivity of Delta to changes in the Spot Price, which is the same for both options.
            """
        )
        st.dataframe(gamma_hedge_results_put.style.format({
            "Spot": "{:.2f}",
            "Delta": "{:.4f}",
            "Gamma": "{:.4f}",
            "Delta Hedge Quantity": "{:.2f}",
            "Delta Hedge Cost": "{:.2f}",
            "Gamma Hedge Quantity": "{:.2f}",
            "Gamma Hedge Cost": "{:.2f}",
            "Gamma PnL": "{:.2f}",
            "Transaction Cost": "{:.2f}",
            "Net Gamma PnL": "{:.2f}"
        }))

        # Affichage du graphique Call (Gamma PnL)
        st.subheader(f"Net Gamma PnL over time - {option_position} Call/Put Option")
        st.line_chart(gamma_hedge_results_call["Net Gamma PnL"])

        # Graphiques delta heding
        fig_call, fig_put = option.plot_delta_hedging_from_gamma_hedge(gamma_hedge_results_call, gamma_hedge_results_put, Strike)
        st.plotly_chart(fig_call, use_container_width=True)
        st.plotly_chart(fig_put, use_container_width=True)

#------------------------------
# Tab6: PnL
#------------------------------
with tab6: 
    # Title
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
            Option PnL Analysis
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Explication du contenu dans Tab6
    with st.expander("What are we doing here?"):
        st.markdown(
            """
            ## Overview
            This section focuses on analyzing the **Profit and Loss (PnL)** of call and put options across various market scenarios and purchase prices. The tab provides dynamic PnL calculations and interactive heatmaps to help traders visualize potential outcomes.

            ---

            ## Features

            ### 1. **Dynamic PnL Calculation**
            The PnL for both call and put options is calculated based on user-defined purchase prices and the current market price of the options. The calculations adjust automatically depending on whether the position is **Long** or **Short**:
            - **For Long Positions**:
            - PnL = Current Option Price - Purchase Price
            - **For Short Positions**:
            - PnL = Purchase Price - Current Option Price

            ### 2. **User Inputs**
            The sidebar allows users to input key parameters for the analysis:
            - **Purchase Price**: The price at which you bought the call or put option.
            - **Spot Prices**: Automatically generates a range of spot prices, including the current spot price for relevance.

            ### 3. **PnL Heatmaps**
            Heatmaps provide a visual representation of how PnL changes across:
            - **Spot Prices**: Prices of the underlying asset.
            - **Purchase Prices**: The price at which the option was bought.
            Separate heatmaps are created for:
            - **Call Options**
            - **Put Options**

            ---

            ## How It Works

            ### 1. **PnL Metrics Display**
            The `calculate_pnl` method computes the PnL for call and put options using the user's inputs:
            - **Purchase Price**
            - **Spot Price**
            - **Option Position (Long or Short)**

            The results are displayed as metrics using `st.metric`, providing a quick summary of the current PnL for both calls and puts.

            ### 2. **Dynamic Heatmap Generation**
            Heatmaps simulate the impact of varying **spot prices** and **purchase prices** on PnL:
            - A temporary `Option` object is created for each combination of spot price and strike price.
            - The method calculates the option price dynamically for each scenario.

            #### Call Option Heatmap
            - **X-axis**: Spot prices.
            - **Y-axis**: Purchase call prices.
            - **Cell Values**: PnL for each combination of spot price and purchase price.

            #### Put Option Heatmap
            - **X-axis**: Spot prices.
            - **Y-axis**: Purchase put prices.
            - **Cell Values**: PnL for each combination of spot price and purchase price.

            ### 3. **Visualization Details**
            - **Colors**: Heatmaps use green to indicate profit and red to indicate loss, making it easy to identify profit zones and breakeven points.
            - **Annotations**: Each cell is annotated with the exact PnL value for better interpretability.

            ---

            ## Why This Tab is Important

            This tab is critical for traders and analysts as it:
            1. **Evaluates Market Scenarios**: Understand how PnL is affected by different spot price movements.
            2. **Highlights Sensitivities**: See how changes in purchase prices impact profitability.
            3. **Identifies Breakeven Levels**: Pinpoint the spot price and purchase price combinations where PnL is zero.
            4. **Aids Decision-Making**: Clear visualizations and metrics provide actionable insights for better risk management and strategy optimization.

            ---
            """
    )

    option = Option(Spot, Strike, Maturity, Rate, Div, Sigma)  
    call_price, put_price = option.option_price() 
    # # Get last call and put prices from API
    # mkt_call_price = option_chain.calls.loc[option_chain.calls['strike'] == Strike, 'lastPrice'].values
    # mkt_put_price = option_chain.puts.loc[option_chain.puts['strike'] == Strike, 'lastPrice'].values
    # mkt_call_price = mkt_call_price[0] if len(mkt_call_price) > 0 else None
    # mkt_put_price = mkt_put_price[0] if len(mkt_put_price) > 0 else None

    # Input utilisateur
    st.subheader("PnL Parameters")
    col1, col2 = st.columns(2)
    with col1:
        purchase_price_call = st.number_input(
            "Purchase Call Price",
            min_value=0.0,
            value = call_price, #if input_mode == "Manual Inputs" else mkt_call_price,
            step=0.01,
            help="The price at which you bought the call"
        )

    with col2:
        purchase_price_put = st.number_input(
            "Purchase Put Price",
            min_value=0.0,
            value = put_price, #if input_mode == "Manual Inputs" else mkt_put_price,
            step=0.01,
            help="The price at which you bought the put"
    )
    
    # Display PnL results
    st.subheader("PnL Results")
    pnl_call, pnl_put = option.calculate_pnl(purchase_price_call, purchase_price_put, option_position)
    # col1, col2 = st.columns(2)
    # col1.metric("PnL (Call)", f"${pnl_call:.2f}")
    # col2.metric("PnL (Put)", f"${pnl_put:.2f}")         

    # ----------------------------------
    # Heatmap for Call PnL
    # ----------------------------------
    
    # Générer différents spot prices basés sur l'input
    spot_values = np.linspace(Spot * 0.5, Spot * 1.5, 10)
    
    # Assurer que l'input de l'utilisateur soit dans la heatmap
    if Spot not in spot_values:
        spot_values = np.append(spot_values, Spot)
        spot_values = np.sort(spot_values)  # Trie les ticks
    
    # Générer différents call prices basés sur l'input
    purchase_call_prices = np.linspace(
        purchase_price_call if purchase_price_call > 0 else call_price * 0.5,
        purchase_price_call * 1.5 if purchase_price_call > 0 else call_price * 1.5,
        10
    )
    
    # Créer la matrice pour le PnL du call
    pnl_matrix_call = np.zeros((len(purchase_call_prices), len(spot_values)))
    
    # Créer la matrice pour le PnL du call avec position prise en compte
    for row_idx_call, purchase_call_price in enumerate(purchase_call_prices):
        for col_idx_call, spot_price in enumerate(spot_values):
            temp_option = Option(spot_price, Strike, Maturity, Rate, Div, Sigma)
            current_call_price, _ = temp_option.option_price()
            
            if option_position == "Long":
                pnl_matrix_call[row_idx_call, col_idx_call] = current_call_price - purchase_call_price
            elif option_position == "Short":
                pnl_matrix_call[row_idx_call, col_idx_call] = purchase_call_price - current_call_price 

    # Mise à jour des étiquettes pour le plot Call
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pnl_matrix_call,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        xticklabels=np.round(spot_values, 2),
        yticklabels=np.round(purchase_call_prices, 2),
        cbar_kws={'label': 'PnL'},
        annot_kws={"size": 7}
    )
    ax.invert_yaxis()
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Purchase Call Price")
    ax.set_title("PnL Heatmap for Call Option")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # ----------------------------------
    # Heatmap for Put PnL
    # ----------------------------------
    
    # Initialiser les plages dynamiques basées sur les inputs utilisateur
    purchase_put_prices = np.linspace(
        purchase_price_put if purchase_price_put > 0 else put_price * 0.5,
        purchase_price_put * 1.5 if purchase_price_put > 0 else put_price * 1.5,
        10
    )
    
    # Créer la matrice pour le PnL du put
    pnl_matrix_put = np.zeros((len(purchase_put_prices), len(spot_values)))
    
    # Créer la matrice pour le PnL du put avec position prise en compte
    for row_idx_put, purchase_put_price in enumerate(purchase_put_prices):
        for col_idx_put, spot_price in enumerate(spot_values):
            temp_option = Option(spot_price, Strike, Maturity, Rate, Div, Sigma)
            _, current_put_price = temp_option.option_price()
            
            if option_position == "Long":
                pnl_matrix_put[row_idx_put, col_idx_put] = current_put_price - purchase_put_price
            elif option_position == "Short":
                pnl_matrix_put[row_idx_put, col_idx_put] = purchase_put_price - current_put_price    
                
    # Mise à jour des étiquettes pour le plot Put
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pnl_matrix_put,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        xticklabels=np.round(spot_values,2),
        yticklabels=np.round(purchase_put_prices, 2),
        cbar_kws={'label': 'PnL'},
        annot_kws={"size": 7}
    )
    ax.invert_yaxis()
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Purchase Put Price")
    ax.set_title("PnL Heatmap for Put Option")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

#------------------------------
# Tab7: Option Strategy
#------------------------------
with tab7:
    # Titre de l'onglet
    st.markdown(
        """
        <h3 style="text-align:center; margin-top:0px; margin-bottom:20px;">
        Option Strategy
        </h3>
        """,
        unsafe_allow_html=True,
    )

    # Disclaimer
    st.markdown("""
    <div style="background-color: #f9f9a9; padding: 10px; border: 1px solid #ccc; border-radius: 5px; color: #333;">
        <strong>Note:</strong> The inputs on the left do not apply to this section.
        This tab is independent and exclusively dedicated to option strategy analysis.
    </div>
    """, unsafe_allow_html=True)

    # Saut de ligne
    st.markdown("<br>", unsafe_allow_html=True)

    # Explanations
    with st.expander("What are Option Strategies?"):
        st.markdown("""
        ## **Option Strategies Overview**
        In this section, we explore popular option trading strategies. These strategies are combinations of calls and puts designed to profit under specific market conditions. Below are explanations of the most common strategies:

        ### **1. Call Spread**
        - **Description:** A call spread involves buying a call option at a lower strike price and selling another call option at a higher strike price.
        - **Position Types:**
            - **Long Call Spread:** The investor is bullish on the underlying asset and expects moderate upward movement. This strategy limits both the profit and the loss.
                - **Buy 1 Call (lower strike)**: Gain if the price increases.
                - **Sell 1 Call (higher strike)**: Reduce cost but cap potential profit.
            - **Short Call Spread:** The investor is bearish or neutral and expects the price to stay below the lower strike price. This strategy has limited loss and gain.
                - **Sell 1 Call (lower strike)**: Profit if the price remains below the strike.
                - **Buy 1 Call (higher strike)**: Protect against significant losses.

        ### **2. Put Spread**
        - **Description:** A put spread involves buying a put option at a higher strike price and selling another put option at a lower strike price.
        - **Position Types:**
            - **Long Put Spread:** The investor is bearish on the underlying asset and expects moderate downward movement. Losses and profits are both capped.
                - **Buy 1 Put (higher strike)**: Gain if the price drops.
                - **Sell 1 Put (lower strike)**: Reduce cost but cap potential profit.
            - **Short Put Spread:** The investor is bullish or neutral and expects the price to stay above the higher strike price. This strategy has limited loss and gain.
                - **Sell 1 Put (higher strike)**: Profit if the price remains above the strike.
                - **Buy 1 Put (lower strike)**: Protect against significant losses.

        ### **3. Strangle**
        - **Description:** A strangle involves buying an out-of-the-money call and an out-of-the-money put. 
        - **Position Types:**
            - **Long Strangle:** The investor expects high volatility and significant price movement in either direction.
                - **Buy 1 Call (OTM)**: Gain if the price rises sharply.
                - **Buy 1 Put (OTM)**: Gain if the price drops sharply.
            - **Short Strangle:** The investor expects low volatility and stable prices.
                - **Sell 1 Call (OTM)**: Gain if the price remains below the call strike.
                - **Sell 1 Put (OTM)**: Gain if the price remains above the put strike.

        ### **4. Straddle**
        - **Description:** A straddle involves buying both an at-the-money call and an at-the-money put.
        - **Position Types:**
            - **Long Straddle:** The investor expects significant price movement in either direction, regardless of the direction.
                - **Buy 1 Call (ATM)**: Gain if the price rises significantly.
                - **Buy 1 Put (ATM)**: Gain if the price drops significantly.
            - **Short Straddle:** The investor expects little to no price movement.
                - **Sell 1 Call (ATM)**: Gain if the price remains stable.
                - **Sell 1 Put (ATM)**: Gain if the price remains stable.

        ### **5. Call Butterfly**
        - **Description:** A call butterfly involves buying one call option at a lower strike price, selling two calls at a middle strike price, and buying one call at a higher strike price.
        - **Position Types:**
            - **Long Call Butterfly:** The investor expects low volatility and the price to stay near the middle strike. This strategy limits both the profit and the loss.
                - **Buy 1 Call (lower strike)**: Gain if the price rises toward the middle strike.
                - **Sell 2 Calls (middle strike)**: Collect premium, profit capped.
                - **Buy 1 Call (higher strike)**: Protect against significant price increases.
            - **Short Call Butterfly:** The investor expects high volatility and large price movements in either direction.
                - **Sell 1 Call (lower strike)**: Loss if the price rises toward the middle strike.
                - **Buy 2 Calls (middle strike)**: Protect against losses.
                - **Sell 1 Call (higher strike)**: Profit if the price moves far from the middle strike.

        ### **6. Put Butterfly**
        - **Description:** A put butterfly involves buying one put option at a higher strike price, selling two puts at a middle strike price, and buying one put at a lower strike price.
        - **Position Types:**
            - **Long Put Butterfly:** The investor expects low volatility and the price to stay near the middle strike. This strategy limits both the profit and the loss.
                - **Buy 1 Put (higher strike)**: Gain if the price drops toward the middle strike.
                - **Sell 2 Puts (middle strike)**: Collect premium, profit capped.
                - **Buy 1 Put (lower strike)**: Protect against significant price decreases.
            - **Short Put Butterfly:** The investor expects high volatility and large price movements in either direction.
                - **Sell 1 Put (higher strike)**: Loss if the price drops toward the middle strike.
                - **Buy 2 Puts (middle strike)**: Protect against losses.
                - **Sell 1 Put (lower strike)**: Profit if the price moves far from the middle strike.

        ### **Summary**
        These strategies allow investors to profit in specific market conditions (bullish, bearish, or neutral). They also help in managing risks by capping potential losses or gains. Each strategy requires careful consideration of the underlying asset's price movement and volatility.
        """)

    # Menu déroulant pour choisir la stratégie
    strategy_choice = st.selectbox("Choose an Option Strategy:", ["Call Spread", "Put Spread", "Strangle", "Straddle", "Call Butterfly", "Put Butterfly"])

    if strategy_choice == "Call Spread":
        st.subheader("Parameters for Call Spread")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            K1 = st.number_input("Strike Price K1 (lower)", value=90.0, step=1.0)
            K2 = st.number_input("Strike Price K2 (higher)", value=110.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, K1, maturity, rate, div, sigma)
        df_call_spread_option_results = option.call_spread(option_position, K1, K2, call_K1_price=None, call_K2_price=None)
        st.subheader("Results")
        st.dataframe(df_call_spread_option_results)

        # Afficher les graphiques
        plot_call_spread_fig = option.plot_call_spread(option_position, spot, K1, K2, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_call_spread_fig)

    elif strategy_choice == "Put Spread":
        st.subheader("Parameters for Put Spread")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            K1 = st.number_input("Strike Price K1 (lower)", value=90.0, step=1.0)
            K2 = st.number_input("Strike Price K2 (higher)", value=110.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, K1, maturity, rate, div, sigma)
        df_put_spread_option_results = option.put_spread(option_position, K1, K2, put_K1_price=None, put_K2_price=None)
        st.subheader("Results")
        st.dataframe(df_put_spread_option_results)

        # Afficher les graphiques
        plot_put_spread_fig = option.plot_put_spread(option_position, spot, K1, K2, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_put_spread_fig)

    elif strategy_choice == "Strangle":
        st.subheader("Parameters for Strangle")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            K_put = st.number_input("Strike Price K_put (lower)", value=90.0, step=1.0)
            K_call = st.number_input("Strike Price K_call (higher)", value=110.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, K_put, maturity, rate, div, sigma)
        df_strangle_option_results = option.strangle(option_position, K_put, K_call, put_price=None, call_price=None)
        st.subheader("Results")
        st.dataframe(df_strangle_option_results)

        # Afficher les graphiques
        plot_strangle_fig = option.plot_strangle(option_position, spot, K_put, K_call, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_strangle_fig)

    elif strategy_choice == "Straddle":
        st.subheader("Parameters for Straddle")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            strike = st.number_input("Strike Price", value=100.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, strike, maturity, rate, div, sigma)
        df_straddle_option_results = option.straddle(option_position, strike, put_price=None, call_price=None)
        st.subheader("Results")
        st.dataframe(df_straddle_option_results)

        # Afficher les graphiques
        plot_straddle_fig = option.plot_straddle(option_position, spot, strike, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_straddle_fig)

    elif strategy_choice == "Call Butterfly":
        st.subheader("Parameters for Call Butterfly")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            K1 = st.number_input("Strike Price K1 (lowest)", value=90.0, step=1.0)
            K2 = st.number_input("Strike Price K2 (middle)", value=100.0, step=1.0)
            K3 = st.number_input("Strike Price K3 (highest)", value=110.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, K1, maturity, rate, div, sigma)
        df_call_butterfly_option_results = option.call_butterfly(option_position, K1, K2, K3, call_K1_price=None, call_K2_price=None, call_K3_price=None)
        st.subheader("Results")
        st.dataframe(df_call_butterfly_option_results)

        # Afficher les graphiques
        plot_call_butterfly_fig = option.plot_call_butterfly(option_position, spot, K1, K2, K3, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_call_butterfly_fig)

    elif strategy_choice == "Put Butterfly":
        st.subheader("Parameters for Put Butterfly")

        # Utiliser des colonnes pour les inputs
        col1, col2 = st.columns(2)

        with col1:
            option_position = st.selectbox("Position", ["Long", "Short"])
            spot = st.number_input("Spot Price", value=100.0, step=1.0)
            K1 = st.number_input("Strike Price K1 (lowest)", value=90.0, step=1.0)
            K2 = st.number_input("Strike Price K2 (middle)", value=100.0, step=1.0)
            K3 = st.number_input("Strike Price K3 (highest)", value=110.0, step=1.0)

        with col2:
            maturity = st.number_input("Maturity (in years)", value=1.0, step=0.1)
            rate = st.number_input("Risk-free Rate", value=0.05, step=0.1)
            div = st.number_input("Dividend Yield", value=0.02, step=0.1)
            sigma = st.number_input("Volatility", value=0.20, step=0.1)

        # Afficher le df de la stratégie
        option = Option(spot, K1, maturity, rate, div, sigma)
        df_put_butterfly_option_results = option.put_butterfly(option_position, K1, K2, K3, put_K1_price=None, put_K2_price=None, put_K3_price=None)
        st.subheader("Results")
        st.dataframe(df_put_butterfly_option_results)

        # Afficher les graphiques
        plot_put_butterfly_fig = option.plot_put_butterfly(option_position, spot, K1, K2, K3, maturity, rate, div, sigma)
        st.subheader("Payoff & Profit Graphs")
        st.plotly_chart(plot_put_butterfly_fig)

