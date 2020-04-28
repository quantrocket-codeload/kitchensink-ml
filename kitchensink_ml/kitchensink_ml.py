# Copyright 2019 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from moonshot import MoonshotML
from moonshot.commission import PerShareCommission
from quantrocket.fundamental import get_sharadar_fundamentals_reindexed_like
from quantrocket import get_prices
from quantrocket.master import get_securities_reindexed_like

class USStockCommission(PerShareCommission):
    BROKER_COMMISSION_PER_SHARE = 0.005

class TheKitchenSinkML(MoonshotML):

    CODE = "kitchensink-ml"
    DB = "sharadar-us-stk-1d"
    DB_FIELDS = ["Close", "Volume"]
    BENCHMARK_DB = "market-1d"
    SPY_SID = "FIBBG000BDTBL9"
    VIX_SID = "IB13455763"
    TRIN_SID = "IB26718743"
    BENCHMARK = SPY_SID
    DOLLAR_VOLUME_TOP_N_PCT = 60
    DOLLAR_VOLUME_WINDOW = 90
    MODEL = None
    LOOKBACK_WINDOW = 252
    COMMISSION_CLASS = USStockCommission

    def prices_to_features(self, prices):

        closes = prices.loc["Close"]

        features = {}

        print("adding fundamental features")
        self.add_fundamental_features(prices, features)
        print("adding quality features")
        self.add_quality_features(prices, features)

        print("adding price and volume features")
        self.add_price_and_volume_features(prices, features)
        print("adding techical indicator features")
        self.add_technical_indicator_features(prices, features)
        print("adding securities master features")
        self.add_securities_master_features(prices, features)
        print("adding market features")
        self.add_market_features(prices, features)

        # Target to predict: next week return
        one_week_returns = (closes - closes.shift(5)) / closes.shift(5).where(closes.shift(5) > 0)
        targets = one_week_returns.shift(-5)

        return features, targets

    def add_fundamental_features(self, prices, features):
        """
        Fundamental features:

        - Enterprise multiple
        - various quarterly values and ratios
        - various trailing-twelve month values and ratios
        """

        closes = prices.loc["Close"]

        # enterprise multiple
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            fields=["EVEBIT", "EBIT"],
            dimension="ART")
        enterprise_multiples = fundamentals.loc["EVEBIT"]
        ebits = fundamentals.loc["EBIT"]
        # Ignore negative earnings
        enterprise_multiples = enterprise_multiples.where(ebits > 0)
        features["enterprise_multiples_ranks"] = enterprise_multiples.rank(axis=1, pct=True).fillna(0.5)

        # Query quarterly fundamentals
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            dimension="ARQ", # As-reported quarterly reports
            fields=[
                "CURRENTRATIO", # Current ratio
                "DE", # Debt to Equity Ratio
                "PB", # Price to Book Value
                "TBVPS", # Tangible Asset Book Value per Share
                "MARKETCAP",
            ])

        for field in fundamentals.index.get_level_values("Field").unique():
            features["{}_ranks".format(field)] = fundamentals.loc[field].rank(axis=1, pct=True).fillna(0.5)

        # Query trailing-twelve-month fundamentals
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            dimension="ART", # As-reported trailing-twelve-month reports
            fields=[
                "ASSETTURNOVER", # Asset Turnover
                "EBITDAMARGIN", # EBITDA Margin
                "EQUITYAVG", # Average Equity
                "GROSSMARGIN", # Gross Margin
                "NETMARGIN", # Profit Margin
                "PAYOUTRATIO", # Payout Ratio
                "PE", # Price Earnings Damodaran Method
                "PE1", # Price to Earnings Ratio
                "PS", # Price Sales (Damodaran Method)
                "PS1", # Price to Sales Ratio
                "ROA", # Return on Average Assets
                "ROE", # Return on Average Equity
                "ROS", # Return on Sales
            ])

        for field in fundamentals.index.get_level_values("Field").unique():
            features["{}_ranks".format(field)] = fundamentals.loc[field].rank(axis=1, pct=True).fillna(0.5)

    def add_quality_features(self, prices, features):
        """
        Adds quality features, based on the Piotroski F-score.
        """
        closes = prices.loc["Close"]

        # Step 1: query relevant indicators
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            dimension="ART", # As-reported TTM reports
            fields=[
               "ROA", # Return on assets
               "ASSETS", # Total Assets
               "NCFO", # Net Cash Flow from Operations
               "DE", # Debt to Equity Ratio
               "CURRENTRATIO", # Current ratio
               "SHARESWA", # Outstanding shares
               "GROSSMARGIN", # Gross margin
               "ASSETTURNOVER", # Asset turnover
           ])
        return_on_assets = fundamentals.loc["ROA"]
        total_assets = fundamentals.loc["ASSETS"]
        operating_cash_flows = fundamentals.loc["NCFO"]
        leverages = fundamentals.loc["DE"]
        current_ratios = fundamentals.loc["CURRENTRATIO"]
        shares_out = fundamentals.loc["SHARESWA"]
        gross_margins = fundamentals.loc["GROSSMARGIN"]
        asset_turnovers = fundamentals.loc["ASSETTURNOVER"]

        # Step 2: many Piotroski F-score components compare current to previous
        # values, so get DataFrames of previous values

        # Step 2.a: get a boolean mask of the first day of each newly reported fiscal
        # period
        fundamentals = get_sharadar_fundamentals_reindexed_like(
            closes,
            dimension="ARQ", # As-reported quarterly reports
            fields=[
               "REPORTPERIOD"
           ])
        fiscal_periods = fundamentals.loc["REPORTPERIOD"]
        are_new_fiscal_periods = fiscal_periods != fiscal_periods.shift()

        periods_ago = 4

        # this function will be applied sid by sid and returns a Series of
        # earlier fundamentals
        def n_periods_ago(fundamentals_for_sid):
            sid = fundamentals_for_sid.name
            # remove all rows except for new fiscal periods
            new_period_fundamentals = fundamentals_for_sid.where(are_new_fiscal_periods[sid]).dropna()
            # Shift the desired number of periods
            earlier_fundamentals = new_period_fundamentals.shift(periods_ago)
            # Reindex and forward-fill to restore original shape
            earlier_fundamentals = earlier_fundamentals.reindex(fundamentals_for_sid.index, method="ffill")
            return earlier_fundamentals

        previous_return_on_assets = return_on_assets.apply(n_periods_ago)
        previous_leverages = leverages.apply(n_periods_ago)
        previous_current_ratios = current_ratios.apply(n_periods_ago)
        previous_shares_out = shares_out.apply(n_periods_ago)
        previous_gross_margins = gross_margins.apply(n_periods_ago)
        previous_asset_turnovers = asset_turnovers.apply(n_periods_ago)

        # Step 3: calculate F-Score components; each resulting component is a DataFrame
        # of booleans
        have_positive_return_on_assets = return_on_assets > 0
        have_positive_operating_cash_flows = operating_cash_flows > 0
        have_increasing_return_on_assets = return_on_assets > previous_return_on_assets
        total_assets = total_assets.where(total_assets > 0) # avoid DivisionByZero errors
        have_more_cash_flow_than_incomes = operating_cash_flows / total_assets > return_on_assets
        have_decreasing_leverages = leverages < previous_leverages
        have_increasing_current_ratios = current_ratios > previous_current_ratios
        have_no_new_shares = shares_out <= previous_shares_out
        have_increasing_gross_margins = gross_margins > previous_gross_margins
        have_increasing_asset_turnovers = asset_turnovers > previous_asset_turnovers

        # Save each boolean F score component as a feature
        features["have_positive_return_on_assets"] = have_positive_return_on_assets.astype(int)
        features["have_positive_operating_cash_flows"] = have_positive_operating_cash_flows.astype(int)
        features["have_increasing_return_on_assets"] = have_increasing_return_on_assets.astype(int)
        features["have_more_cash_flow_than_incomes"] = have_more_cash_flow_than_incomes.astype(int)
        features["have_decreasing_leverages"] = have_decreasing_leverages.astype(int)
        features["have_increasing_current_ratios"] = have_increasing_current_ratios.astype(int)
        features["have_no_new_shares"] = have_no_new_shares.astype(int)
        features["have_increasing_gross_margins"] = have_increasing_gross_margins.astype(int)
        features["have_increasing_asset_turnovers"] = have_increasing_asset_turnovers.astype(int)

        # Sum the components to get the F-Score and saves the ranks as a feature
        f_scores = (
            have_positive_return_on_assets.astype(int)
            + have_positive_operating_cash_flows.astype(int)
            + have_increasing_return_on_assets.astype(int)
            + have_more_cash_flow_than_incomes.astype(int)
            + have_decreasing_leverages.astype(int)
            + have_increasing_current_ratios.astype(int)
            + have_no_new_shares.astype(int)
            + have_increasing_gross_margins.astype(int)
            + have_increasing_asset_turnovers.astype(int)
        )
        features["f_score_ranks"] = f_scores.rank(axis=1, pct=True).fillna(0.5)

    def add_price_and_volume_features(self, prices, features):
        """
        Price and volume features, or features derived from price and volume:

        - return ranks
        - price level
        - dollar volume rank
        - volatility ranks
        - volatility spikes
        - volume spikes
        """
        closes = prices.loc["Close"]

        # yearly, monthly, weekly, 2-day, daily returns ranks
        one_year_returns = (closes.shift(22) - closes.shift(252)) / closes.shift(252) # exclude most recent month, per classic momentum
        one_month_returns = (closes - closes.shift(22)) / closes.shift(22)
        one_week_returns = (closes - closes.shift(5)) / closes.shift(5)
        two_day_returns = (closes - closes.shift(2)) / closes.shift(2)
        one_day_returns = closes.pct_change()
        features["1yr_returns_ranks"] = one_year_returns.rank(axis=1, pct=True).fillna(0.5)
        features["1mo_returns_ranks"] = one_month_returns.rank(axis=1, pct=True).fillna(0.5)
        features["1wk_returns_ranks"] = one_week_returns.rank(axis=1, pct=True).fillna(0.5)
        features["2d_returns_ranks"] = two_day_returns.rank(axis=1, pct=True).fillna(0.5)
        features["1d_returns_ranks"] = one_day_returns.rank(axis=1, pct=True).fillna(0.5)

        # whether returns were positive
        features["last_1year_was_positive"] = (one_year_returns > 0).astype(int)
        features["last_1month_was_positive"] = (one_month_returns > 0).astype(int)
        features["last_1week_was_positive"] = (one_week_returns > 0).astype(int)
        features["last_2day_was_positive"] = (two_day_returns > 0).astype(int)
        features["last_1day_was_positive"] = (one_day_returns > 0).astype(int)

        # price level
        features["price_below_10"] = closes < 10
        features["price_below_2"] = closes < 2

        # dollar volume ranks
        volumes = prices.loc["Volume"]
        avg_dollar_volumes = (closes * volumes).rolling(63).mean()
        features["dollar_volume_ranks"] = avg_dollar_volumes.rank(axis=1, ascending=True, pct=True).fillna(0.5)

        # quarterly volatility ranks
        quarterly_stds = closes.pct_change().rolling(window=63).std()
        features["quaterly_std_ranks"] = quarterly_stds.rank(axis=1, pct=True).fillna(0.5)

        # volatility spikes
        volatility_1d_vs_quarter = closes.pct_change().abs() / quarterly_stds.where(quarterly_stds > 0)
        features["2std_volatility_spike"] = (volatility_1d_vs_quarter >= 2).astype(int)
        features["volatility_spike_ranks"] = volatility_1d_vs_quarter.rank(axis=1, pct=True).fillna(0.5)

        # volume spike
        avg_volumes = volumes.rolling(window=63).mean()
        volume_1d_vs_quarter = volumes / avg_volumes.where(avg_volumes > 0)
        features["2x_volume_spike"] = (volume_1d_vs_quarter >= 2).astype(int)
        features["volume_spike_ranks"] = volume_1d_vs_quarter.rank(axis=1, pct=True).fillna(0.5)

    def add_technical_indicator_features(self, prices, features):
        """
        Various technical indicators:

        - Bollinger bands
        - RSI
        - Stochastic oscillator
        - Money Flow Index
        """
        closes = prices.loc["Close"]

        # relative position within Bollinger Bands (0 = at or below lower band, 1 = at or above upper band)
        mavgs = closes.rolling(20).mean()
        stds  = closes.rolling(20).std()
        upper_bands = mavgs + (stds * 2)
        lower_bands = mavgs - (stds * 2)
        # Winsorize at upper and lower bands
        winsorized_closes = closes.where(closes > lower_bands, lower_bands).where(closes < upper_bands, upper_bands)
        features["close_vs_bbands"] = (winsorized_closes - lower_bands) / (winsorized_closes - lower_bands)

        # RSI (0-1)
        returns = closes.diff()
        avg_gains = returns.where(returns > 0).rolling(window=14, min_periods=1).mean()
        avg_losses = returns.where(returns < 0).abs().rolling(window=14, min_periods=1).mean()
        relative_strengths = avg_gains / avg_losses.where(avg_losses != 0)
        features["RSI"] = 1 - (1 / (1 + relative_strengths.fillna(0.5)))

        # Stochastic oscillator (0-1)
        highest_highs = closes.rolling(window=14).max()
        lowest_lows = closes.rolling(window=14).min()
        features["stochastic"] = (closes - lowest_lows) / (highest_highs - lowest_lows)

        # Money flow (similar to RSI but volume-weighted) (0-1)
        money_flows = closes * prices.loc["Volume"]
        positive_money_flows = money_flows.where(returns > 0).rolling(window=14, min_periods=1).sum()
        negative_money_flows = money_flows.where(returns < 0).rolling(window=14, min_periods=1).sum()
        money_flow_ratios = positive_money_flows / negative_money_flows.where(negative_money_flows > 0)
        features["money_flow"] = 1 - (1 / (1 + money_flow_ratios.fillna(0.5)))

    def add_securities_master_features(self, prices, features):
        """
        Features from the securities master:

        - ADR?
        - sector
        """
        closes = prices.loc["Close"]

        securities = get_securities_reindexed_like(closes, fields=["sharadar_Category", "sharadar_Sector"])

        # Is it an ADR?
        categories = securities.loc["sharadar_Category"]
        unique_categories = categories.iloc[0].unique()
        # this dataset includes several ADR classifications, all of which start with "ADR "
        features["are_adrs"] = categories.isin([cat for cat in unique_categories if cat.startswith("ADR ")]).astype(int)

        # Which sector? (sectors must be one-hot encoded - see usage guide for more)
        sectors = securities.loc["sharadar_Sector"]
        for sector in sectors.stack().unique():
            features["sector_{}".format(sector)] = (sectors == sector).astype(int)

    def add_market_features(self, prices, features):
        """
        Market price, volatility, and breadth, some of which are queried from a
        database and some of which are calculated from the Sharadar data:

        - whether S&P 500 is above or below its 200-day moving average
        - where VIX falls within the range of 12 - 30
        - where 10-day NYSE TRIN falls within the range of 0.5 to 2
        - McClellan oscillator
        - Hindenburg Omen
        """
        closes = prices.loc["Close"]

        # Get prices for SPY, VIX, TRIN-NYSE
        market_prices = get_prices(self.BENCHMARK_DB,
                                   fields="Close",
                                   start_date=closes.index.min(),
                                   end_date=closes.index.max())
        market_closes = market_prices.loc["Close"]

        # Is S&P above its 200-day?
        spy_closes = market_closes[self.SPY_SID]
        spy_200d_mavg = spy_closes.rolling(200).mean()
        spy_above_200d = (spy_closes > spy_200d_mavg).astype(int)
        # Must reindex like closes in case indexes differ
        spy_above_200d = spy_above_200d.reindex(closes.index, method="ffill")
        features["spy_above_200d"] = closes.apply(lambda x: spy_above_200d)

        # VIX and TRIN don't go back as far as Sharadar data, so we may need a filler DataFrame
        fillers = pd.DataFrame(0.5, index=closes.index, columns=closes.columns)

        # Where does VIX fall within the range of 12-30?
        try:
            vix = market_closes[self.VIX_SID]
        except KeyError:
            features["vix"] = fillers
        else:
            vix_high = 30
            vix_low = 12
            # Winsorize VIX
            vix = vix.where(vix > vix_low, vix_low).where(vix < vix_high, vix_high)
            vix_as_pct = (vix - vix_low) / (vix_high - vix_low)
            vix_as_pct = vix_as_pct.reindex(closes.index, method="ffill")
            features["vix"] = closes.apply(lambda x: vix_as_pct).fillna(0.5)

        # Where does NYSE TRIN fall within the range of 0.5-2?
        try:
            trin = market_closes[self.TRIN_SID]
        except KeyError:
            features["trin"] = fillers
        else:
            trin = trin.rolling(window=10).mean()
            trin_high = 2
            trin_low = 0.5
            # Winsorize TRIN
            trin = trin.where(trin > trin_low, trin_low).where(trin < trin_high, trin_high)
            trin_as_pct = (trin - trin_low) / (trin_high - trin_low)
            trin_as_pct = trin_as_pct.reindex(closes.index, method="ffill")
            features["trin"] = closes.apply(lambda x: trin_as_pct).fillna(0.5)

        # McClellan oscillator
        total_issues = closes.count(axis=1)
        returns = closes.pct_change()
        advances = returns.where(returns > 0).count(axis=1)
        declines = returns.where(returns < 0).count(axis=1)
        net_advances = advances - declines
        pct_net_advances = net_advances / total_issues
        ema_19 = pct_net_advances.ewm(span=19).mean()
        ema_39 = pct_net_advances.ewm(span=39).mean()
        mcclellan_oscillator = (ema_19 - ema_39) * 10
        # Winsorize at 50 and -50
        mcclellan_oscillator = mcclellan_oscillator.where(mcclellan_oscillator < 50, 50).where(mcclellan_oscillator > -50, -50)
        features["mcclellan_oscillator"] = closes.apply(lambda x: mcclellan_oscillator).fillna(0)

        # Hindenburg omen (and new 52-week highs/lows)
        one_year_highs = closes.rolling(window=252).max()
        one_year_lows = closes.rolling(window=252).min()
        new_one_year_highs = (closes > one_year_highs.shift()).astype(int)
        new_one_year_lows = (closes < one_year_lows.shift()).astype(int)
        features["new_one_year_highs"] = new_one_year_highs
        features["new_one_year_lows"] = new_one_year_lows
        pct_one_year_highs = new_one_year_highs.sum(axis=1) / total_issues
        pct_one_year_lows = new_one_year_lows.sum(axis=1) / total_issues
        hindenburg_omens = (pct_one_year_highs > 0.028) & (pct_one_year_lows > 0.028) & (spy_closes > spy_closes.shift(50))
        # Omen lasts for 30 days
        hindenburg_omens = hindenburg_omens.where(hindenburg_omens).fillna(method="ffill", limit=30).fillna(False).astype(int)
        hindenburg_omens = hindenburg_omens.reindex(closes.index, method="ffill")
        features["hindenburg_omens"] = closes.apply(lambda x: hindenburg_omens)

    def predictions_to_signals(self, predictions, prices):
        closes = prices.loc["Close"]
        volumes = prices.loc["Volume"]
        avg_dollar_volumes = (closes * volumes).rolling(self.DOLLAR_VOLUME_WINDOW).mean()
        dollar_volume_ranks = avg_dollar_volumes.rank(axis=1, ascending=False, pct=True)
        have_adequate_dollar_volumes = dollar_volume_ranks <= (self.DOLLAR_VOLUME_TOP_N_PCT/100)

        # Save the predictions and prices so we can analyze them
        self.save_to_results("Prediction", predictions)
        self.save_to_results("Close", closes)
        self.save_to_results("Volume", volumes)

        # Buy (sell) stocks with best (worst) predicted return
        have_best_predictions = predictions.where(have_adequate_dollar_volumes).rank(ascending=False, axis=1) <= 10
        have_worst_predictions = predictions.where(have_adequate_dollar_volumes).rank(ascending=True, axis=1) <= 10
        signals = have_best_predictions.astype(int).where(have_best_predictions, -have_worst_predictions.astype(int).where(have_worst_predictions, 0))
        return signals

    def signals_to_target_weights(self, signals, prices):
        # Allocate equal weights
        daily_signal_counts = signals.abs().sum(axis=1)
        weights = signals.div(daily_signal_counts, axis=0).fillna(0)

        # Rebalance weekly
        # Resample daily to weekly, taking the first day's signal
        # For pandas offset aliases, see https://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        weights = weights.resample("W").first()
        # Reindex back to daily and fill forward
        weights = weights.reindex(prices.loc["Close"].index, method="ffill")

        return weights

    def target_weights_to_positions(self, weights, prices):
        # Enter the position the day after the signal
        return weights.shift()

    def positions_to_gross_returns(self, positions, prices):

        closes = prices.loc["Close"]
        gross_returns = closes.pct_change() * positions.shift()
        return gross_returns
