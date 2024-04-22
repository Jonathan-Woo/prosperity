from datamodel import (
    Listing,
    Observation,
    ConversionObservation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Trade,
    TradingState,
    Symbol,
    Time,
    Product,
    Position,
    UserId,
    ObservationValue,
)
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import statistics as stats
import math
import jsonpickle
import json


class TraderDataDTO:
    def __init__(self):
        self._prices = {}
        self._implied_vol_historical = pd.Series(dtype='float64')

    def accept_implied_vol(self, implied_vol):
        self._implied_vol_historical = pd.concat([self._implied_vol_historical, pd.Series(implied_vol)], ignore_index=True)

    def get_implied_vol_historical(self):
        return self._implied_vol_historical

    def accept_product_price(self, symbol, price):
        if symbol not in self._prices:
            self._prices[symbol] = []

        self._prices[symbol].append(price)

    def drop_product_price(self, symbol):
        if symbol not in self._prices:
            return
        self._prices[symbol].pop(0)

    def get_product_price(self, symbol):
        return self._prices[symbol]

    def to_json(self):
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json_string: str):
        if json_string is None or json_string == "":
            return TraderDataDTO()

        return jsonpickle.decode(json_string)


class Trader:

    def __init__(self):
        self.result = {}
        self.traderData = None

        self.amethysts_params = {"position_limit": 20}

        self.starfruit_params = {
            "position_limit": 20,
            "LR_coefs": [
                0.19540403,
                0.20473853,
                0.2620262,
                0.3339295,
                29.42326089,
            ],
        }

        self.basket_params = {
            'diff_historical': pd.Series(dtype='float64'),
            'notable_times': {},
            'swings': 0
        }

        self.orchids_params = {"position_limit": 100, "size_per_price": 33}

        self.rose_params = {
            "position_limit": 60,
            "LR_coefs": [
                -0.01013357020076396,
                0.0015946678488045762,
                0.015126217751578666,
                0.9924815213735724,
                13.463289386777129,
            ],
        }

        self.strawberry_params = {
            "position_limit": 350,
            "LR_coefs": [
                -0.011226934464677521,
                0.031696501288656975,
                0.11234473932527322,
                0.8668941528673054,
                1.1747772611294447,
            ],
        }

        self.chocolate_params = {
            "position_limit": 250,
            "LR_coefs": [
                -0.00730403985266707,
                0.004748038433910769,
                0.01861020684263674,
                0.9831564961367624,
                6.289025217299543,
            ],
        }

        self.coco_params = {
            'coconut_prices': pd.Series(dtype='float64')
        }

    def update_stored_price(self, product, price, window_size):
        """
        Updates the stored price of a product in the trader data.
        """
        self.traderData.accept_product_price(product, price)
        if len(self.traderData.get_product_price(product)) > window_size:
            self.traderData.drop_product_price(product)
            assert len(self.traderData.get_product_price(product)) == window_size

    def linear_predict_price(self, lr_coefs, product):
        """
        Predicts the price of a product using the LR coefs and stored past prices. Rounds to whole prices.
        """
        assert len(lr_coefs) == len(self.traderData.get_product_price(product)) + 1
        expected_price = np.array(lr_coefs).dot(
            np.array(self.traderData.get_product_price(product) + [1])
        )
        return round(expected_price)

    def starfruit(self):
        if "STARFRUIT" in self.state.position:
            cur_position = self.state.position["STARFRUIT"]
        else:
            cur_position = 0
        # Position limits dictate that current position + aggregated buys must not exceed position limit
        # and current position - aggregated sells must not exceed position limit
        # So, keep track of the new long and short positions
        new_long_position, new_short_position = cur_position, cur_position
        position_limit = self.starfruit_params["position_limit"]
        # LR coefs includes a bias/constant term. So, the window size is the length of the coefs - 1.
        window_size = len(self.starfruit_params["LR_coefs"]) - 1
        self.result["STARFRUIT"] = []

        highest_market_bid = max(self.state.order_depths["STARFRUIT"].buy_orders.keys())
        lowest_market_ask = min(self.state.order_depths["STARFRUIT"].sell_orders.keys())
        mid_price = (highest_market_bid + lowest_market_ask) / 2

        # Update the trader data with the latest price
        self.update_stored_price("STARFRUIT", mid_price, window_size)

        # Start trading after the window of prices has been filled.
        if len(self.traderData.get_product_price("STARFRUIT")) < window_size:
            return

        # Compute expected price
        expected_price = self.linear_predict_price(
            self.starfruit_params["LR_coefs"], "STARFRUIT"
        )

        # Market orders
        # Buy undervalued
        # Buy at value if we are short
        for ask, qty in self.state.order_depths["STARFRUIT"].sell_orders.items():
            if (
                (ask < expected_price) or ((cur_position < 0) and (ask == mid_price))
            ) and new_long_position < position_limit:
                order_qty = min(-qty, position_limit - new_long_position)
                new_long_position += order_qty
                self.result["STARFRUIT"].append(Order("STARFRUIT", ask, order_qty))

        # Sell overvalued
        # Sell at value if we are long
        for bid, qty in self.state.order_depths["STARFRUIT"].buy_orders.items():
            if (
                (bid > expected_price) or ((cur_position > 0) and (bid == mid_price))
            ) and new_short_position > -position_limit:
                order_qty = min(qty, position_limit + new_short_position)
                new_short_position -= order_qty
                self.result["STARFRUIT"].append(Order("STARFRUIT", bid, -order_qty))

        # Market making
        # Setup orders to tighten the spread
        # The price must be at least 1 away from the current market price or the expected price
        tighter_bid = highest_market_bid + 1
        tighter_ask = lowest_market_ask - 1

        if new_long_position < position_limit:
            order_qty = position_limit - new_long_position
            new_long_position += order_qty
            order_price = min(tighter_bid, expected_price - 1)
            self.result["STARFRUIT"].append(Order("STARFRUIT", order_price, order_qty))
        if new_short_position > -position_limit:
            order_qty = position_limit + new_short_position
            new_short_position -= order_qty
            order_price = max(tighter_ask, expected_price + 1)
            self.result["STARFRUIT"].append(Order("STARFRUIT", order_price, -order_qty))

    def amethyst(self):
        if "AMETHYSTS" in self.state.position:
            cur_position = self.state.position["AMETHYSTS"]
        else:
            cur_position = 0
        # Position limits dictate that current position + aggregated buys must not exceed position limit
        # and current position - aggregated sells must not exceed position limit
        # So, keep track of the new long and short positions
        new_long_position, new_short_position = cur_position, cur_position
        mean_price = 10000
        position_limit = self.amethysts_params["position_limit"]
        self.result["AMETHYSTS"] = []

        # Market orders
        # Buy undervalued (< 10000)
        # Buy at value (= 10000) if we are short
        for ask, qty in self.state.order_depths["AMETHYSTS"].sell_orders.items():
            if (
                (ask < mean_price) or ((cur_position < 0) and (ask == mean_price))
            ) and new_long_position < position_limit:
                order_qty = min(-qty, position_limit - new_long_position)
                new_long_position += order_qty
                self.result["AMETHYSTS"].append(Order("AMETHYSTS", ask, order_qty))
        # Sell overvalued (> 10000)
        # Sell at value (= 10000) if we are long
        for bid, qty in self.state.order_depths["AMETHYSTS"].buy_orders.items():
            if (
                (bid > mean_price) or ((cur_position > 0) and (bid == mean_price))
            ) and new_short_position > -position_limit:
                order_qty = min(qty, position_limit + new_short_position)
                new_short_position -= order_qty
                self.result["AMETHYSTS"].append(Order("AMETHYSTS", bid, -order_qty))

        # Market making
        # Setup orders to tighten the spread
        highest_market_bid = max(self.state.order_depths["AMETHYSTS"].buy_orders.keys())
        lowest_market_ask = min(self.state.order_depths["AMETHYSTS"].sell_orders.keys())

        tighter_bid = highest_market_bid + 1
        tighter_ask = lowest_market_ask - 1

        # Depending on the current position, adjust the spread.
        if cur_position < 0 and new_long_position < position_limit:
            order_price = min(tighter_bid + 1, mean_price - 1)
        elif cur_position > 15 and new_long_position < position_limit:
            order_price = min(tighter_bid - 1, mean_price - 1)
        else:
            order_price = min(tighter_bid, mean_price - 1)
        order_qty = position_limit - new_long_position
        new_long_position += order_qty
        self.result["AMETHYSTS"].append(Order("AMETHYSTS", order_price, order_qty))

        if cur_position > 0 and new_short_position > -position_limit:
            order_price = max(tighter_ask - 1, mean_price + 1)
        elif cur_position < -15 and new_short_position > -position_limit:
            order_price = max(tighter_ask + 1, mean_price + 1)
        else:
            order_price = max(tighter_ask, mean_price + 1)
        order_qty = position_limit + new_short_position
        new_short_position -= order_qty
        self.result["AMETHYSTS"].append(Order("AMETHYSTS", order_price, -order_qty))

    def orchids(self):
        if "ORCHIDS" in self.state.position:
            cur_position = self.state.position["ORCHIDS"]
        else:
            cur_position = 0
        new_long_position, new_short_position = cur_position, cur_position
        position_limit = self.orchids_params["position_limit"]
        margin_of_safety = 0
        size_per_price = self.orchids_params["size_per_price"]
        self.result["ORCHIDS"] = []

        highest_market_bid = max(self.state.order_depths["ORCHIDS"].buy_orders.keys())
        lowest_market_ask = min(self.state.order_depths["ORCHIDS"].sell_orders.keys())

        # Calculating net purchase or selling price
        # through conversions at current timestep,
        # as an estimate for next timestep

        conv_ask = self.state.observations.conversionObservations["ORCHIDS"].askPrice
        conv_bid = self.state.observations.conversionObservations["ORCHIDS"].bidPrice
        transportFees = self.state.observations.conversionObservations[
            "ORCHIDS"
        ].transportFees
        exportTariff = self.state.observations.conversionObservations[
            "ORCHIDS"
        ].exportTariff
        importTariff = self.state.observations.conversionObservations[
            "ORCHIDS"
        ].importTariff
        sunlight = self.state.observations.conversionObservations["ORCHIDS"].sunlight
        humidity = self.state.observations.conversionObservations["ORCHIDS"].humidity

        # Compute the net price if we were to trade with the south.
        conversion_purchase_price = conv_ask + importTariff + transportFees
        conversion_sell_price = conv_bid - exportTariff - transportFees

        # Parameters
        lowest_we_would_sell = round(conversion_purchase_price + margin_of_safety)
        highest_we_would_buy = round(conversion_sell_price - margin_of_safety)

        # Buy from South (next timestep) and sell in normal market (now)
        if conversion_purchase_price < highest_market_bid:
            for bid, qty in self.state.order_depths["ORCHIDS"].buy_orders.items():
                # If bid price facilitates arbitrage
                if (
                    conversion_purchase_price < bid
                    and new_short_position > -position_limit
                ):
                    order_qty = min(qty, position_limit + new_short_position)
                    new_short_position -= order_qty
                    self.result["ORCHIDS"].append(Order("ORCHIDS", bid, -order_qty))

            # since we are selling to arb, let's try to put limit asks as well
            diff = math.floor(highest_market_bid - conversion_purchase_price)
            if diff > 1:
                qty_per_price = (position_limit + new_short_position) // diff
                for i in range(diff):
                    if new_short_position > -position_limit:
                        order_qty = min(
                            qty_per_price, position_limit + new_short_position
                        )
                        new_short_position -= order_qty
                        self.result["ORCHIDS"].append(
                            Order("ORCHIDS", highest_market_bid - i, -order_qty)
                        )

        # Sell to South (next timestep) and buy in normal market (now)
        if conversion_sell_price > lowest_market_ask:
            for ask, qty in self.state.order_depths["ORCHIDS"].sell_orders.items():
                # If ask price facilitates arbitrage
                if conversion_sell_price > ask and new_long_position < position_limit:
                    order_qty = min(-qty, position_limit - new_long_position)
                    new_long_position += order_qty
                    self.result["ORCHIDS"].append(Order("ORCHIDS", ask, order_qty))

            # since we are buying to arb, let's try to put limit bids as well
            diff = math.floor(conversion_sell_price - lowest_market_ask)
            if diff > 1:
                qty_per_price = (position_limit - new_long_position) // diff
                for i in range(diff):
                    if new_long_position < position_limit:
                        order_qty = min(
                            qty_per_price, position_limit - new_long_position
                        )
                        new_long_position += order_qty
                        self.result["ORCHIDS"].append(
                            Order("ORCHIDS", lowest_market_ask + i, order_qty)
                        )

        # Market make when no immediate arb available
        if (
            conversion_purchase_price > highest_market_bid
            and conversion_sell_price < lowest_market_ask
        ):
            # placing limit asks @ purchase + 1 and above
            max_iterations = (
                (position_limit + new_short_position) // size_per_price
            ) + 1
            for i in range(1, max_iterations + 1):
                if new_short_position > -position_limit:
                    order_qty = min(size_per_price, position_limit + new_short_position)
                    new_short_position -= order_qty
                    self.result["ORCHIDS"].append(
                        Order("ORCHIDS", lowest_we_would_sell + i, -order_qty)
                    )

            # # placing limit bids @ sale - 1 and below
            max_iterations = (
                (position_limit - new_long_position) // size_per_price
            ) + 1
            for i in range(1, max_iterations + 1):
                if position_limit > new_long_position:
                    order_qty = min(size_per_price, position_limit - new_long_position)
                    new_long_position += order_qty
                    self.result["ORCHIDS"].append(
                        Order("ORCHIDS", highest_we_would_buy - i, order_qty)
                    )

    def basket(self):
        try:
            pos_limit = {
                "CHOCOLATE": 250,
                "STRAWBERRIES": 350,
                "ROSES": 60,
                "GIFT_BASKET": 60,
            }
            curr_pos = {
                "CHOCOLATE": self.state.position.get("CHOCOLATE", 0),
                "ROSES": self.state.position.get("ROSES", 0),
                "STRAWBERRIES": self.state.position.get("STRAWBERRIES", 0),
                "GIFT_BASKET": self.state.position.get("_GIFT_BASKET", 0),
            }

            margin_of_safety = 40
            size_per_price = 10

            orders_basket = []

            """MARKET TAKING"""
            if "CHOCOLATE" in self.state.order_depths:
                bids_choc = sorted(
                    list(self.state.order_depths["CHOCOLATE"].buy_orders.items()),
                    key=lambda x: x[0],
                    reverse=True,
                )
                asks_choc = sorted(
                    list(self.state.order_depths["CHOCOLATE"].sell_orders.items()),
                    key=lambda x: x[0],
                    reverse=False,
                )
                best_bid_choc = bids_choc[0][0]
                best_ask_choc = asks_choc[0][0]
                mid_choc = (best_ask_choc + best_bid_choc) / 2

            if "STRAWBERRIES" in self.state.order_depths:
                bids_berry = sorted(
                    list(self.state.order_depths["STRAWBERRIES"].buy_orders.items()),
                    key=lambda x: x[0],
                    reverse=True,
                )
                asks_berry = sorted(
                    list(self.state.order_depths["STRAWBERRIES"].sell_orders.items()),
                    key=lambda x: x[0],
                    reverse=False,
                )
                best_bid_berry = bids_berry[0][0]
                best_ask_berry = asks_berry[0][0]
                mid_berry = (best_ask_berry + best_bid_berry) / 2

            if "ROSES" in self.state.order_depths:
                bids_roses = sorted(
                    list(self.state.order_depths["ROSES"].buy_orders.items()),
                    key=lambda x: x[0],
                    reverse=True,
                )
                asks_roses = sorted(
                    list(self.state.order_depths["ROSES"].sell_orders.items()),
                    key=lambda x: x[0],
                    reverse=False,
                )
                best_bid_roses = bids_roses[0][0]
                best_ask_roses = asks_roses[0][0]
                mid_roses = (best_ask_roses + best_bid_roses) / 2

            if "GIFT_BASKET" in self.state.order_depths:
                bids_basket = sorted(
                    list(self.state.order_depths["GIFT_BASKET"].buy_orders.items()),
                    key=lambda x: x[0],
                    reverse=True,
                )
                asks_basket = sorted(
                    list(self.state.order_depths["GIFT_BASKET"].sell_orders.items()),
                    key=lambda x: x[0],
                    reverse=False,
                )
                best_bid_basket = bids_basket[0][0]

                best_ask_basket = asks_basket[0][0]
                mid_basket = (best_bid_basket + best_ask_basket) / 2

            # basket_act = 6 * asks_berry[0][0] + 4 * asks_choc[0][0] + asks_roses[0][0]

            nav = 6 * mid_berry + 4 * mid_choc + mid_roses

            diff_historical = self.basket_params['diff_historical']
            diff = mid_basket - nav
            diff_historical = pd.concat([diff_historical, pd.Series(diff)], ignore_index=True)
            self.basket_params['diff_historical'] = diff_historical

            factor = 1
            size_per_take = 12

            if "GIFT_BASKET" in self.state.position:
                position = self.state.position["GIFT_BASKET"]
            else:
                position = 0

            if self.state.timestamp > 50000:
                std = stats.stdev(diff_historical)
                avg = stats.mean(diff_historical)
                if diff > avg + factor * std:
                    "short"
                    # print("short signal")
                    position_limit = 60 + position
                    taken = 0
                    "MARKET ORDERS"
                    for bid in bids_basket:
                        # If we have inventory space
                        if position_limit > 0:
                            if taken < size_per_take:
                                order_quantity = min(bid[1], position_limit, size_per_take)
                                position_limit -= order_quantity
                                taken += order_quantity
                                orders_basket.append(Order('GIFT_BASKET', bid[0], -order_quantity))
                        else:
                            break

                if diff < avg - factor * std:
                    # print("long signal")
                    "long"

                    position_limit = 60 - position
                    taken = 0
                    "MARKET ORDERS"
                    for ask in asks_basket:
                        # If we have inventory space
                        if position_limit > 0:
                            if taken < size_per_take:
                                order_quantity = min(-ask[1], position_limit, size_per_take)
                                position_limit -= order_quantity
                                taken += order_quantity
                                orders_basket.append(Order('GIFT_BASKET', ask[0], order_quantity))
                                # print(f"best ask is {asks_basket[0][0]} for a qty of {asks_basket[0][1]}. We are buying for {ask[0]} at qty {order_quantity}.")
                            else:
                                break
                        else:
                            break

                self.result["GIFT_BASKET"] = orders_basket
        except:
            pass

    def coconuts(self):
        try:
            orders_coconut = []
            orders_coupon = []
            if "COCONUT" in self.state.order_depths:
                COCONUT_bids = sorted(list(self.state.order_depths["COCONUT"].buy_orders.items()), key=lambda x: x[0], reverse=True)
                COCONUT_asks = sorted(list(self.state.order_depths["COCONUT"].sell_orders.items()), key=lambda x: x[0], reverse=False)
                best_COCONUT_bid = COCONUT_bids[0][0]
                best_COCONUT_ask = COCONUT_asks[0][0]
                COCONUT_midpoint = (best_COCONUT_ask + best_COCONUT_bid)/2

            # Get current position
            if "COCONUT" in self.state.position:
                COCONUT_position = self.state.position["COCONUT"]
            else:
                COCONUT_position = 0

            if "COCONUT_COUPON" in self.state.order_depths:
                COCONUT_COUPON_bids = sorted(list(self.state.order_depths["COCONUT_COUPON"].buy_orders.items()), key=lambda x: x[0], reverse=True)
                COCONUT_COUPON_asks = sorted(list(self.state.order_depths["COCONUT_COUPON"].sell_orders.items()), key=lambda x: x[0], reverse=False)
                best_COCONUT_COUPON_bid = COCONUT_COUPON_bids[0][0]
                best_COCONUT_COUPON_ask = COCONUT_COUPON_asks[0][0]
                COCONUT_COUPON_midpoint = (best_COCONUT_COUPON_ask + best_COCONUT_COUPON_bid)/2

            # Get current position
            if "COCONUT_COUPON" in self.state.position:
                COCONUT_COUPON_position = self.state.position["COCONUT_COUPON"]
            else:
                COCONUT_COUPON_position = 0

            market_price = COCONUT_COUPON_midpoint
            S = COCONUT_midpoint
            K = 10000
            t = 250
            r = 0

            def normal_cdf(x):
                t = 1 / (1 + 0.2316419 * abs(x))
                y = (0.319381530 + (-0.356563782 + (1.781477937 + (-1.821255978 + 1.330274429 * t) * t) * t) * t) * math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
                return 0.5 * (1 + math.erf(x / math.sqrt(2))) if x >= 0 else 0.5 * (1 - math.erf(-x / math.sqrt(2)))

            def black_scholes_price(S, K, t, r, sigma, option_type='call'):
                d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
                d2 = d1 - sigma * np.sqrt(t)
                if option_type == 'call':
                    price = S * normal_cdf(d1) - K * math.exp(-r * t) * normal_cdf(d2)
                else:
                    price = K * math.exp(-r * t) * normal_cdf(-d2) - S * normal_cdf(-d1)
                return price

            def implied_volatility(market_price, S, K, t, r, initial_guess=0.01012, tol=1e-6, max_iter=100):
                sigma = initial_guess
                for _ in range(max_iter):
                    price = black_scholes_price(S, K, t, r, sigma)
                    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
                    vega = S * np.sqrt(t) * normal_cdf(d1)
                    diff = price - market_price
                    if abs(diff) < tol:
                        return sigma
                    sigma -= diff / vega
                raise ValueError("Implied volatility calculation did not converge.")

            implied_vol = implied_volatility(market_price, S, K, t, r)
            self.traderData.accept_implied_vol(implied_vol)

            # coconut_prices = self.coco_params['coconut_prices']
            # coconut_prices = pd.concat([coconut_prices, pd.Series(COCONUT_midpoint)], ignore_index=True)
            # self.coco_params['coconut_prices'] = coconut_prices

            size_per_take_coco = 300
            size_per_take_coupon = 20

            if 8000 < self.state.timestamp < 999000:
                true_vol = stats.mean(self.traderData.get_implied_vol_historical())
                # differenced_coconut_prices = coconut_prices.diff()
                # true_vol = np.std(differenced_coconut_prices)

                if implied_vol < true_vol:
                    """ Signal to long coupons"""
                    position_limit = 600 - COCONUT_COUPON_position
                    taken = 0
                    # For placing an order at best current market taking price
                    # order_quantity = min(-COCONUT_COUPON_asks[0][1], position_limit)
                    # position_limit -= order_quantity
                    # orders_coupon.append(Order('COCONUT_COUPON', COCONUT_COUPON_asks[0][0], order_quantity))

                    ### For walking the book with market orders up to qty size_per_take
                    for ask in COCONUT_COUPON_asks:
                        # If we have inventory space
                        if position_limit > 0:
                            if taken < size_per_take_coupon:
                                order_quantity = min(-ask[1], position_limit, size_per_take_coupon)
                                position_limit -= order_quantity
                                taken += order_quantity
                                orders_coupon.append(Order('COCONUT_COUPON', ask[0], order_quantity))
                        else:
                            break

                    ### For placing limit order undercutting the book for qty up to size_per_take
                    if position_limit > 0:
                        order_quantity = min(position_limit, size_per_take_coupon)
                        position_limit -= order_quantity
                        orders_coupon.append(Order('COCONUT_COUPON', best_COCONUT_COUPON_bid + 1, order_quantity))

                else:
                    """ Signal to short coupons """
                    position_limit = 600 + COCONUT_COUPON_position
                    taken = 0

                    ### For placing an order at best current market taking price
                    # order_quantity = min(COCONUT_COUPON_bids[0][1], position_limit)
                    # position_limit -= order_quantity
                    # orders_coupon.append(Order('COCONUT_COUPON', COCONUT_COUPON_bids[0][0], -order_quantity))

                    ### For walking the book with market orders up to qty size_per_take
                    for bid in COCONUT_COUPON_bids:
                        # If we have inventory space
                        if position_limit > 0:
                            if taken < size_per_take_coupon:
                                order_quantity = min(bid[1], position_limit, size_per_take_coupon)
                                position_limit -= order_quantity
                                taken += order_quantity
                                orders_coupon.append(Order('COCONUT_COUPON', bid[0], -order_quantity))
                        else:
                            break

                    ### For placing limit order undercutting the book for qty up to size_per_take
                    if position_limit > 0:
                        order_quantity = min(position_limit, size_per_take_coupon)
                        position_limit -= order_quantity
                        orders_coupon.append(Order('COCONUT_COUPON', best_COCONUT_COUPON_ask - 1, -order_quantity))

            else:
                # Emptying out our position slowly before end of day
                # Coconuts
                if COCONUT_position > 0:
                    # Shorting coconuts to offset position
                    "MARKET ORDERS"
                    order_quantity = min(COCONUT_bids[0][1], COCONUT_position)
                    orders_coconut.append(Order('COCONUT', COCONUT_bids[0][0], -order_quantity))
                else:
                    # Longing coconuts to offset position
                    "MARKET ORDERS"
                    order_quantity = min(-COCONUT_asks[0][1], -COCONUT_position)
                    orders_coconut.append(Order('COCONUT', COCONUT_asks[0][0], order_quantity))
                # Coupons
                if COCONUT_COUPON_position > 0:
                    # Shorting coupons to offset position
                    "MARKET ORDERS"
                    order_quantity = min(COCONUT_COUPON_bids[0][1], COCONUT_COUPON_position)
                    orders_coupon.append(Order('COCONUT_COUPON', COCONUT_COUPON_bids[0][0], -order_quantity))
                else:
                    # Longing coupons to offset position
                    "MARKET ORDERS"
                    order_quantity = min(-COCONUT_COUPON_asks[0][1], -COCONUT_COUPON_position)
                    orders_coupon.append(Order('COCONUT_COUPON', COCONUT_COUPON_asks[0][0], order_quantity))

            self.result["COCONUT_COUPON"] = orders_coupon

        except:
            pass

    def run(self, state: TradingState):
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.state = state
        self.traderData = TraderDataDTO.from_json(state.traderData)
        self.result = {}

        self.amethyst()
        self.starfruit()
        self.orchids()
        # For the orchids' strategy, convert all accrued positions.
        position = 0
        if "ORCHIDS" in state.position:
            position = state.position["ORCHIDS"]
        self.basket()

        self.coconuts()

        # print(self.state.observations.conversionObservations)

        conversions = -position
        traderData = TraderDataDTO.to_json(self.traderData)
        logger.flush(state, self.result, conversions, traderData)  # For visualizer
        return self.result, conversions, traderData


# Ignore code below, it's for the visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()
