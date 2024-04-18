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

    def __eq__(self, other):
        return

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

    def etf(self):
        def market_make(product, product_params):
            """
            This market making function is specific to the ETF products.
            Very similar to starfruit.
            """
            if product in self.state.position:
                cur_position = self.state.position[product]
            else:
                cur_position = 0

            new_long_position, new_short_position = cur_position, cur_position
            # TODO: Fix this shit. Shouldn't have to pass product params.
            position_limit = product_params["position_limit"]
            window_size = len(product_params["LR_coefs"]) - 1
            self.result[product] = []

            highest_market_bid = max(self.state.order_depths[product].buy_orders.keys())
            lowest_market_ask = min(self.state.order_depths[product].sell_orders.keys())
            mid_price = (highest_market_bid + lowest_market_ask) / 2

            # Update the trader data with the latest price
            self.update_stored_price(product, mid_price, window_size)

            # Start trading after the window of prices has been filled.
            if len(self.traderData.get_product_price(product)) < window_size:
                return

            # Compute expected price
            expected_price = self.linear_predict_price(
                product_params["LR_coefs"], product
            )

            # Market orders
            # Buy undervalued
            # Buy at value if we are short
            for ask, qty in self.state.order_depths[product].sell_orders.items():
                if (
                    (ask < expected_price)
                    or ((cur_position < 0) and (ask == mid_price))
                ) and new_long_position < position_limit:
                    order_qty = min(-qty, position_limit - new_long_position)
                    new_long_position += order_qty
                    self.result[product].append(Order(product, ask, order_qty))

            # Sell overvalued
            # Sell at value if we are long
            for bid, qty in self.state.order_depths[product].buy_orders.items():
                if (
                    (bid > expected_price)
                    or ((cur_position > 0) and (bid == mid_price))
                ) and new_short_position > -position_limit:
                    order_qty = min(qty, position_limit + new_short_position)
                    new_short_position -= order_qty
                    self.result[product].append(Order(product, bid, -order_qty))

            # Market making
            # Setup orders to tighten the spread
            # The price must be at least 1 away from the current market price or the expected price
            tighter_bid = highest_market_bid + 1
            tighter_ask = lowest_market_ask - 1

            if new_long_position < position_limit:
                order_qty = position_limit - new_long_position
                new_long_position += order_qty
                order_price = min(tighter_bid, expected_price - 1)
                self.result[product].append(Order(product, order_price, order_qty))
            if new_short_position > -position_limit:
                order_qty = position_limit + new_short_position
                new_short_position -= order_qty
                order_price = max(tighter_ask, expected_price + 1)
                self.result[product].append(Order(product, order_price, -order_qty))

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

        threshold = 380
        margin_of_safety = 10

        orders_etf = []
        orders_choc = []
        orders_roses = []
        orders_berries = []

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
            bids_etf = sorted(
                list(self.state.order_depths["GIFT_BASKET"].buy_orders.items()),
                key=lambda x: x[0],
                reverse=True,
            )
            asks_etf = sorted(
                list(self.state.order_depths["GIFT_BASKET"].sell_orders.items()),
                key=lambda x: x[0],
                reverse=False,
            )
            best_bid_etf = bids_etf[0][0]
            best_ask_etf = asks_etf[0][0]
            mid_etf = (best_bid_etf + best_ask_etf) / 2

        # etf_act = 6 * asks_berry[0][0] + 4 * asks_choc[0][0] + asks_roses[0][0]

        nav = 6 * mid_berry + 4 * mid_choc + mid_roses
        new_short_position = 0
        new_long_position = 0

        if mid_etf - nav > 380 + margin_of_safety:
            "short"

            "MARKET ORDERS"
            etf_qty = -min(
                bids_etf[0][1], pos_limit["GIFT_BASKET"] + curr_pos["GIFT_BASKET"]
            )
            assert etf_qty < 0
            curr_pos["GIFT_BASKET"] += etf_qty
            new_short_position = curr_pos["GIFT_BASKET"]
            orders_etf.append(Order("GIFT_BASKET", best_bid_etf, round(etf_qty)))

            # "LIMIT ORDERS"
            # new_mid = (bids_etf[1][0] + best_ask_etf)/2
            # diff = math.floor(best_ask_etf - new_mid)
            # if diff > 1:
            #     step = (-pos_limit['GIFT_BASKET']-curr_pos['GIFT_BASKET'])/(diff-1)
            #     for i in range(diff):
            #         if i!=0:
            #             orders_etf.append(Order('GIFT_BASKET', math.floor(new_mid)+i, step))

            # berry_qty = min(6*etf_qty, pos_limit['STRAWBERRIES']-berry_pos)
            # choc_qty = min(4*etf_qty, pos_limit['CHOCOLATE']-choc_pos)
            # rose_qty = min(etf_qty, pos_limit['ROSES']-rose_pos)
            # orders_berries.append(Order('STRAWBERRIES', best_ask_berry, berry_qty))
            # orders_choc.append(Order('CHOCOLATE', best_ask_choc, choc_qty))
            # orders_roses.append(Order('ROSES', best_ask_roses, rose_qty))

            # "LIMIT ORDERS adapted from Hashim's Orchids"
            # size_per_price = 33
            # sale = nav + 380 + margin_of_safety
            # position_limit = pos_limit['GIFT_BASKET'] + curr_pos['GIFT_BASKET']
            # max_iterations = (position_limit//size_per_price)+1
            # for i in range(1, max_iterations+1):
            #     if position_limit > 0:
            #         order_for = -min(size_per_price, position_limit)
            #         position_limit += order_for
            #         orders_etf.append(Order('GIFT_BASKET', sale+i, order_for))
            #         curr_pos['GIFT_BASKET'] += order_for
            #     else: break

        if mid_etf - nav < 380 - margin_of_safety:
            "long"
            # etf_qty = min(bids_roses[0][1], bids_choc[0][1]//4, bids_berry[0][1]//6)
            """MARKET ORDERS"""
            etf_qty = min(
                -asks_etf[0][1], pos_limit["GIFT_BASKET"] - curr_pos["GIFT_BASKET"]
            )
            curr_pos["GIFT_BASKET"] += etf_qty
            new_long_position = curr_pos["GIFT_BASKET"]
            assert etf_qty > 0
            orders_etf.append(Order("GIFT_BASKET", best_ask_etf, round(etf_qty)))

            # orders_etf.append(Order('GIFT_BASKET', best_bid_etf+2, -etf_qty))

            # """LIMIT ORDERS"""
            # size_per_price = 33
            # purchase = nav + 380 - margin_of_safety
            # position_limit = pos_limit['GIFT_BASKET'] - curr_pos['GIFT_BASKET']
            # max_iterations = (position_limit//size_per_price)+1
            # for i in range(1, max_iterations+1):
            #     if position_limit > 0:
            #         order_for = min(size_per_price, position_limit)
            #         position_limit += order_for
            #         orders_etf.append(Order('GIFT_BASKET', purchase-i, order_for))
            #         curr_pos['GIFT_BASKET'] += order_for
            #     else: break

            # berry_qty = max(-6*etf_qty, -pos_limit['STRAWBERRIES']-berry_pos)
            # choc_qty = max(-4*etf_qty, -pos_limit['CHOCOLATE']-choc_pos)
            # rose_qty = max(-etf_qty, -pos_limit['ROSES']-rose_pos)
            # orders_berries.append(Order('STRAWBERRIES', best_ask_berry, berry_qty))
            # orders_choc.append(Order('CHOCOLATE', best_ask_choc, choc_qty))
            # orders_roses.append(Order('ROSES', best_ask_roses, rose_qty))

        # Assume for now no other method is trading these products individually
        market_make("ROSES", self.rose_params)
        market_make("STRAWBERRIES", self.strawberry_params)
        market_make("CHOCOLATE", self.chocolate_params)

        self.result["CHOCOLATE"] = orders_choc
        self.result["ROSES"] = orders_roses
        self.result["STRAWBERRIES"] = orders_berries
        self.result["GIFT_BASKET"] = orders_etf

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

        # For the orchids strategy, convert all accrued positions.
        position = 0
        if "ORCHIDS" in state.position:
            position = state.position["ORCHIDS"]

        print(self.state.observations.conversionObservations)

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
