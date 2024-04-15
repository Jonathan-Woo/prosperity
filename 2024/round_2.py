from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import List, Dict, Any
import string

import pandas as pd
import numpy as np
import statistics as stats
import math
import json
import jsonpickle


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
    def from_json(json_string: string):
        if json_string is None or json_string == "":
            return TraderDataDTO()

        return jsonpickle.decode(json_string)


class Trader:

    def __init__(self):
        self.result = {}
        self.traderData = None

        # For AMETHYSTS
        self.amethysts_params = {"position_limit": 20}

        # For STARFRUIT
        self.starfruit_params = {
            "position_limit": 20,
            "LR_coefs": [0.1, 0.1],
        }

    def starfruit(self):
        if "STARFRUIT" in self.state.position:
            cur_position = self.state.position["STARFRUIT"]
        else:
            cur_position = 0
        new_long_position, new_short_position = cur_position, cur_position
        position_limit = self.starfruit_params["position_limit"]
        window_size = len(self.starfruit_params["LR_coefs"])
        self.result["STARFRUIT"] = []

        highest_market_bid = max(self.state.order_depths["STARFRUIT"].buy_orders.keys())
        lowest_market_ask = min(self.state.order_depths["STARFRUIT"].sell_orders.keys())
        mid_price = (highest_market_bid + lowest_market_ask) / 2

        # Update the trader data with the latest price
        self.traderData.accept_product_price("STARFRUIT", mid_price)
        if len(self.traderData.get_product_price("STARFRUIT")) > window_size:
            self.traderData.drop_product_price("STARFRUIT")
            assert len(self.traderData.get_product_price("STARFRUIT")) == window_size

        # Start trading after the window of prices has been filled.
        if len(self.traderData.get_product_price("STARFRUIT")) < window_size:
            return

        # Compute expected price
        expected_price = np.array(self.starfruit_params["LR_coefs"]).dot(
            self.traderData.get_product_price("STARFRUIT")
        )
        expected_price = round(expected_price)

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
            order_price = min(tighter_bid + 1, mean_price - 1)
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

        print("AMETHYST ORDERS: ", self.result["AMETHYSTS"])

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Update properties from the last timestep.
        self.state = state
        self.traderData = TraderDataDTO.from_json(state.traderData)

        print(self.state.position)

        self.amethyst()
        self.starfruit()

        conversions = 1
        traderData = TraderDataDTO.to_json(self.traderData)
        # logger.flush(state, self.result, conversions, traderData)
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


# logger = Logger()
