from datamodel import Listing, Observation, ConversionObservation, Order, OrderDepth, ProsperityEncoder, Trade, TradingState, Symbol, Time, Product, Position, UserId, ObservationValue
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import statistics as stats
import math
import jsonpickle
import json


class Trader:

    def __init__(self):
        self.result = {}

        # For AMETHYSTS
        self.amethysts_params ={
            'position_limit': 20
        }

        # For STARFRUIT
        self.starfruit_params = {
            'position_limit': 20,
            'past_prices': pd.Series(dtype='float64'),
            'predictions': pd.Series(dtype='float64'),
            'last_market_taking_action': "NONE"
        }

    def ORCHIDS(self, state: TradingState):
        orders = []
        bids = []
        asks = []
        if "ORCHIDS" in state.order_depths:
            bids = sorted(list(state.order_depths["ORCHIDS"].buy_orders.items()), key=lambda x: x[0], reverse=True)
            asks = sorted(list(state.order_depths["ORCHIDS"].sell_orders.items()), key=lambda x: x[0], reverse=False)
            best_bid = bids[0][0]
            best_ask = asks[0][0]

        # Get current position
        if "ORCHIDS" in state.position:
            position = state.position["ORCHIDS"]
        else:
            position = 0

        # logger.print(f"ORCHIDS: Current position is {position}.")

        # Calculating net purchase or selling price
        # through conversions at current timestep,
        # as an estimate for next timestep

        ask_conv = state.observations.conversionObservations["ORCHIDS"].askPrice
        bid_conv = state.observations.conversionObservations["ORCHIDS"].bidPrice
        transportFees = state.observations.conversionObservations["ORCHIDS"].transportFees
        exportTariff = state.observations.conversionObservations["ORCHIDS"].exportTariff
        importTariff = state.observations.conversionObservations["ORCHIDS"].importTariff
        sunlight = state.observations.conversionObservations["ORCHIDS"].sunlight
        humidity = state.observations.conversionObservations["ORCHIDS"].humidity

        # Net purchase price
        purchase = ask_conv + importTariff + transportFees
        # Net sales price
        sale = bid_conv - exportTariff - transportFees

        # Parameters
        margin_of_safety = 0
        size_per_price = 25

        lowest_we_would_sell = round(purchase + margin_of_safety)
        highest_we_would_buy = round(sale - margin_of_safety)


        ###

        buy_arb_opportunity = False
        sell_arb_opportunity = False

        if purchase < best_bid:
            buy_arb_opportunity = True
        if best_ask < sale:
            sell_arb_opportunity = True

        # Buy from South (next timestep) and sell in normal market (now)
        if buy_arb_opportunity:
            assert not sell_arb_opportunity
            position_limit = 100 + position
            for bid in bids:
                # If bid price facilitates arbitrage
                if purchase < bid[0]:
                    # If we have inventory space
                    if position_limit > 0:
                        order_quantity = min(bid[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('ORCHIDS', bid[0], -order_quantity))
                        # logger.print(f"ORCHIDS: Selling {order_quantity} at {bid[0]}.")
                    else:
                        break
                else:
                    break

            # since we are selling to arb, let's try to put limit asks as well
            diff = best_ask - lowest_we_would_sell
            if diff > 0:
                while position_limit > 0:
                    for i in range(1, diff):
                        if position_limit > 0:
                            order_quantity = min(15, position_limit)
                            position_limit -= order_quantity
                            orders.append(Order('ORCHIDS', best_ask - i, -order_quantity))
                            # logger.print(f"ORCHIDS: Market order selling {order_quantity} at {best_ask - i}.")
                        else:
                            break

        # End of 'Buy from South, sell locally'


        # Sell to South (next timestep) and buy in normal market (now)
        if sell_arb_opportunity:
            assert not buy_arb_opportunity
            position_limit = 100 - position
            for ask in asks:
                # If ask price facilitates arbitrage
                if ask[0] < sale:
                    # If we have inventory space
                    if position_limit > 0:
                        order_quantity = min(-ask[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('ORCHIDS', ask[0], order_quantity))
                        # logger.print(f"ORCHIDS: Buying {order_quantity} at {ask[0]}.")
                    else:
                        break
                else:
                    break

            # since we are buying to arb, let's try to put limit bids as well
            diff = highest_we_would_buy - best_bid
            if diff > 0:
                while position_limit > 0:
                    for i in range(1, diff):
                        if position_limit > 0:
                            order_quantity = min(15, position_limit)
                            position_limit -= order_quantity
                            orders.append(Order('ORCHIDS', best_bid + i, order_quantity))
                            # logger.print(f"ORCHIDS: Market order buying {order_quantity} at {best_bid + i}.")
                        else:
                            break

        # End of 'Sell to South, buy locally'

        # Market making arb, when no immediate arb available
        if not (buy_arb_opportunity or sell_arb_opportunity):
            # placing limit asks @ purchase + 1 and above
            position_limit = 100 + position
            max_iterations = (position_limit // size_per_price) + 1
            for i in range(1, max_iterations+1):
                if position_limit > 0:
                    order_quantity = min(size_per_price, position_limit)
                    position_limit -= order_quantity
                    orders.append(Order('ORCHIDS', lowest_we_would_sell + i, -order_quantity))
                    # logger.print(f"ORCHIDS: Market order selling {order_quantity} at {lowest_we_would_sell + i}.")
                else:
                    break

            # placing limit bids @ sale - 1 and below
            position_limit = 100 - position
            # logger.print(f"highest we'd buy is {highest_we_would_buy - 1}")
            for i in range(1, max_iterations+1):
                if position_limit > 0:
                    order_quantity = min(size_per_price, position_limit)
                    position_limit -= order_quantity
                    orders.append(Order('ORCHIDS', highest_we_would_buy - i, order_quantity))
                    # logger.print(f"most recent order is {Order('ORCHIDS', highest_we_would_buy - i, order_quantity)}")
                    # logger.print(f"ORCHIDS: Market order buying {order_quantity} at {highest_we_would_buy - i}.")
                else:
                    break

        # End of Market making arb

        self.result["ORCHIDS"] = orders

    def run(self, state: TradingState):
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.result = {}
        self.ORCHIDS(state)
        # logger.print(self.result["ORCHIDS"])
        if "ORCHIDS" in state.position:
            position = state.position["ORCHIDS"]
        else:
            position = 0
        conversions = -position
        # logger.print(f"Conversions this iteration is {conversions}")
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        logger.flush(state, self.result, conversions, traderData) # For visualizer
        return self.result, conversions, traderData


# Ignore code below, it's for the visualizer
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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

        return value[:max_length - 3] + "..."

logger = Logger()
