from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import string

import pandas as pd
import numpy as np
import statistics as stats
import math
import json


class Trader:

    def __init__(self):
        self.result = {}

        # For AMETHYSTS
        self.amethysts_params ={
            'position_limit':20
        }

        # For STARFRUIT
        self.starfruit_params = {
            'position_limit': 20,
            'past_prices': pd.Series(dtype='float64'),
            'predictions': pd.Series(dtype='float64'),
            'last_market_taking_action': "NONE"
        }

    def STARFRUIT(self, state: TradingState):
        market_make = "no"
        market_take = "yes"
        [position_limit, past_prices, predictions, last_market_taking_action] = self.starfruit_params.values()
        # Bids from highest to lowest
        # Asks from lowest to highest
        orders = []
        bids = []
        asks = []
        if 'STARFRUIT' in state.order_depths:
            bids = sorted(list(state.order_depths['STARFRUIT'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            asks = sorted(list(state.order_depths['STARFRUIT'].sell_orders.items()), key=lambda x: x[0], reverse=False)

        # Get current position
        if 'STARFRUIT' in state.position:
            position = state.position['STARFRUIT']
        else:
            position = 0

        logger.print(f"STARFRUIT: Current position is {position}.")

        midpoint = (asks[0][0] + bids[0][0]) / 2
        past_prices = pd.concat([past_prices, pd.Series(midpoint)], ignore_index=True)
        self.starfruit_params['past_prices'] = past_prices

        if len(past_prices) > 1:
            prediction = past_prices[len(past_prices)-2] - 0.7 * (predictions[len(predictions)-1] - past_prices[len(past_prices)-2])
            predictions = pd.concat([predictions, pd.Series(prediction)], ignore_index=True)
        elif len(past_prices) == 1:
            prediction = past_prices[0]
            predictions = pd.concat([predictions, pd.Series(prediction)], ignore_index=True)

        self.starfruit_params['predictions'] = predictions

        logger.print(f"The STARFRUIT midprice now is {midpoint}, we predict the price to become {predictions[len(predictions)-1]}.")

        # BUY if lowest ask is below predicted price
        position_limit = 20 - position

        if market_take == "yes":
            if (len(asks) > 0 and asks[0][0] < prediction):
                # Walk through from the lowest asks upwards
                for ask in asks:
                    # If the ask is less than 10000, buy as much as possible
                    if ask[0] < prediction:
                        # If we have reached the position limit, stop buying
                        if position_limit > 0:
                            order_quantity = min(-ask[1], position_limit)
                            position_limit -= order_quantity
                            orders.append(Order('STARFRUIT', ask[0], order_quantity))
                            print(f"STARFRUIT: Buying {order_quantity} at {ask[0]}.")
                            last_market_taking_action = "BUY"
                        else:
                            break
                    else:
                        break

        # If we have any remaining position limit, place an ask to buy at 9999 or lower
        if market_make == "yes":
            if position_limit > 0:
                order_quantity = position_limit
                orders.append(Order('STARFRUIT', 9999, order_quantity))

        # SELL if highest bid is above prediction
        position_limit = 20 + position

        if market_take == "yes":
            if (len(bids) > 0 and bids[0][0] > prediction):
                # Walk through from the highest bids downwards
                for bid in bids:
                    # If the bid is greater than 10000, sell as much as possible
                    if bid[0] > 10000:
                        # If we have reached the position limit, stop selling
                        if position_limit > 0:
                            order_quantity = min(bid[1], position_limit)
                            position_limit -= order_quantity
                            orders.append(Order('STARFRUIT', bid[0], -order_quantity))
                            print(f"STARFRUIT: Selling {order_quantity} at {bid[0]}.")
                            last_market_taking_action = "SELL"
                        else:
                            break
                    else:
                        break

        # If we have any remaining position limit, place a bid to sell at 10001 or higher
        if market_make == "yes":
            if position_limit > 0:
                order_quantity = position_limit
                orders.append(Order('STARFRUIT', 10001, -order_quantity))

        self.result['STARFRUIT'] = orders

    def max_vol_quote(self, order_dict, buy):
        """The highest vol is indicative of actual market sentiment;
        this is what we tried to do earlier with the vwap"""
        best_quote, best_vol = 0, -float('inf')
        total_vol = 0
        for item in order_dict:
            vol = order_dict[item]
            if buy==0:
                vol *= -1
            total_vol += vol
            if vol > best_vol:
                best_vol, best_quote = vol, item
        return total_vol, best_quote

    def amethyst(self, state: TradingState):
        

    def smm(self, product, state, acc_bid, acc_ask):
        """Simple Market making"""
        buy_orders, sell_orders = state.order_depths[product].buy_orders, state.order_depths[product].sell_orders
        demand, best_bid = self.max_vol_quote(buy_orders, 1)
        supply, best_ask = self.max_vol_quote(sell_orders, 0)
        curr_pos = state.position.get(product, 0)

        orders = []



        "MARKET MAKING"
        curr_spread = best_ask - best_bid
        assert curr_spread > 0
        sig_level = 1.5
        #     if demand > sig_level * supply:
        #             ask = best_ask
        #             bid = best_bid + 1
        #     if supply > sig_level * demand:
        #         ask = best_ask - 1
        #         bid = best_bid
        #     else:
        ask = best_ask-1
        bid = best_bid+1

        #adjusting spread based on last execution price
        # if state.market_trades.get(product, None) is not None:
        #     adj_bid, adj_ask = bid, ask
        #     last_trade = (sorted(state.market_trades[product], key=lambda x: x.timestamp, reverse=True))[0]
        #     last_trade_price = last_trade.price
        #     if bid < last_trade_price < ask:
        #         dist_bid, dist_ask = abs(bid-last_trade_price), abs(ask-last_trade_price)
        #         if dist_bid <= dist_ask: adj_bid = last_trade_price
        #         if dist_ask < dist_bid: adj_ask = last_trade_price
        #     if last_trade_price < bid:
        #         adj_bid = last_trade_price
        #     if ask < last_trade_price:
        #         adj_ask = last_trade_price
        #     bid, ask = adj_bid, adj_ask

        if curr_pos == 0: #increase volume and test again
            orders.append(Order(product, ask, -5))
            orders.append(Order(product, bid, 5))
        if 15 < curr_pos <= 20:
            order_for = -curr_pos
            orders.append(Order(product, ask-1, order_for))
            curr_pos += order_for
        if 0 < curr_pos <= 15:
            order_for = round(0.75*-curr_pos) + round(0.25*curr_pos)
            orders.append(Order(product, ask, round(0.75*-curr_pos)))
            orders.append(Order(product, bid, round(0.25 * curr_pos)))
            curr_pos += order_for
        if -15 <= curr_pos < 0:
            order_for = round(0.75*-curr_pos) + round(0.25*curr_pos)
            orders.append(Order(product, bid, round(0.75*-curr_pos)))
            orders.append(Order(product, ask, round(0.25 * curr_pos)))
            curr_pos += order_for
        if -20 <= curr_pos < -15:
            order_for = -curr_pos
            orders.append(Order(product, bid+1, -curr_pos))
            curr_pos += order_for



        "MARKET TAKING"
        new_pos, buys = self.market_buy(product, sell_orders, acc_bid, curr_pos)
        curr_pos += new_pos
        new_pos, sells = self.market_sell(product, buy_orders, acc_ask, curr_pos)
        curr_pos += new_pos
        orders.extend(buys+sells)



        return orders

    def market_buy(self, product, sell_orders, acceptable_price, curr_pos):
        """Modularizing market buy order"""
        buys = []
        order_for = 0
        if len(sell_orders) != 0:
            best_ask, best_ask_amount = list(sell_orders.items())[0]
            if int(best_ask) < acceptable_price:
                order_for = min(-best_ask_amount, 20-curr_pos)
                print("BUY", str(order_for) + "x", best_ask)
                buys.append(Order(product, best_ask, order_for))
        return order_for, buys

    def market_sell(self, product, buy_orders, acceptable_price, curr_pos):
        """Modularizing market sell orders"""
        sells = []
        order_for = 0
        if len(buy_orders) != 0:
            best_bid, best_bid_amount = list(buy_orders.items())[0]
            if int(best_bid) > acceptable_price:
                # Similar situation with sell orders
                order_for = max(-best_bid_amount, -20-curr_pos)
                print("SELL", str(-order_for) + "x", best_bid)
                sells.append(Order(product, best_bid, order_for))
        return order_for, sells

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.result = {}
        my_bids = {'AMETHYSTS':10000, 'STARFRUIT':0}
        my_asks = {'AMETHYSTS':10000, 'STARFRUIT':0}
        self.STARFRUIT(state)
        self.result['AMETHYSTS'] = self.smm('AMETHYSTS', state, my_bids['AMETHYSTS'], my_asks['AMETHYSTS'])

        conversions = 1
        traderData = "SAMPLE"

        logger.flush(state, self.result, conversions, traderData)
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
