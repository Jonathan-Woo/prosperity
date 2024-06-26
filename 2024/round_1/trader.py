from datamodel import OrderDepth, UserId, TradingState, Order, Symbol, Listing, Trade, Observation
from typing import List, Optional, Dict, Any
import string
import math
import jsonpickle
import numpy as np

def market_buy(product: string, sell_orders: Dict[int, int], acceptable_price, curr_pos):
        """Modularizing market buy order"""
        buys = []
        order_for = 0
        if len(sell_orders) != 0:
                best_ask, best_ask_amount = list(sell_orders.items())[0]
                if int(best_ask) <= acceptable_price:
                    order_for = min(-best_ask_amount, 20-curr_pos)
                    print("BUY", str(order_for) + "x", best_ask)
                    #logger.print("BUY", str(order_for) + "x", best_ask)
                    buys.append(Order(product, best_ask, order_for))
        return order_for, buys

def market_sell(product: string, buy_orders: Dict[int, int], acceptable_price, curr_pos):
    """Modularizing market sell orders"""
    sells = []
    order_for = 0
    if len(buy_orders) != 0:
            best_bid, best_bid_amount = list(buy_orders.items())[0]
            if int(best_bid) >= acceptable_price:
                # Similar situation with sell orders
                order_for = max(-best_bid_amount, -20-curr_pos)
                print("SELL", str(order_for) + "x", best_bid)
                #logger.print("SELL", str(-order_for) + "x", best_bid)
                sells.append(Order(product, best_bid, order_for))
    return order_for, sells

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


class TraderDataDTO:
    def __init__(self):
        self._prices = {}

    def accept_product_price(self, symbol, price):
        if symbol not in self._prices:
            self._prices[symbol] = []

        self._prices[symbol].append(price)

    def get_product_price(self, symbol):
        return self._prices[symbol]

    def to_json(self):
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json_string: string):
        if json_string is None or json_string == "":
            return TraderDataDTO()

        return jsonpickle.decode(json_string)

class TrendTrader:
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

    def lin_reg(self, cache):
        #may need tweaking; highest for most recent, but not too low for the past
        lag_wts = np.array([-0.6036, 0.0006, -0.1043, -0.4289])
        #intercept =  0 #4.3920470091215975
        return lag_wts@np.array(cache)
    

    def run(self, product: string, state: TradingState, trader_data_dto: TraderDataDTO):
        buy_orders = state.order_depths[product].buy_orders
        sell_orders = state.order_depths[product].sell_orders
        _, best_bid = self.max_vol_quote(buy_orders, 1)
        _, best_ask = self.max_vol_quote(sell_orders, 0)
        highest_bid = sorted(list(buy_orders.keys()), reverse=True)[0]
        lowest_ask = sorted(list(sell_orders.keys()))[0]
        curr_mid = (highest_bid+lowest_ask)/2

        curr_pos = state.position.get(product, 0)
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        ratio, ema_fast, ema_slow = 0, 0, 0
        acc_bid = {'STARFRUIT': -1e9}
        acc_ask = {'STARFRUIT': 1e9}

        acceptable_price = 1e9
        #to_prop = TraderData()
        # s for state
        s = trader_data_dto

        if not s.is_initialized():
            #try versions with both waiting for pred, and not waiting
            if state.timestamp/100<=2:
                s.accept_price_n(curr_mid)

            if state.timestamp/100==3:
                pred_n = self.lin_reg([s.price_n_minus_1-s.price_n_minus_2,\
                                     s.price_n_minus_2-s.price_n_minus_3, 
                                     0, 0]) + s.price_n_minus_1
                s.accept_price_n(curr_mid)
                s.accept_pred_n(pred_n)

            if state.timestamp/100==4:
                pred_n = self.lin_reg([s.price_n_minus_1-s.price_n_minus_2, \
                                     s.price_n_minus_2-s.price_n_minus_3, 
                                     curr_mid-s.pred_n, 0]) + s.price_n_minus_1    
                s.accept_price_n(curr_mid)
                s.accept_pred_n(pred_n)

        if s.is_initialized():
            acceptable_price = self.lin_reg(\
                  [s.price_n_minus_1-s.price_n_minus_2, \
                   s.price_n_minus_2-s.price_n_minus_3, \
                    curr_mid - s.pred_n, s.pred_n - s.pred_n_minus_1])
            s.accept_price_n(curr_mid)
            s.accept_pred_n(acceptable_price)
            
            acc_ask[product] = acceptable_price + 1
            acc_bid[product] = acceptable_price - 1
        
        print("Acceptable price : " + str(acceptable_price))
        print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))


        """Market Taking"""
        if not s.is_initialized():
            order_for, buys = market_buy(product, order_depth.sell_orders, acc_bid[product], curr_pos)
            curr_pos += order_for
            order_for, sells = market_sell(product, order_depth.buy_orders, acc_ask[product], curr_pos)
            curr_pos += order_for
            orders.extend(buys+sells)


        """Market Making"""
        
        ask = best_ask #min(best_ask - 1, acc_ask[product])
        bid = best_bid #max(best_bid + 1, acc_bid[product])
        assert best_ask > best_bid
        """Using ratio to gauge uptrend/downtrend -> adjusting bid/ask volume"""
        
        
        quantiles = [0.999738, 0.999999]
        if curr_pos >= 15 :#and ratio > 1:
            orders.append(Order(product, ask-1, -curr_pos))
        if curr_pos <= -15 :#and ratio < 1:
            orders.append(Order(product, bid+1, -curr_pos))
        if ratio <= quantiles[0]:
            """uptick"""
            orders.append(Order(product, bid, 20-curr_pos))
            #orders.append(Order(product, ask, -round(0.25*(20-curr_pos))))
        if ratio >= quantiles[1]:
            """downtick"""
            orders.append(Order(product, ask, -20-curr_pos))
            #orders.append(Order(product, bid, -round(0.25*(-20-curr_pos))))

        return orders

class SMMTrader:
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

        if curr_pos == 0:
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
    
    def run(self, state: TradingState):
        my_bids = {'AMETHYSTS':10000, 'STARFRUIT':0}
        my_asks = {'AMETHYSTS':10000, 'STARFRUIT':0}
        for product in state.order_depths:
            if product=='AMETHYSTS':
                return self.smm(product, state, my_bids[product], my_asks[product])
    
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
    

logger = Logger()
    
class Trader:
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        trader_data_dto = TraderDataDTO.from_json(state.traderData) 
        trend_trader = TrendTrader()
        smm_trader = SMMTrader()

        result = {}
        
        for product in state.order_depths:
            if product == 'STARFRUIT':
                result[product] = trend_trader.run(product, state, trader_data_dto)
            if product == 'AMETHYSTS':
                result[product] = smm_trader.run(state)
    
        conversions = 1
        trader_data = trader_data_dto.to_json()
        # logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data