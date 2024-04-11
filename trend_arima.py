from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Optional 
from dataclasses import dataclass
import string
import math
import jsonpickle
import numpy as np

class Trader:
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
    

    def trend(self, product, state):
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
        prev = TraderData(state.traderData)
        #to_prop = TraderData()

        if not prev.is_initialized():
            #try versions with both waiting for pred, and not waiting
            if state.timestamp/100<=2:
                prev.accept(curr_mid)

            if state.timestamp/100==3:
                pred = self.lin_reg([prev.price_n_minus_1-prev.price_n_minus_2,\
                                     prev.price_n_minus_2-prev.price_n_minus_3, 
                                     0, 0]) + prev.price_n_minus_1
                prev.pred = pred
                prev.accept(curr_mid)

            if state.timestamp/100==4:
                pred = self.lin_reg([prev.price_n_minus_1-prev.price_n_minus_2, \
                                     prev.price_n_minus_2-prev.price_n_minus_3, 
                                     curr_mid-prev.pred, 0]) + prev.price_n_minus_1    
                prev.pred_n_minus_1 = prev.pred
                prev.pred = pred
                prev.accept(curr_mid)


        if prev.is_initialized():
            acceptable_price = self.lin_reg(\
                  [prev.price_n_minus_1-prev.price_n_minus_2, \
                   prev.price_n_minus_2-prev.price_n_minus_3, \
                    curr_mid - prev.pred, prev.pred - prev.pred_n_minus_1])
            prev.pred_n_minus_1 = prev.pred
            prev.pred = acceptable_price
            prev.accept(curr_mid)
            

            acc_ask[product] = acceptable_price + 1
            acc_bid[product] = acceptable_price - 1

        
        print("Acceptable price : " + str(acceptable_price))
        print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))



        """Market Taking"""
        if not prev.is_initialized():
            order_for, buys = self.market_buy(product, order_depth.sell_orders, acc_bid[product], curr_pos)
            curr_pos += order_for
            order_for, sells = self.market_sell(product, order_depth.buy_orders, acc_ask[product], curr_pos)
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

        return orders, prev.to_json()
        #result[product] = orders

##############################################################################################
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        
        for product in state.order_depths:
            if product=='STARFRUIT':
                result[product], prop = self.trend(product, state)
    
        traderData = jsonpickle.encode(prop) 
        
        conversions = 1
        return result, conversions, traderData
    
#############################################################################################
    def market_buy(self, product, sell_orders, acceptable_price, curr_pos):
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

    def market_sell(self, product, buy_orders, acceptable_price, curr_pos):
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
    

@dataclass
class TraderData:
    # The last price seen (NOT the current tick)
    price_n_minus_1: Optional[float] = None
    # The second last price seen
    price_n_minus_2: Optional[float] = None
    price_n_minus_3: Optional[float] = None

    pred: Optional[float] = None

    pred_n_minus_1: Optional[float] = None

    #price_n_minus_1_error: Optional[float] = None
    #price_n_minus_2_error: Optional[float] = None

    def to_json(self):
        return jsonpickle.encode(self)

    @staticmethod
    def from_json(json_string):
        return jsonpickle.decode(json_string)
    
    def accept(self, new:float):
        self.price_n_minus_3 = self.price_n_minus_2
        self.price_n_minus_2 = self.price_n_minus_1
        self.price_n_minus_1 = new
    
    def is_initialized(self):
        return self.price_n_minus_1 is not None \
            and self.price_n_minus_2 is not None\
            and self.pred is not None and self.pred_n_minus_1 is not None

INITIAL_TRADER_DATA = TraderData()