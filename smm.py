from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle

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

    def smm(self, product, state, acc_bid, acc_ask):
        """Simple Market making"""
        buy_orders, sell_orders = state.order_depths[product].buy_orders, state.order_depths[product].sell_orders
        demand, best_bid = self.max_vol_quote(buy_orders, 1)
        supply, best_ask = self.max_vol_quote(sell_orders, 0)
        curr_pos = state.position.get(product, 0)

        orders = []

        "MARKET TAKING"
        new_pos, buys = self.market_buy(product, sell_orders, acc_bid, curr_pos)
        curr_pos += new_pos
        new_pos, sells = self.market_sell(product, buy_orders, acc_ask, curr_pos)
        curr_pos += new_pos
        orders.extend(buys+sells)

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

        if 15 < curr_pos <= 20:
            orders.append(Order(product, ask-1, -curr_pos))
        if 0 < curr_pos <= 15:
            orders.append(Order(product, ask, round(0.75*-curr_pos)))
            orders.append(Order(product, bid, round(0.25 * curr_pos)))
        if -15 <= curr_pos < 0:
            orders.append(Order(product, bid, round(0.75*-curr_pos)))
            orders.append(Order(product, ask, round(0.25 * curr_pos)))
        if -20 <= curr_pos < -15:
            orders.append(Order(product, bid+1, -curr_pos))
        return orders
    
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        my_bids = {'AMETHYSTS':10000, 'STARFRUIT':0}
        my_asks = {'AMETHYSTS':10000, 'STARFRUIT':0}
        result = {}
        positions = state.position
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]

            if product=='AMETHYSTS':
                result[product] = self.smm(product, state, my_bids[product], my_asks[product])
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1
        return result, conversions, traderData
    
####################################################################################
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
    