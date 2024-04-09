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
    

    # def smm_vol(self, product, state: TradingState, acc_bid, acc_ask, sds):

    #     buy_orders, sell_orders = state.order_depths[product].buy_orders , state.order_depths[product].sell_orders
    #     demand, best_bid = self.max_vol_quote(buy_orders, 1)
    #     supply, best_ask = self.max_vol_quote(sell_orders, 0)
    #     mid = (best_ask+best_bid)/2
    #     curr_pos = state.position.get(product, 0)
    #     prev_state = jsonpickle.decode(state.traderData) if state.traderData!=''else None
    #     orders = []
    #     "MARKET TAKING"
    #     new_pos, buys = self.market_buy(product, sell_orders, acc_bid, curr_pos)
    #     curr_pos += new_pos
    #     new_pos, sells = self.market_sell(product, buy_orders, acc_ask, curr_pos)
    #     curr_pos += new_pos
    #     orders.extend(buys+sells)

    #     "MARKET MAKING"
    #     curr_spread = best_ask - best_bid
    #     assert curr_spread > 0
    #     sig_level = 1.5
    #     ratio = None
    #     if prev_state is not None:
    #         ema = prev_state['ema']
    #         ratio = mid/ema
    #         alpha = prev_state['alpha']
            
    #     if demand > sig_level * supply:
    #             ask = best_ask
    #             bid = best_bid + 1
    #     if supply > sig_level * demand:
    #         ask = best_ask - 1
    #         bid = best_bid
    #     else:
    #         ask = best_ask-1
    #         bid = best_bid+1

    #     if ratio is None or (sds[1]<=ratio<=sds[0]):   
    #         if 10 < curr_pos <= 20:
    #             orders.append(Order(product, ask-1, -curr_pos))
    #         if 0 < curr_pos <= 10:
    #             orders.append(Order(product, ask, -curr_pos))
    #         if -10 <= curr_pos < 0:
    #             orders.append(Order(product, bid, -curr_pos))
    #         if -20 <= curr_pos < -10:
    #             orders.append(Order(product, bid+1, -curr_pos))
        
    #     elif ratio is not None and (ratio < sds[1] or sds[0] < ratio):
    #         if curr_pos > 0:
    #             orders.append(Order(product, ask+2, -curr_pos))
    #         if curr_pos < 0:
    #             orders.append(Order(product, bid-2, -curr_pos))
    #     propagate = {'ema': alpha*mid + (1-alpha)*ema, 'alpha': alpha}
    #     return orders, propagate
    
    
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
    