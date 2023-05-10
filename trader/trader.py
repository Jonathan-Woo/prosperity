# The Python code below is the minimum code that is required in a submission file:
# 1. The "datamodel" imports at the top. Using the typing library is optional.
# 2. A class called "Trader", this class name should not be changed.
# 3. A run function that takes a tradingstate as input and outputs a "result" dict.

import pandas as pd
import numpy as np
import statistics as stats
import math

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        self.result = {}

        #For pearls
        self.p_params = {
            'position_limit': 20
        }

        #For bananas
        self.b_params = {
            'position_limit': 20,
            'past_prices': pd.Series(dtype='float64'),
            'avg_was_above': None,
            'last_ewm': None,
            'span': 200,
            'window': 320
        }
        self.b_params['alpha'] = 2/(self.b_params['span'] + 1)

        #For pina coladas and coconuts
        self.pina_coco_params = {
            'pina_position_limit': 300,
            'coco_position_limit': 600,
            'beta': 1,
            'mu': 7000,
            'upper_threshold': 45,
            'lower_threshold': -45,
            'fast_window': 200,
            'slow_window': 50,
            'past_spreads': pd.Series(dtype='float64'),
        }

        #For berries
        self.berries_params = {
            'position_limit': 250,
            'threshold_1': 350000,
            'threshold_2': 700000,
        }

        #For diving gear
        self.diving_gear_params = {
            'position_limit': 50,
            'upper_threshold': 10000,
            'lower_threshold': 100000,
            'diff_distance': 5,
            'past_n_dolphins': [],
            'buying': False,
            'selling': False,
            'buying_rate': 5,
            'selling_rate': 5
        }

    def pearls(self, state: TradingState):
        #PEARLS
        [position_limit] = self.p_params.values()

        #Always maximize position for pearls
        #Bids from highest to lowest
        #Asks from lowest to highest
        orders = []
        bids = []
        asks = []
        if 'PEARLS' in state.order_depths:
            bids = sorted(list(state.order_depths['PEARLS'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            asks = sorted(list(state.order_depths['PEARLS'].sell_orders.items()), key=lambda x: x[0], reverse=False)
        
        #Get current position
        if 'PEARLS' in state.position:
            position = state.position['PEARLS']
        else:
            position = 0
                
        #Buy if lowest ask is below 10000
        if(len(asks) > 0 and asks[0][0] < 10000):
            position_limit = 20 - position
            #Walk through from the lowest asks upwards
            for ask in asks:
                #If the ask is less than 10000, buy as much as possible
                if ask[0] < 10000:
                    #If we have reached the position limit, stop buying
                    if position_limit > 0:
                        order_quantity = min(-ask[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('PEARLS', ask[0], order_quantity))
                    else:
                        break
                else:
                    break
            
            #if we have any remaining position limit, place an ask to buy at 9999 or lower
            if position_limit > 0:
                order_quantity = position_limit
                orders.append(Order('PEARLS', 9999, order_quantity))

        #Sell if highest bid is above 10000
        elif(len(bids) > 0 and bids[0][0] > 10000):
            position_limit = 20 + position
            #Walk through from the highest bids downwards
            for bid in bids:
                #If the bid is greater than 10000, sell as much as possible
                if bid[0] > 10000:
                    #If we have reached the position limit, stop selling
                    if position_limit > 0:
                        order_quantity = min(bid[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('PEARLS', bid[0], -order_quantity))
                    else:
                        break
                else:
                    break
            
            #if we have any remaining position limit, place a bid to sell at 10001 or higher
            if position_limit > 0:
                order_quantity = position_limit
                orders.append(Order('PEARLS', 10001, -order_quantity))
        
        self.result['PEARLS'] = orders

    def bananas(self, state: TradingState):
        #BANANAS
        [position_limit, past_prices, avg_was_above, last_ewm, span, window, alpha] = self.b_params.values()

        orders = []
        bids = []
        asks = []
        if 'BANANAS' in state.order_depths:
            bids = sorted(list(state.order_depths['PEARLS'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            asks = sorted(list(state.order_depths['PEARLS'].sell_orders.items()), key=lambda x: x[0], reverse=False)

        #Get current position
        if 'BANANAS' in state.position:
            position = state.position['BANANAS']
        else:
            position = 0

        midpoint = bids[0][0] + (asks[0][0] - bids[0][0]) / 2
        past_prices = pd.concat([past_prices, pd.Series(midpoint)], ignore_index=True)
        self.b_params['past_prices'] = past_prices

        #TODO: REWRITE AVG TO MAKE IT MORE EFFICIENT
        avg = past_prices.rolling(window, min_periods=0).mean().iloc[-1]

        #For the first timestep, only setup the ewm
        if last_ewm == None:
            self.b_params['last_ewm'] = midpoint
        #For the second timestep, only setup the avg_was_above
        elif avg_was_above == None:
            ewm = alpha * midpoint + (1 - alpha) * last_ewm
            self.b_params['avg_was_above'] = avg > ewm
            self.b_params['last_ewm'] = alpha * midpoint + (1 - alpha) * last_ewm
        else:
            ewm = alpha * midpoint + (1 - alpha) * last_ewm
            #If the average is above the ewm, and it was below the ewm last time, the average has crossed from below, buy
            if avg > ewm and not avg_was_above:
                position_limit = 20 - position
                #Walk through from the lowest asks upwards
                for ask in asks:
                    #If we have reached the position limit, stop buying
                    if position_limit > 0:
                        order_quantity = min(-ask[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('BANANAS', ask[0], order_quantity))
                    else:
                        break
                self.b_params['avg_was_above'] = True

            #If the average is below the ewm, and it was above the ewm last time, the average has crossed from above sell
            elif avg < ewm and avg_was_above:
                position_limit = 20 + position
                #Walk through from the highest bids downwards
                for bid in bids:
                    #If we have reached the position limit, stop selling
                    if position_limit > 0:
                        order_quantity = min(bid[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('BANANAS', bid[0], -order_quantity))
                    else:
                        break
                self.b_params['avg_was_above'] = False
            self.b_params['last_ewm'] = ewm

        self.result['BANANAS'] = orders

    def pina_coco(self, state: TradingState):
        #PINA COLADAS & COCONUTS
        [p_position_limit, c_position_limit, beta, mu, upper_threshold, lower_threshold, slow_window, fast_window, past_spreads] = self.pina_coco_params.values()

        p_orders = []
        c_orders = []
        p_bids = []
        p_asks = []
        if 'PINA_COLADAS' in state.order_depths:
            p_bids = sorted(list(state.order_depths['PINA_COLADAS'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            p_bids = [x[0] for x in p_bids for _ in range(x[1])]
            p_asks = sorted(list(state.order_depths['PINA_COLADAS'].sell_orders.items()), key=lambda x: x[0], reverse=False)
            p_asks = [x[0] for x in p_asks for _ in range(-x[1])]
        n_p_bids = len(p_bids)
        n_p_asks = len(p_asks)

        c_bids = []
        c_asks = []
        if 'COCONUTS' in state.order_depths:
            c_bids = sorted(list(state.order_depths['COCONUTS'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            c_bids = [x[0] for x in c_bids for _ in range(x[1])]
            c_asks = sorted(list(state.order_depths['COCONUTS'].sell_orders.items()), key=lambda x: x[0], reverse=False)
            c_asks = [x[0] for x in c_asks for _ in range(-x[1])]
        n_c_bids = len(c_bids)
        n_c_asks = len(c_asks)

        if n_p_bids == 0 or n_p_asks == 0 or n_c_bids == 0 or n_c_asks == 0:
            return

        #Get current position
        if 'PINA_COLADAS' in state.position:
            p_position = state.position['PINA_COLADAS']
        else:
            p_position = 0
        if 'COCONUTS' in state.position:
            c_position = state.position['COCONUTS']
        else:
            c_position = 0

        #Compute the position of the pinas - coconut spread
        p_mid = (p_bids[0] + p_asks[0]) / 2
        c_mid = (c_bids[0] + c_asks[0]) / 2
        
        diff = p_mid - c_mid - mu
        past_spreads = pd.concat([past_spreads, pd.Series(diff)], ignore_index=True)
        self.pina_coco_params['past_spreads'] = past_spreads

        slow_avg = past_spreads.rolling(slow_window, min_periods=0).mean().iloc[-1]
        fast_avg = past_spreads.rolling(fast_window, min_periods=0).mean().iloc[-1]

        #If the spread is wide, short pinas and long coconuts
        #Use the moving average to determine if the spread is widening or narrowing
        if diff > upper_threshold and fast_avg < slow_avg:
            #Sell pinas & buy coconuts
            p_position_limit = p_position_limit + p_position
            c_position_limit = c_position_limit - c_position

            #Match the highest bid of the pinas with 2 of the lowest asks of the cocos
            while n_p_bids > 0 and n_p_asks > 0 and n_c_bids > 0 and n_c_asks > 1 and p_position_limit > 0 and c_position_limit > 1:
                p_mid = (p_bids[0] + p_asks[0]) / 2
                c_mid = ((c_bids[0] + c_asks[0]) / 2 + (c_bids[0] + c_asks[1]) / 2) / 2
                diff = p_mid - c_mid*beta

                if diff < upper_threshold:
                    break

                p_bid = p_bids.pop(0)
                c_ask = c_asks.pop(0)
                c_ask2 = c_asks.pop(0)

                p_position_limit -= 1
                c_position_limit -= 2

                p_orders.append(Order('PINA_COLADAS', p_bid, -1))
                c_orders.append(Order('COCONUTS', c_ask, 1))
                c_orders.append(Order('COCONUTS', c_ask2, 1))
                n_p_bids -= 1
                n_c_asks -= 2

        #If the spread is narrow, long pinas and short coconuts
        #Use the moving average to determine if the spread is widening or narrowing
        elif diff < lower_threshold and fast_avg > slow_avg:
            #Buy pinas & sell coconuts
            p_position_limit = p_position_limit - p_position
            c_position_limit = c_position_limit + c_position

            #Match the lowest ask of the pinas with 2 of the highest bids of the cocos
            while n_p_bids > 0 and n_p_asks > 0 and n_c_bids > 1 and n_c_asks > 0 and p_position_limit > 0 and c_position_limit > 1:
                p_mid = (p_bids[0] + p_asks[0]) / 2
                c_mid = ((c_bids[0] + c_asks[0]) / 2 + (c_bids[1] + c_asks[0]) / 2) / 2
                diff = p_mid - c_mid*beta

                if diff > lower_threshold:
                    break

                p_ask = p_asks.pop(0)
                c_bid = c_bids.pop(0)
                c_bid2 = c_bids.pop(0)

                p_position_limit -= 1
                c_position_limit -= 2

                p_orders.append(Order('PINA_COLADAS', p_ask, 1))
                c_orders.append(Order('COCONUTS', c_bid, -1))
                c_orders.append(Order('COCONUTS', c_bid2, -1))
                n_p_asks -= 1
                n_c_bids -= 2

        self.result['PINA_COLADAS'] = p_orders
        self.result['COCONUTS'] = c_orders

    def berries(self, state: TradingState):
        [position_limit, threshold_1, threshold_2] = self.berries_params.values()

        orders = []
        bids = []
        asks = []
        if 'BERRIES' in state.order_depths:
            bids = sorted(list(state.order_depths['BERRIES'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            bids = [x[0] for x in bids for _ in range(x[1])]
            asks = sorted(list(state.order_depths['BERRIES'].sell_orders.items()), key=lambda x: x[0], reverse=False)

        #Get current position
        if 'BERRIES' in state.position:
            position = state.position['BERRIES']
        else:
            position = 0

        timestep = state.timestamp
        
        #Buy 1 unit of berries for each timestep before the first threshold
        if timestep < threshold_1 and timestep >= threshold_1 - position_limit * 100:
            orders.append(Order('BERRIES', asks[0][0], 1))
        #Sell and short between the first and second threshold
        #Sell 2 at each timestep
        elif timestep >= 495000 and timestep < 505000:
            orders.append(Order('BERRIES', bids[0], -1))
            orders.append(Order('BERRIES', bids[1], -1))
        #Cover short position after the second threshold
        elif timestep >= threshold_2 and position < 0:
            orders.append(Order('BERRIES', asks[0][0], 1))

        self.result['BERRIES'] = orders

    def diving_gear(self, state: TradingState):
        [position_limit, upper_threshold, lower_threshold, diff_distance, past_n_dolphins, buying, selling, buying_rate, selling_rate] = self.diving_gear_params.values()

        orders = []
        bids = []
        asks = []
        if 'DIVING_GEAR' in state.order_depths:
            bids = sorted(list(state.order_depths['DIVING_GEAR'].buy_orders.items()), key=lambda x: x[0], reverse=True)
            asks = sorted(list(state.order_depths['DIVING_GEAR'].sell_orders.items()), key=lambda x: x[0], reverse=False)
        
        #Get current position
        if 'DIVING_GEAR' in state.position:
            position = state.position['DIVING_GEAR']
        else:
            position = 0

        #Get the number of dolphins
        #Update the past number of dolphins
        past_n_dolphins.append(state.observations['DOLPHIN_SIGHTINGS'])
        print(past_n_dolphins)

        #Build up the window of past observations
        if(len(past_n_dolphins) < diff_distance):
            return
        elif len(past_n_dolphins) > diff_distance:
            past_n_dolphins.pop(0)

        self.past_n_dolphins = past_n_dolphins

        #Get the difference between the current number of dolphins and the number of dolphins diff_distance timesteps ago
        diff = past_n_dolphins[-1] - past_n_dolphins[0]

        #If we're in the buying phase, buy diving gear based on the buying rate
        if buying:
            #Stop buying if we've reached the position limit
            if position_limit - position <= buying_rate:
                self.buying = False
            position_limit = min(position_limit - position, buying_rate)
            for ask in asks:
                if position_limit > 0:
                    order_quantity = min(ask[1], position_limit)
                    orders.append(Order('DIVING_GEAR', ask[0], order_quantity))
                    position_limit -= order_quantity
                else:
                    break
        
        #If we're in the selling phase, sell diving gear based on the selling rate
        elif selling:
            #Stop selling if we've reached the position limit
            if position_limit + position <= selling_rate:
                self.selling = False
            position_limit = min(position_limit + position, selling_rate)
            for bid in bids:
                if position_limit > 0:
                    order_quantity = min(bid[1], position_limit)
                    orders.append(Order('DIVING_GEAR', bid[0], -order_quantity))
                    position_limit -= order_quantity
                else:
                    break
        
        #If we're not in the buying or selling phase, check if we should enter either phase
        #Long diving gear if the number of dolphins is increasing
        elif diff > upper_threshold:
            #Buy diving gear
            self.buying = True
            position_limit = min(position_limit - position, buying_rate)
            for ask in asks:
                if position_limit > 0:
                    order_quantity = min(ask[1], position_limit)
                    orders.append(Order('DIVING_GEAR', ask[0], order_quantity))
                    position_limit -= order_quantity
                else:
                    break

        #Short diving gear if the number of dolphins is decreasing
        elif diff < lower_threshold:
            #Sell diving gear
            self.selling = True
            position_limit = min(position_limit + position, selling_rate)
            for bid in bids:
                if position_limit > 0:
                    order_quantity = min(bid[1], position_limit)
                    orders.append(Order('DIVING_GEAR', bid[0], -order_quantity))
                    position_limit -= order_quantity
                else:
                    break

        self.result['DIVING_GEAR'] = orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        self.result = {}
        self.pearls(state)
        self.bananas(state)
        self.pina_coco(state)
        self.berries(state)
        self.diving_gear(state)

        return self.result