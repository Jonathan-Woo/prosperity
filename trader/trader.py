# The Python code below is the minimum code that is required in a submission file:
# 1. The "datamodel" imports at the top. Using the typing library is optional.
# 2. A class called "Trader", this class name should not be changed.
# 3. A run function that takes a tradingstate as input and outputs a "result" dict.

import pandas as pd
import numpy as np
import statistics as stats
import math
import typing as t

from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__():
        self.test = np.array([1,2,3])

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}

        self.test *= 2
        print(self.test)
        
        ###################################################################
        #PEARLS
        #Always maximize position for pearls
        #Bids from highest to lowest
        #Asks from lowest to highest
        bids = []
        if 'PEARLS' in state.order_depths:
            bids = sorted(list(state.order_depths['PEARLS'].buy_orders.items()), key=lambda x: x[0], reverse=True)
        asks = []
        if 'PEARLS' in state.order_depths:
            asks = sorted(list(state.order_depths['PEARLS'].sell_orders.items()), key=lambda x: x[0], reverse=False)
        
        #Get current position
        if 'PEARLS' in state.position:
            position = state.position['PEARLS']
        else:
            position = 0
        
        orders = []
        
        #Buy if lowest ask is below 10000
        if(len(asks) > 0 and asks[0][0] < 10000):
            position_limit = 20 - position
            #Walk through from the lowest asks upwards
            for ask in asks:
                #If the ask is less than 10000, buy as much as possible
                if ask[0] < 10000:
                    #If we have reached the position limit, stop buying
                    if position_limit > 0:
                        order_quantity = min(ask[1], position_limit)
                        position_limit -= order_quantity
                        orders.append(Order('PEARLS', ask[0], order_quantity))
                    else:
                        break
                else:
                    break
            
            #if we have any remaining position limit,
            #TODO
            if position_limit > 0:
                pass

        #Sell if highest bid is above 10000
        if(len(bids) > 0 and bids[0][0] > 10000):
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
            
            #if we have any remaining position limit
            #TODO
            if position_limit > 0:
                pass

        result['PEARLS'] = orders

        ###################################################################
        #BANANAS


        return result