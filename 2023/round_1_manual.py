#%%
#Generate list of all possibilities
items = [
    'pizza',
    'wasabi',
    'snowball',
    'shells'
]

possible_trades = []

#3 trades
for item1 in items:
    cur_trade = ['shells']
    cur_trade.append(item1)
    for item2 in items:
        cur_trade.append(item2)
        for item3 in items:
            cur_trade.append(item3)
            for item4 in items:
                cur_trade.append(item4)
                cur_trade.append('shells')
                possible_trades.append(cur_trade.copy())
                cur_trade.pop(-1)
                cur_trade.pop(-1)
            cur_trade.pop(-1)
        cur_trade.pop(-1)

#%%
#Create table of trade values
index = {
    ('pizza', 'pizza'): 1,
    ('pizza', 'wasabi'): 0.5,
    ('pizza', 'snowball'): 1.45,
    ('pizza', 'shells'): 0.75,

    ('wasabi', 'pizza'): 1.95,
    ('wasabi', 'wasabi'): 1,
    ('wasabi', 'snowball'): 3.1,
    ('wasabi', 'shells'): 1.49,

    ('snowball', 'pizza'): 0.67,
    ('snowball', 'wasabi'): 0.31,
    ('snowball', 'snowball'): 1,
    ('snowball', 'shells'): 0.48,

    ('shells', 'pizza'): 1.34,
    ('shells', 'wasabi'): 0.64,
    ('shells', 'snowball'): 1.98,
    ('shells', 'shells'): 1,
}

#%%
#Compute value of each trade
trade_values = {}

for trade in possible_trades:
    value = 1
    for i in range(len(trade)-1):
        value *= index[(trade[i], trade[i+1])]
    trade_values[tuple(trade)] = value

trade_values = list(trade_values.items())

trade_values.sort(key=lambda x: x[1], reverse=True)
# %%
