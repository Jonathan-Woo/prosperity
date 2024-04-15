import unittest
from trader import TraderDataDTO, Trader

class TestTrader(unittest.TestCase):
    def test_accept_prices(self):
        trader = TraderDataDTO()
        trader.accept_product_price("STARFRUIT", 1)
        trader.accept_product_price("STARFRUIT", 2)
        trader.accept_product_price("STARFRUIT", 3)

        json = trader.to_json()
        new = TraderDataDTO.from_json(json)

        self.assertEqual(new.get_product_price("STARFRUIT"), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()