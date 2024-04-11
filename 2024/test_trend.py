import unittest
from trend import TraderData

class TestTrend(unittest.TestCase):
    def test_TraderData_roundtrips(self):
        td = TraderData()
        td.price_n_minus_1 = 1
        td.price_n_minus_2 = 2
        td.price_n_minus_1_error = 3
        td.price_n_minus_2_error = 4

        json_str = td.to_json()
        td2 = TraderData.from_json(json_str)
        self.assertEqual(td, td2)

if __name__ == '__main__':
    unittest.main()