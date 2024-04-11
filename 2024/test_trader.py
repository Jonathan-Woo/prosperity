import unittest
from trader import TraderDataDTO, Trader

class TestTrader(unittest.TestCase):
    def test_TraderData_parses_default_instance_when_string_is_None(self):
        td = TraderDataDTO.from_json(None)
        self.assertEqual(td, TraderDataDTO())

    def test_TraderData_parses_default_instance_when_string_is_empty(self):
        td = TraderDataDTO.from_json("")
        self.assertEqual(td, TraderDataDTO())

    def test_TraderData_roundtrips(self):
        td = TraderDataDTO()

        json_str = td.to_json()
        td2 = TraderDataDTO.from_json(json_str)
        self.assertEqual(td, td2)

    def test_run_Trader(self):
        trader = Trader()
        # trader.run(None)

if __name__ == '__main__':
    unittest.main()