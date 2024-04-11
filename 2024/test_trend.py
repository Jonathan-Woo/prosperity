import unittest
from trend import TraderDataDTO

class TestTrend(unittest.TestCase):
    def test_TraderData_parses_default_instance_when_string_is_None(self):
        td = TraderDataDTO.from_json(None)
        self.assertEqual(td, TraderDataDTO())

    def test_TraderData_parses_default_instance_when_string_is_empty(self):
        td = TraderDataDTO.from_json("")
        self.assertEqual(td, TraderDataDTO())

    def test_TraderData_roundtrips(self):
        td = TraderDataDTO(1, 2, 3, 4)

        json_str = td.to_json()
        td2 = TraderDataDTO.from_json(json_str)
        self.assertEqual(td, td2)

if __name__ == '__main__':
    unittest.main()