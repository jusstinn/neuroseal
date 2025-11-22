import unittest
from unittest.mock import patch
import sys
import argparse
import neuroseal.cli 

class TestCLI(unittest.TestCase):
    
    def test_lock_command(self):
        with patch("neuroseal.cli.apply_lock") as mock_apply_lock:
            test_args = ["neuroseal", "lock", "input_model", "output_dir", "--scale", "50.0", "--token", "hf_test", "--password", "secret"]
            with patch.object(sys, 'argv', test_args):
                neuroseal.cli.main()
            
            mock_apply_lock.assert_called_with("input_model", "output_dir", 50.0, token="hf_test", password="secret")

if __name__ == "__main__":
    unittest.main()
