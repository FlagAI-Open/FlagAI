# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import unittest

print('test syn')
test_dir = './tests'
test_report_path = './test_report'
discover = unittest.defaultTestLoader.discover(test_dir, pattern='test_*.py')
with open(test_report_path, "w") as report_file:
    runner = unittest.TextTestRunner(stream=report_file, verbosity=2)
    #runner=unittest.TextTestRunner()
    runner.run(discover)
