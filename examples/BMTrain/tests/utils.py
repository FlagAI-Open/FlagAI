def assert_eq(a, b):
    assert a == b, f"{a} != {b}"

def assert_neq(a, b):
    assert a != b, f"{a} == {b}"

def assert_lt(a, b):
    assert a < b, f"{a} >= {b}"

def assert_gt(a, b):
    assert a > b, f"{a} <= {b}"

def assert_all_eq(a, b):
    assert_eq((a==b).all(), True)