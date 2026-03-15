from shared.validation import validate_features

VALID = {"age": 35.0, "income": 55000.0, "credit_score": 720.0, "debt_ratio": 1.2, "num_accounts": 5.0}

def test_valid():
    assert validate_features(VALID) is None

def test_missing():
    assert validate_features({"age": 35.0}) is not None

def test_extra():
    bad = {**VALID, "extra": 1.0}
    assert any(e["field"] == "extra" for e in validate_features(bad))

def test_below_min():
    bad = {**VALID, "age": 5.0}
    assert any(e["field"] == "age" for e in validate_features(bad))

def test_above_max():
    bad = {**VALID, "credit_score": 900.0}
    assert any(e["field"] == "credit_score" for e in validate_features(bad))
