# shared/schemas/feature_schema.py
FEATURE_SCHEMA = {
    "age": {"type": "float", "min": 18.0, "max": 120.0, "nullable": False},
    "income": {"type": "float", "min": 0.0, "max": 1e7, "nullable": False},
    "credit_score": {"type": "float", "min": 300.0, "max": 850.0, "nullable": False},
    "debt_ratio": {"type": "float", "min": 0.0, "max": 5.0, "nullable": False},
    "num_accounts": {"type": "float", "min": 0.0, "max": 50.0, "nullable": False},
}

NUM_CLASSES = 3
EMBEDDING_DIM = 64
FEATURE_NAMES = sorted(FEATURE_SCHEMA.keys())
INPUT_DIM = len(FEATURE_NAMES)
