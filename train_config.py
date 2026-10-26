"""Shared training configuration so train_model.py and retrain.py agree."""

FEATURES = [
    'duration', 'credit_amount', 'installment_commitment',
    'residence_since', 'age', 'existing_credits', 'num_dependents',
]
TARGET = 'class'
LABEL_MAP = {'good': 1, 'bad': 0}
