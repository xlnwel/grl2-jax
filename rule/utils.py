from core.typing import ModelPath, get_basic_model_name

def is_rule_strategy(model: ModelPath):
    name = get_basic_model_name(model.model_name)
    return name.endswith('rule')
