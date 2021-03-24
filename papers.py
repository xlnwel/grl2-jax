from collections import defaultdict
Month = [
    '', 
    'January', 
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
]
def computer_vision():
    papers = {
        'computer vision': {
            Month[3]: {
                'relation': 1,
                'layer': 1,
                'network': 4,
            }
        }
    }
    return papers

def sequential_model():
    papers = {
        'sequential_model': {
            Month[3]: {
                'nlp': 1,
            }
        }
    }
    return papers

def reinforcement_learning():
    papers = {
        'reinforcement_learning': {
            Month[3]: {
                'policy gradient': 2,                # policy gradient
                'generalization': 1,
                'marl': 6,              # multi-agent RL
                'neuron science': 1,                # neuron science
                'auxiliary task': 1,
                'tricks': 2,
            }
    }
    }
    return papers

if __name__ == '__main__':
    papers = {
        **computer_vision(),
        **sequential_model(),
        **reinforcement_learning(),
    }
    month_summary = defaultdict(dict)
    month_total = {}
    total = 0
    for m in Month[1:]:
        month_total[m] = 0
        for field_name, field in papers.items():
            if m not in field:
                continue
            month_summary[m][field_name] = sum(field[m].values())
            month_total[m] += month_summary[m][field_name]
        total += month_total[m]
    print(f'Total: {total}')
    for m, field in month_summary.items():
        print(f'{m}: {month_total[m]}')
        for field_name, n in field.items():
            print(f'\t{field_name}: {n}')
    
    # print(f'Total number of papers read: {total}')