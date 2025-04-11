from functools import partial
from .csqa import csqa_dataset
from .race import race_dataset
from .obqa import openbookqa_dataset
from .arc import arc_dataset
from .tqa import tqa_dataset
from repe_eval.tasks.medical.medqa import medqa_dataset
from repe_eval.tasks.medical.medmcqa import medmcqa_dataset
from repe_eval.tasks.medical.pubmedqa import pubmedqa_dataset

def task_dataset(task):
    datasets_function = {
        'csqa': csqa_dataset,
        'race': race_dataset,
        'obqa': openbookqa_dataset,
        'arc_easy': partial(arc_dataset, 'ARC-Easy'),
        'arc_challenge': partial(arc_dataset, 'ARC-Challenge'),
        'tqa': tqa_dataset,
        'medqa': medqa_dataset,
        'medmcqa': medmcqa_dataset,
        'pubmedqa': pubmedqa_dataset,
    }

    assert task in datasets_function, f"{task} not implemented"
    return datasets_function[task]