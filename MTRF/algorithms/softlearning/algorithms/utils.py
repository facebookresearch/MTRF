from copy import deepcopy


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC
    algorithm = SAC(*args, **kwargs)
    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL
    algorithm = SQL(*args, **kwargs)
    return algorithm


def create_MultiSAC_algorithm(variant, *args, **kwargs):
    from .multi_sac import MultiSAC
    algorithm = MultiSAC(*args, **kwargs)
    return algorithm


def create_PhasedSAC_algorithm(variant, *args, **kwargs):
    from .phased_sac import PhasedSAC
    algorithm = PhasedSAC(*args, **kwargs)
    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'MultiSAC': create_MultiSAC_algorithm,
    'PhasedSAC': create_PhasedSAC_algorithm,
    'SQL': create_SQL_algorithm,
}


def get_algorithm_from_variant(variant,
                               *args,
                               **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']

    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    algorithm = ALGORITHM_CLASSES[algorithm_type](
        variant, *args, **algorithm_kwargs, **kwargs)

    return algorithm
