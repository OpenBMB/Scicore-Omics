class _GatheredParameters:
    def __init__(self, params):
        self.params = params

    def __enter__(self):
        return self.params

    def __exit__(self, exc_type, exc, tb):
        return False


class _Zero:
    GatheredParameters = _GatheredParameters


zero = _Zero()
