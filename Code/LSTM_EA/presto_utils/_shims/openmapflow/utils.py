def _stub(name):
    def f(*a, **k):
        raise RuntimeError(
            f"openmapflow.{name} was actually called - the shim only exists to satisfy "
            "an unused import in presto.dataops. This code path is not supported here.")
    return f

def memoized(f):
    return f
