def _stub(name):
    def f(*a, **k):
        raise RuntimeError(
            f"openmapflow.{name} was actually called - the shim only exists to satisfy "
            "an unused import in presto.dataops. This code path is not supported here.")
    return f

calculate_ndvi = _stub("engineer.calculate_ndvi")
load_tif = _stub("engineer.load_tif")
remove_bands = _stub("engineer.remove_bands")
