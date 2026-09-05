def _stub(name):
    def f(*a, **k):
        raise RuntimeError(
            f"openmapflow.{name} was actually called - the shim only exists to satisfy "
            "an unused import in presto.dataops. This code path is not supported here.")
    return f

create_ee_image = _stub("ee_exporter.create_ee_image")
ee_safe_str = _stub("ee_exporter.ee_safe_str")
get_ee_task_list = _stub("ee_exporter.get_ee_task_list")
