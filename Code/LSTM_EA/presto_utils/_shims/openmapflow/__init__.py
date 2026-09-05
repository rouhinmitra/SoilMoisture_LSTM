"""Import-time shim for openmapflow.

presto.dataops.utils (which provides construct_single_presto_input) transitively
imports openmapflow through dataops.pipelines.s1_s2_era5_srtm, but only uses the
BAND-NAME CONSTANTS from that module.  The openmapflow functions are called only
inside TIF-fetching/EE-export methods that the embedding path never touches.

openmapflow 0.2.1 pins a pandas that will not cythonize on Python 3.12, so it
cannot be installed here.  These stubs satisfy the import and RAISE LOUDLY if
anything actually calls them - no silent wrong answers.
"""
