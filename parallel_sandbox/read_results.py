from attrbench.metrics.deletion.result import DeletionResult
from attrbench.metrics.impact_coverage.result import ImpactCoverageResult
from attrbench.metrics.max_sensitivity.result import MaxSensitivityResult
from attrbench.metrics.minimal_subset.result import MinimalSubsetResult
from attrbench.metrics.sensitivity_n.result import SensitivityNResult


if __name__ == "__main__":
    coverage = ImpactCoverageResult.load("coverage.h5")
    deletion = DeletionResult.load("deletion.h5")
    irof = DeletionResult.load("irof.h5")
    maxsens = MaxSensitivityResult.load("maxsens.h5")
    minimal_subset = MinimalSubsetResult.load("minimal_subset.h5")
    sens_n = SensitivityNResult.load("sensn.h5")
    seg_sens_n = SensitivityNResult.load("segsensn.h5")
