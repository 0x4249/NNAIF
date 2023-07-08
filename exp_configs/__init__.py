# Local imports
from . import low_dimensional_cutest_exps
from . import medium_dimensional_cutest_exps
from . import high_dimensional_cutest_exps
from . import very_high_dimensional_cutest_exps

# Low dimensional (0 < d <= 50) experiments from CUTEst test problem set
LOW_DIM_CUTEST_EXPS = {}
LOW_DIM_CUTEST_EXPS.update(low_dimensional_cutest_exps.LOW_DIM_CUTEST_EXP_GROUPS)

# Medium dimensional (50 < d <= 200) experiments from CUTEst test problem set
MED_DIM_CUTEST_EXPS = {}
MED_DIM_CUTEST_EXPS.update(medium_dimensional_cutest_exps.MED_DIM_CUTEST_EXP_GROUPS)

# High dimensional (200 < d <= 3000) experiments from CUTEst test problem set
HIGH_DIM_CUTEST_EXPS = {}
HIGH_DIM_CUTEST_EXPS.update(high_dimensional_cutest_exps.HIGH_DIM_CUTEST_EXP_GROUPS)

# Very high dimensional (d > 3000) experiments from CUTEst test problem set
VERY_HIGH_DIM_CUTEST_EXPS = {}
VERY_HIGH_DIM_CUTEST_EXPS.update(very_high_dimensional_cutest_exps.VERY_HIGH_DIM_CUTEST_EXP_GROUPS)
