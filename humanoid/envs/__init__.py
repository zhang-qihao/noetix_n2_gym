from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from humanoid.utils.task_registry import task_registry

# ---------------------------------------------- Base ----------------------------------------------
from .n2.n2_env import N2Env
from .n2.n2_config import N2_10DofCfg, N2_10DofCfgPPO
from .n2.n2_dwl_config import N2DWLRoughCfg, N2DWLRoughCfgPPO

task_registry.register( "n2", N2Env, N2_10DofCfg(), N2_10DofCfgPPO() )
task_registry.register( "n2_dwl", N2Env, N2DWLRoughCfg(), N2DWLRoughCfgPPO() )

# ---------------------------------------------- AMP ----------------------------------------------
from .n2.n2_amp_env import N2AMPEnv
from .n2.n2_10dof_amp_config import N2_10DofAMPCfg, N2_10DofAMPCfgPPO
from .n2.n2_18dof_amp_config import N2_18DofAMPCfg, N2_18DofAMPCfgPPO
from .n2.n2_20dof_amp_config import N2_20DofAMPCfg, N2_20DofAMPCfgPPO

task_registry.register( "n2_10dof_amp", N2AMPEnv, N2_10DofAMPCfg(), N2_10DofAMPCfgPPO() )
task_registry.register( "n2_18dof_amp", N2AMPEnv, N2_18DofAMPCfg(), N2_18DofAMPCfgPPO() )
task_registry.register( "n2_20dof_amp", N2AMPEnv, N2_20DofAMPCfg(), N2_20DofAMPCfgPPO() )









