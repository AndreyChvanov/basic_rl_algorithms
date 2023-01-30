# For register environments
from gym.envs.registration import register

# Register default env
register(
    id="default-v0",
    entry_point='frozen_lake.envs:Default'
)

# Register fall env
register(
    id="fall-v0",
    entry_point='frozen_lake.envs:Fall'
)
