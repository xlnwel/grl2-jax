### State Dynamics v1

<!-- We learn three independent function approximators using MLPs
1. (obs, action) -> next_obs
2. (obs, action) -> reward
3. (next_obs) -> discount -->

Learn three independent function approximators as follows:
1. (env_state, action) -> next_env_state
2. (next_env_state) -> next_agent_state, next_agent_obs
3. (env_state, action) -> reward
4. (next_env_state) -> discount
