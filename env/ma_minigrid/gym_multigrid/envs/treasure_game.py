from ..multigrid import *

class TreasureGameEnv(MultiGridEnv):
    """
    Environment where the agents fetch the key and open the treasure.
    """
    
    def __init__(
        self,
        size=None,
        view_size=3,
        max_episode_steps=100,
        width=None,
        height=None,
        seed=7,
        partial_obs=False,
        num_agents=2,
        num_keys=1,
        num_treasures=1,
        **kwargs,
    ):
        self.num_agents = num_agents
        self.num_keys = num_keys
        self.num_treasures = num_treasures
        assert self.num_treasures == 1, "Currently support one treasure."
        self.pickup_key_rew = 1
        self.open_treasure_rew = 5

        self.world = World

        agents = []
        agents_index = [i for i in range(self.num_agents)]
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size))

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps=max_episode_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed,
            agents=agents,
            agent_view_size=view_size,
            partial_obs=partial_obs,
        )

        assert not self.partial_obs
        self.observation_space = spaces.Box(
            low=-1,
            high=4,
            shape=(width * height + 2,),
            # dtpye='uint8',
        )
        self.ob_dim = width*height + 2
        self.state_dim = width*height + 2*self.num_agents

    def get_state(self):
        # wall: -1, floor: 0, key: 1, treasure: 2, ego_agent: 4.
        state = np.zeros(self.width * self.height)
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is None:
                    continue
                elif cell.type == "wall":
                    state[j * self.width + i] = -1
                elif cell.type == "key":
                    state[j * self.width + i] = 1
                elif cell.type == "box":
                    state[j * self.width + i] = 2
                elif cell.type == "agent":
                    state[j * self.width + i] = 4            
                else:
                    assert 0, cell
        state = np.hstack([state] + [self.agents[agent_idx].dir_vec for agent_idx in range(self.num_agents)])
        return state

    def _obs_for_agents(self, agent_idx):
        # wall: -1, floor: 0, key: 1, treasure: 2, ego_agent: 3, other_agent: 4
        ob = np.zeros(self.width * self.height)
        
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid.get(i, j)
                if cell is None:
                    continue
                elif cell.type == "wall":
                    ob[j * self.width + i] = -1
                elif cell.type == "key":
                    ob[j * self.width + i] = 1
                elif cell.type == "box":
                    ob[j * self.width + i] = 2
                elif cell.type == "agent":
                    if self.agents[agent_idx].pos[0] == i \
                        and self.agents[agent_idx].pos[1] == j:
                        ob[j * self.width + i] = 3
                    else:
                        ob[j * self.width + i] = 4            
                else:
                    assert 0, cell
        ob = np.hstack([ob, self.agents[agent_idx].dir_vec])
        return ob

    def reset(self):

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        assert not self.partial_obs
        obs = [self._obs_for_agents(i) for i in range(self.num_agents)]

        return obs

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)

        for _ in range(self.num_keys):
            self.place_obj(Key(self.world, color='yellow'))
        self.key_picked = False

        for _ in range(self.num_treasures):
            self.treasure_pos = self.place_obj(Box(self.world, color='blue'))

        # Randomize the player start position and orientation
        for a in self.agents:
            self.place_agent(a)

    def _reset_resources(self):
        # del treasure box
        treasure_box = self.grid.get(*self.treasure_pos)
        self.grid.set(*self.treasure_pos, None)
        del treasure_box

        # del key
        for agent in self.agents:
            if agent.carrying:
                assert agent.carrying.type == "key"
                del agent.carrying
                agent.carrying = None
                
        for _ in range(self.num_keys):
            self.place_obj(Key(self.world, color='yellow'))

        self.key_picked = False

        for _ in range(self.num_treasures):
            self.treasure_pos = self.place_obj(Box(self.world, color='blue'))
    
    def _reward(self, i, rewards, reward=1):
        for j in range(len(rewards)):
            rewards[j] += reward
    
    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.type == "key":
                    self.key_picked = True
                    self._reward(i, rewards, self.pickup_key_rew)
                elif fwd_cell.type == "box":
                    return True # try open the treasure box
                else:
                    assert 0, fwd_cell.type
                if self.agents[i].carrying is None:
                    self.agents[i].carrying = fwd_cell
                    self.agents[i].carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
            elif fwd_cell.type=='agent':
                if fwd_cell.carrying:
                    if self.agents[i].carrying is None:
                        self.agents[i].carrying = fwd_cell.carrying
                        fwd_cell.carrying = None

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        if self.agents[i].carrying:
            if fwd_cell:
                if fwd_cell.type == 'objgoal' and fwd_cell.target_type == self.agents[i].carrying.type:
                    assert 0, "no objgoal in this map"
                    if self.agents[i].carrying.index in [0, fwd_cell.index]:
                        self._reward(fwd_cell.index, rewards, fwd_cell.reward)
                        self.agents[i].carrying = None
                elif fwd_cell.type=='agent':
                    if fwd_cell.carrying is None:
                        fwd_cell.carrying = self.agents[i].carrying
                        self.agents[i].carrying = None
            else:
                if self.agents[i].carrying.type == "key":
                    self.key_picked = False
                    self._reward(i, rewards, -self.pickup_key_rew)
                self.grid.set(*fwd_pos, self.agents[i].carrying)
                self.agents[i].carrying.cur_pos = fwd_pos
                self.agents[i].carrying = None

    def step(self, actions):
        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        try_open_treasures = [False, False]
        for i in order:

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[i] == self.actions.still:
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                if fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        assert 0, "No type goal in this map."
                        done = True
                        self._reward(i, rewards, 1)
                    elif fwd_cell.type == 'switch':
                        assert 0, "No type switch in this map."
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                elif fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif 'build' in self.actions.available and actions[i]==self.actions.build:
                self._handle_build(i, rewards, fwd_pos, fwd_cell)

            # Pick up an object
            elif actions[i] == self.actions.pickup:
                try_open_treasure = self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
                try_open_treasures[i] = True if try_open_treasure else False
            
            # Drop an object
            elif actions[i] == self.actions.drop:
                self._handle_drop(i, rewards, fwd_pos, fwd_cell)

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        if try_open_treasures[0] and try_open_treasures[1] and self.key_picked:
            self._reward(0, rewards, self.open_treasure_rew)
            self._reset_resources()
            
        if self.step_count >= self.max_steps:
            done = True

        assert not self.partial_obs
        obs = [self._obs_for_agents(i) for i in range(self.num_agents)]

        return obs, rewards, done, {}


class TreasureGameEnv10x15N2(TreasureGameEnv):
    def __init__(self):
        super().__init__(size=None,
        height=8,
        width=7)