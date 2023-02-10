import random
import numpy as np
import gym, os
import cv2

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from env.utils import compute_aid2uids
from core.typing import AttrDict2dict


REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 1,
    "DISH_PICKUP_REWARD": 0,
    "SOUP_PICKUP_REWARD": 0,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0
}


def compute_pots_ingradients(terrain):
    pots_pos = []
    n_pots = 0
    n_onions = 0
    n_tomatoes = 0
    for y, row in enumerate(terrain):
        for x, i in enumerate(row):
            if i == 'P':
                n_pots += 1
                pots_pos.append((x, y))
            elif i == 'O':
                n_onions += 1
            elif i == 'T':
                n_tomatoes += 1
    assert len(pots_pos) == n_pots
    return pots_pos, n_pots, n_onions, n_tomatoes


class Overcooked:
    def __init__(self, config):
        config = AttrDict2dict(config)
        self.name = config['env_name'].split('-', 1)[-1]
        self._mdp = OvercookedGridworld.from_layout_name(
            layout_name=self.name, **config.get('layout_params', {}), rew_shaping_params=REW_SHAPING_PARAMS)
        self._env = OvercookedEnv.from_mdp(self._mdp, horizon=config['max_episode_steps'], info_level=0)

        self.max_episode_steps = config['max_episode_steps']
        self.dense_reward = config.get('dense_reward', False)
        self._add_goal = config.get('add_goal', False)
        self._featurize = config.get('featurize', False)
        if self._featurize:
            mlp = MediumLevelActionManager.from_pickle_or_compute(self._mdp, NO_COUNTERS_PARAMS)
            self.featurize_fn = lambda x: np.stack(self._mdp.featurize_state(x, mlp))

        self.pots_pos, self.n_pots, self.n_onions, self.n_tomatoes = compute_pots_ingradients(self._mdp.terrain_mtx)
        self.goal_size = self.n_pots * 2

        self.uid2aid = config.get('uid2aid', [0, 1])
        self.aid2uids = compute_aid2uids(self.uid2aid)
        self.n_units = len(self.uid2aid)
        self.n_agents = len(self.aid2uids)

        self.action_space = [gym.spaces.Discrete(len(Action.ALL_ACTIONS)) 
            for _ in range(self.n_agents)]
        self.action_shape = [a.shape for a in self.action_space]
        self.action_dim = [a.n for a in self.action_space]
        self.action_dtype = [np.int32
            for _ in range(self.n_agents)]
        self.is_action_discrete = [True for _ in range(self.n_agents)]

        self.obs_shape = self._get_observation_shape()
        self.obs_dtype = self._get_observation_dtype()

        self._render_initialized = False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    # def get_screen(self, **kwargs):
    #     """
    #     Standard way to view the state of an esnvironment programatically
    #     is just to print the Env object
    #     """
    #     return self._env.__repr__()
    
    def random_action(self):
        return np.random.randint(0, self.action_dim, self.n_units)

    def _get_observation_shape(self):
        dummy_state = self._mdp.get_standard_start_state()
        if self._featurize:
            shape = self._env.featurize_state_mdp(dummy_state)[0].shape
            obs_shape = [dict(
                obs=shape,
                global_state=shape, 
            ) for _ in self.aid2uids]
        else:
            shape = self._env.lossless_state_encoding_mdp(dummy_state)[0].shape
            obs_shape = [dict(
                obs=shape,
            ) for _ in self.aid2uids]
        if self._add_goal:
            for shape in obs_shape:
                shape['goal'] = (self.goal_size,)
        return obs_shape

    def _get_observation_dtype(self):
        if self._featurize:
            obs_dtype = [dict(
                obs=np.float32,
                global_state=np.float32,
            ) for _ in self.aid2uids]
        else:
            obs_dtype = [dict(
                obs=np.float32,
            ) for _ in self.aid2uids]
        if self._add_goal:
            for dtype in obs_dtype:
                dtype['goal'] = np.float32
        return obs_dtype

    def reset(self):
        self._env.reset()
        obs = self._get_obs(self._env.state)
        self._score = np.zeros(self.n_units, dtype=np.float32)
        self._dense_score = np.zeros(self.n_units, dtype=np.float32)
        self._epslen = 0

        return obs

    def step(self, action):
        assert len(action) == 2, action
        real_action = Action.ALL_ACTIONS[action[0]], Action.ALL_ACTIONS[action[1]]
        state, reward, done, info = self._env.step(real_action)
        rewards = reward * np.ones(self.n_units, np.float32)
        self._score += rewards
        self._epslen += 1
        if self.dense_reward:
            dense_reward = max(info['shaped_r_by_agent'])
            rewards += dense_reward * np.ones(self.n_units, np.float32)
        # else:
        #     print(reward, info['sparse_r_by_agent'])
        self._dense_score += rewards
        obs = self._get_obs(state, action)
        dones = done * np.ones(self.n_units, np.float32)
        info = dict(
            score=self._score,
            epslen=self._epslen,
            dense_score=self._dense_score,
            game_over=done,
        )

        rewards = [rewards[uids] for uids in self.aid2uids]
        dones = [dones[uids] for uids in self.aid2uids]

        return obs, rewards, dones, info

    def _get_obs(self, state, action=None):
        if self._featurize:
            obs = self._env.featurize_state_mdp(state)
            obs = [np.expand_dims(o, 0).astype(np.float32) for o in obs]
            obs = [dict(
                obs=o, 
                global_state=o, 
            ) for o in obs]
        else:
            obs = [dict(
                obs=np.expand_dims(o, 0).astype(np.float32)) 
                for o in self._env.lossless_state_encoding_mdp(state)]
        if self._add_goal:
            goal = self._get_pots_status()
            for o in obs:
                o['goal'] = np.expand_dims(goal, 0)
        return obs

    def _get_pots_status(self):
        goal = np.ones(self.goal_size, np.float32)
        for i, pos in enumerate(self.pots_pos):
            if pos in self._env.state.objects:
                soup = self._env.state.objects[pos]
                for x in soup.ingredients:
                    if x == 'tomato':
                        goal[2*i] -= 1
                    elif x == 'onion':
                        goal[2*i+1] -= 1
        return goal

    def render_init(self):
        """Do initial work for rendering. Currently we don't support tomatoes
        """
        self._render_initialized = True
        
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overcooked_env/assets")
        chefs_dir = os.path.join(base_dir, "chefs")
        objects_dir = os.path.join(base_dir, "objects")
        terrain_dir = os.path.join(base_dir, "terrain")
        
        self.onion_time = 15
        self.tomato_time = 7

        def block_read(file_path):
            """
            read block image
            """
            return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        def build_chef(chef_dict, direction, color, background):
            """
            build information of chef
            """
            # load basic blocks
            chef_arr = block_read(os.path.join(chefs_dir, f"{direction}.png"))
            chef_dish = block_read(os.path.join(chefs_dir, f"{direction}-dish.png"))
            chef_tomato = block_read(os.path.join(chefs_dir, f"{direction}-tomato.png"))
            chef_stomato = block_read(os.path.join(chefs_dir, f"{direction}-soup-tomato.png"))
            chef_onion = block_read(os.path.join(chefs_dir, f"{direction}-onion.png"))
            chef_sonion = block_read(os.path.join(chefs_dir, f"{direction}-soup-onion.png"))
            hat_arr = block_read(os.path.join(chefs_dir, f"{direction}-{color}hat.png"))
            # compute masks
            hat_mask = (hat_arr[:, :, -1] != 0)[:, :, None]
            chefs_mask = (chef_arr[:, :, -1] != 0)[:, :, None]

            def blocks_overlay(body_arr):
                """overy hat_block, chef_block and background"""
                return (hat_mask * hat_arr + (1 - hat_mask) * body_arr) * chefs_mask + (1 - chefs_mask) * background

            chef_dict[direction]["ept"] = blocks_overlay(chef_arr)
            chef_dict[direction]["dish"] = blocks_overlay(chef_dish)
            chef_dict[direction]["onion"] = blocks_overlay(chef_onion)
            chef_dict[direction]["sonion"] = blocks_overlay(chef_sonion)
            chef_dict[direction]["tomato"] = blocks_overlay(chef_tomato)
            chef_dict[direction]["stomato"] = blocks_overlay(chef_stomato)

        # basic terrain blocks
        counter_arr = block_read(os.path.join(terrain_dir, "counter.png"))
        floor_arr = block_read(os.path.join(terrain_dir, "floor.png"))
        onions_arr = block_read(os.path.join(terrain_dir, "onions.png"))
        tomatoes_arr = block_read(os.path.join(terrain_dir, "tomatoes.png"))
        dishes_arr = block_read(os.path.join(terrain_dir, "dishes.png"))
        pot_arr = block_read(os.path.join(terrain_dir, "pot.png"))
        serve_arr = block_read(os.path.join(terrain_dir, "serve.png"))
        self.counter_arr, self.pot_arr = counter_arr, pot_arr

        # define label2img
        label2img = {
            "X": counter_arr,
            " ": floor_arr,
            "O": onions_arr,
            "D": dishes_arr,
            "P": pot_arr,
            "S": serve_arr,
            "T": tomatoes_arr,
        }

        # define terrain array
        self.block_size = (15, 15, 4)
        self.block_h, self.block_w = self.block_size[0], self.block_size[1]
        H, W = len(self._mdp.terrain_mtx), len(self._mdp.terrain_mtx[0])
        self.terrain_arr = np.zeros((H*self.block_h, W*self.block_w, self.block_size[2]))
        self.terrain_arr[:, :] = [153, 178, 199, 255]
        for row_idx, row in enumerate(self._mdp.terrain_mtx):
            for col_idx, ele in enumerate(row):
                self.terrain_arr[row_idx*self.block_h:(row_idx+1)*self.block_h, col_idx*self.block_w:(col_idx+1)*self.block_w] = label2img[ele]    
        
        # blocks relating to chefs
        self.blue_chef = {direction: {"ept": None, "onion": None, "dish": None, "sonion": None, "tomato": None, "stomato": None} for direction in ["SOUTH", "NORTH", "EAST", "WEST"]}
        self.green_chef = {direction: {"ept": None, "onion": None, "dish": None, "sonion": None, "tomato": None, "stomato": None} for direction in ["SOUTH", "NORTH", "EAST", "WEST"]}
        for direction in ["SOUTH", "NORTH", "EAST", "WEST"]:
            build_chef(self.blue_chef, direction, "blue", floor_arr)
            build_chef(self.green_chef, direction, "green", floor_arr)
        self.chefs = [self.blue_chef, self.green_chef]

        # get item blocks
        self.ob_dish_arr = block_read(os.path.join(objects_dir, "dish.png"))
        self.ob_onion_arr = block_read(os.path.join(objects_dir, "onion.png"))
        self.ob_tomato_arr = block_read(os.path.join(objects_dir, "tomato.png"))
        self.ob_pot_exp_arr = block_read(os.path.join(objects_dir, "pot-explosion.png"))
        self.ob_onion_1_arr = block_read(os.path.join(objects_dir, "soup-onion-1-cooking.png"))
        self.ob_onion_2_arr = block_read(os.path.join(objects_dir, "soup-onion-2-cooking.png"))
        self.ob_onion_3_arr = block_read(os.path.join(objects_dir, "soup-onion-3-cooking.png"))
        self.ob_onion_cooked_arr = block_read(os.path.join(objects_dir, "soup-onion-cooked.png"))
        self.ob_onion_dish_arr = block_read(os.path.join(objects_dir, "soup-onion-dish.png"))
        self.ob_tomato_1_arr = block_read(os.path.join(objects_dir, "soup-tomato-1-cooking.png"))
        self.ob_tomato_2_arr = block_read(os.path.join(objects_dir, "soup-tomato-2-cooking.png"))
        self.ob_tomato_3_arr = block_read(os.path.join(objects_dir, "soup-tomato-3-cooking.png"))
        self.ob_tomato_cooked_arr = block_read(os.path.join(objects_dir, "soup-tomato-cooked.png"))
        self.ob_tomato_dish_arr = block_read(os.path.join(objects_dir, "soup-tomato-dish.png"))

        # Orientation
        self.tuple2direction = {
            Direction.NORTH: "NORTH",
            Direction.SOUTH: "SOUTH",
            Direction.EAST: "EAST",
            Direction.WEST: "WEST",
        }

    def get_screen(self):
        """Function for the env's rendering.
        """
        if not self._render_initialized:
            self.render_init()

        def embed_arr(sub_arr, background_arr):
            """
            Embed sub_arr into the background_arr.
            """
            mask = (sub_arr[:, :, -1] != 0)[:, :, None]
            return mask * sub_arr + (1 - mask) * background_arr
        
        players_dict = {player.position: player for player in self._env.state.players}
        frame = self.terrain_arr.copy()
        for y, terrain_row in enumerate(self._mdp.terrain_mtx):
            for x, element in enumerate(terrain_row):
                if (x, y) in players_dict:
                    player = players_dict[(x, y)]
                    orientation = player.orientation
                    assert orientation in Direction.ALL_DIRECTIONS
                    
                    direction_name = self.tuple2direction[orientation]
                    player_object = player.held_object
                    if player_object:
                        # TODO: how to deal with held objects
                        player_idx_lst = [i for i, p in enumerate(self._env.state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        if player_object.name == "onion":
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["onion"]
                        elif player_object.name == "tomato":
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["tomato"]
                        elif player_object.name == "dish":
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                self.chefs[player_idx_lst[0]][direction_name]["dish"]
                        elif player_object.name == "soup":
                            assert not player_object.is_idle
                            soup_type = player_object.ingredients[0]
                            if soup_type == "onion":
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                    self.chefs[player_idx_lst[0]][direction_name]["sonion"]
                            elif soup_type == "tomato":
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = \
                                    self.chefs[player_idx_lst[0]][direction_name]["stomato"]
                            else:
                                assert 0, soup_type
                        else:
                            raise ValueError(f"Unsupported player_object.name {player_object.name}")
                    else:
                        player_idx_lst = [i for i, p in enumerate(self._env.state.players) if p.position == player.position]
                        assert len(player_idx_lst) == 1
                        frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = self.chefs[player_idx_lst[0]][direction_name]["ept"]
                else:
                    if element == "X" and self._env.state.has_object((x, y)):
                        counter_obj = self._env.state.get_object((x, y))
                        if counter_obj.name == "onion":
                            dynamic_arr = embed_arr(self.ob_onion_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif counter_obj.name == "tomato":
                            dynamic_arr = embed_arr(self.ob_tomato_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif counter_obj.name == "dish":
                            dynamic_arr = embed_arr(self.ob_dish_arr, self.counter_arr)
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        elif counter_obj.name == "soup":
                            assert not counter_obj.is_idle
                            soup_type = counter_obj.ingredients[0]
                            if soup_type == "onion":
                                dynamic_arr = embed_arr(self.ob_onion_dish_arr, self.counter_arr)
                            elif soup_type == "tomato":
                                dynamic_arr = embed_arr(self.ob_tomato_dish_arr, self.counter_arr)
                            else:
                                assert 0, soup_type
                            frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                        else:
                            raise ValueError(f"Unsupported object name on counter {counter_obj.name}")
                    elif element == "P" and self._env.state.has_object((x, y)):
                        soup_obj = self._env.state.get_object((x, y))
                        if soup_obj.is_idle:
                            continue
                        ingredients = soup_obj.ingredients
                        soup_type, num_items = ingredients[0], len(ingredients)
                        assert soup_type in ["onion", "tomato"], "Currently we only support the visualization of onion/tomato type."
                        if soup_type == "onion":
                            if num_items == 1:
                                dynamic_arr = embed_arr(self.ob_onion_1_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            elif num_items == 2:
                                dynamic_arr = embed_arr(self.ob_onion_2_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            elif num_items >= 3:
                                if not soup_obj.is_ready:
                                    dynamic_arr = embed_arr(self.ob_onion_3_arr, self.pot_arr)
                                    frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                                else:
                                    dynamic_arr = embed_arr(self.ob_onion_cooked_arr, self.pot_arr)
                                    frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            else:
                                raise ValueError(f"Invalid num_items for pot {num_items}") 
                        elif soup_type == "tomato":
                            if num_items == 1:
                                dynamic_arr = embed_arr(self.ob_tomato_1_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            elif num_items == 2:
                                dynamic_arr = embed_arr(self.ob_tomato_2_arr, self.pot_arr)
                                frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            elif num_items >= 3:
                                if not soup_obj.is_ready:
                                    dynamic_arr = embed_arr(self.ob_tomato_3_arr, self.pot_arr)
                                    frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                                else:
                                    dynamic_arr = embed_arr(self.ob_tomato_cooked_arr, self.pot_arr)
                                    frame[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w] = dynamic_arr
                            else:
                                raise ValueError(f"Invalid num_items for pot {num_items}") 
                            
        return frame[:, :, :-1][:, :, ::-1]/255.
        # # self.viewer.imshow(frame[:, :, :-1]/255.)
        # if not self.is_remote:
        #     cv2.imshow('window', frame[:, :, :-1]/255.)
        #     cv2.waitKey(1)
        # else:
        #     cv2.imwrite(os.path.join(self.img_save_dir, f"{self.t_ep}.png"), frame[:, :, :-1])
        #     self.t_ep += 1
        #     if done:
        #         # transform imgs into video
        #         video_path = os.path.join(self.video_save_dir, "video0.avi")
        #         self.img2video(self.img_save_dir, video_path)

    def close(self):
        pass


if __name__ == '__main__':
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('env', type=str)
        parser.add_argument('--interactive', '-i', action='store_true')
        args = parser.parse_args()
        return args
    args = parse_args()
    config = dict(
        env_name=args.env,
        max_episode_steps=400,
        dense_reward=True,
        featurize=True,
        add_goal=False
        # layout_params={
        #     'onion_time': 1,
        # }
    )
    def action2char(action):
        dic = {
            'w': (0, -1),
            's': (0, 1),
            'a': (-1, 0),
            'd': (1, 0),
            'q': (0, 0),
            'e': 'interact',
        }
        a1, a2 = dic[action[0]], dic[action[1]]
        return Action.ACTION_TO_CHAR[a1], Action.ACTION_TO_CHAR[a2]
    def action2array(action):
        dic = {
            'w': 0,
            's': 1,
            'a': 3,
            'd': 2,
            'q': 4,
            'e': 5,
        }
        return np.array([[dic[action[0]]], [dic[action[1]]]])

    env = Overcooked(config)
    obs = env.reset()
    print(obs[0]['obs'].max(), obs[0]['obs'].min())
    d = False
    while not np.all(d):
        # print(env.get_screen())
        if args.interactive:
            a = input('action: ').split(' ')
        else:
            a = env.random_action()
        o, r, d, i = env.step(a)
        # print(o['goal'])
        print("Curr reward: (sparse)", i['sparse_reward'], "\t(dense)", i['dense_reward'])
        print('Reward', r)
