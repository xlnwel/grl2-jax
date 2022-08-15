import numpy as np


def do_flatten(obj):
    """Run flatten on either python list or numpy array."""
    if type(obj) == list:
        return np.array(obj).flatten()
    return obj.flatten()


class FeatureEncoder:
    OBS_SHAPE = (195,)
    ACTION_DIM = 19
    HIDDEN_SHAPE = (176,)
    def __init__(
        self, 
        aid2uids, 
        use_idx, 
        use_hidden, 
        use_event, 
        use_action_mask, 
        agentwise_global_state
    ):
        self.aid2uids = aid2uids
        assert len(self.aid2uids) in (1, 2), len(self.aid2uids)
        for uids in self.aid2uids:
            assert len(uids) == 11, len(uids)
        self.use_idx = use_idx
        self.use_hidden = use_hidden
        self.use_event = use_event
        self.use_action_mask = use_action_mask
        self.agentwise_global_state = agentwise_global_state

    def get_obs_shape(self, n_agents):
        n_units = 11
        shape = []
        for _ in range(n_agents):
            s = dict(
                obs=self.OBS_SHAPE, 
                global_state=self.OBS_SHAPE if self.agentwise_global_state else self.HIDDEN_SHAPE, 
                prev_reward=(), 
                prev_action=(self.ACTION_DIM,), 
            )
            if self.use_action_mask:
                s['action_mask'] = (self.ACTION_DIM,)
            if self.use_idx:
                s['idx'] = (n_units,)
            if self.use_hidden:
                s['hidden_state'] = self.HIDDEN_SHAPE
            if self.use_event:
                s['event'] = (3,)
            shape.append(s)

        return shape

    def get_obs_dtype(self, n_agents):
        dtype = []
        for _ in range(n_agents):
            d = dict(
                obs=np.float32, 
                global_state=np.float32, 
                prev_reward=np.float32, 
                prev_action=np.float32, 
            )
            if self.use_action_mask:
                d['action_mask'] = bool
            if self.use_idx:
                d['idx'] = np.float32
            if self.use_hidden:
                d['hidden_state'] = np.float32
            if self.use_event:
                d['event'] = np.float32
            dtype.append(d)

        return dtype

    def construct_observations(self, observations, action, reward):
        obs_array = []
        hidden_array = []
        for i, o in enumerate(observations):
            side = i >= 11
            uid = i - side * 11
            obs_array.append(self.get_obs(o, side, uid))
            if self.use_hidden or not self.agentwise_global_state:
                hidden_array.append(self.get_hidden_state(o))
        obs_array = np.stack(obs_array)
        final_obs = [dict(
            obs=obs_array[uids], 
            prev_reward=reward[aid], 
            prev_action=action[aid]
        ) for aid, uids in enumerate(self.aid2uids)]
        if self.use_idx:
            for obs in final_obs:
                obs['idx'] = np.eye(11, dtype=np.float32)
        if self.use_hidden or not self.agentwise_global_state:
            hidden_array = np.stack(hidden_array)
        if self.agentwise_global_state:
            for obs in final_obs:
                obs['global_state'] = obs['obs'].copy()
        else:
            for uids, obs in zip(self.aid2uids, final_obs):
                obs['global_state'] = hidden_array[uids].copy()
        if self.use_hidden:
            for uids, obs in zip(self.aid2uids, final_obs):
                obs['hidden_state'] = hidden_array[uids]

        return final_obs            

    def get_obs(self, obs, side, player_num):
        # if player_num is None:
        #     player_num = obs["active"]
        # # print(f'aid: {side}, uid: {player_num}')
        # assert player_num == obs['active'], (player_num, obs['active'], obs['left_team_active'])

        # player info
        team = 'left_team' if side == 0 else 'right_team'
        player_pos_x, player_pos_y = obs[team][player_num]
        player_pos = np.array([player_pos_x, player_pos_y], dtype=np.float32)
        player_direction = obs[f"{team}_direction"][player_num]
        player_speed = np.linalg.norm(player_direction)
        player_speed_vec = np.array([*player_direction, player_speed], dtype=np.float32)
        player_role = obs[f"{team}_roles"][player_num]
        player_role_onehot = self._encode_role_onehot(player_role)
        player_tired = obs[f"{team}_tired_factor"][player_num]
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]
        player_prop = np.array([player_tired, is_dribbling, is_sprinting], dtype=np.float32)
        player_state = np.concatenate(
            (
                player_pos,
                player_speed_vec,
                player_role_onehot,
                player_prop,
            )
        )
        assert player_state.shape == (18,), player_state

        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_xy_relative = [ball_x_relative, ball_y_relative]
        ball_distance = np.linalg.norm(ball_xy_relative)
        ball_dist_vec = np.array([*ball_xy_relative, ball_distance], dtype=np.float32)
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_xy_speed = [ball_x_speed, ball_y_speed]
        ball_speed = np.linalg.norm(ball_xy_relative)
        ball_speed_vec = np.array([*ball_xy_speed, ball_speed], dtype=np.float32)
        ball_owned = np.zeros(3, dtype=np.float32)
        ball_owned[obs["ball_owned_team"]] = 1
        ball_zone = self._encode_ball_which_zone(ball_x, ball_y)
        ball_close = np.array([ball_distance < 0.03], np.float32)
        ball_to_goal = np.array([ball_x + 1, ball_y], dtype=np.float32)
        ball_state = np.concatenate(
            (
                ball_dist_vec, 
                ball_speed_vec,
                ball_owned, 
                ball_zone, 
                ball_close, 
                ball_to_goal
            )
        )
        assert ball_state.shape == (18,), ball_state.shape

        if side == 0:
            obs_left_team = np.delete(obs["left_team"], player_num, axis=0)
            obs_left_team_direction = np.delete(
                obs["left_team_direction"], player_num, axis=0
            )
            left_team_tired = np.delete(
                obs["left_team_tired_factor"], player_num, axis=0
            ).reshape(-1, 1)
        else:
            obs_left_team = obs["left_team"]
            obs_left_team_direction = obs["left_team_direction"]
            left_team_tired = obs["left_team_tired_factor"].reshape(-1, 1)

        left_team_relative = obs_left_team - player_pos
        left_team_distance = np.linalg.norm(
            left_team_relative, axis=1, keepdims=True
        )
        left_team_dist_vec = np.concatenate([left_team_relative, left_team_distance], -1)
        left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)
        left_team_speed_vec = np.concatenate([obs_left_team_direction, left_team_speed], axis=-1)
        left_team_state = np.concatenate(
            (
                left_team_dist_vec,
                left_team_speed_vec,
                left_team_tired,
            ),
            axis=1,
        ).astype(np.float32)
        assert left_team_state.shape == (10 if side == 0 else 11, 7), left_team_state.shape

        if side == 1:
            obs_right_team = np.delete(obs["right_team"], player_num, axis=0)
            obs_right_team_direction = np.delete(
                obs["right_team_direction"], player_num, axis=0
            )
            right_team_tired = np.delete(
                obs["right_team_tired_factor"], player_num, axis=0
            ).reshape(-1, 1)
        else:
            obs_right_team = np.array(obs["right_team"])
            obs_right_team_direction = np.array(obs["right_team_direction"])
            right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        right_team_relative = obs_right_team - player_pos
        right_team_distance = np.linalg.norm(
            right_team_relative, axis=1, keepdims=True
        )
        right_team_dist_vec = np.concatenate([right_team_relative, right_team_distance], -1)
        right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
        right_team_speed_vec = np.concatenate([obs_right_team_direction, right_team_speed], axis=-1)
        right_team_state = np.concatenate(
            (
                right_team_dist_vec,
                right_team_speed_vec,
                right_team_tired,
            ),
            axis=1,
        ).astype(np.float32)
        assert right_team_state.shape == (10 if side == 1 else 11, 7), right_team_state.shape
        
        game_mode = np.zeros(7, np.float32)
        game_mode[obs['game_mode']] = 1

        score = self.get_score(obs)
        obs = np.concatenate([
            player_state, 
            ball_state, 
            left_team_state.reshape(-1), 
            right_team_state.reshape(-1), 
            game_mode, 
            score
        ])
        assert obs.shape == self.OBS_SHAPE, obs.shape
        assert obs.dtype == np.float32, obs.dtype

        return obs

    def get_hidden_state(self, obs):
        # player info
        hidden_state = []

        hidden_state.extend(do_flatten(obs['left_team']))
        hidden_state.extend(do_flatten(obs['right_team']))
        hidden_state.extend(do_flatten(obs['left_team_direction']))
        hidden_state.extend(do_flatten(obs['right_team_direction']))
        hidden_state.extend(do_flatten(obs['left_team_tired_factor']))
        hidden_state.extend(do_flatten(obs['right_team_tired_factor']))
        hidden_state.extend(do_flatten(obs['left_team_yellow_card']))
        hidden_state.extend(do_flatten(obs['right_team_yellow_card']))
        hidden_state.extend(do_flatten(obs['left_team_active']))
        hidden_state.extend(do_flatten(obs['right_team_active']))

        ball = obs['ball'][:2]
        ball_to_goal = [ball[0] + 1, ball[1]]
        hidden_state.extend(do_flatten(ball))
        hidden_state.extend(ball_to_goal)
        hidden_state.extend(do_flatten(obs['ball_direction']))
        hidden_state.append(obs['ball_owned_team'])
        hidden_state.append(obs['ball_owned_player'])

        score = self.get_score(obs)
        hidden_state.extend(do_flatten(score))
        hidden_state.append(obs['steps_left'])

        game_mode = np.zeros(7, np.float32)
        game_mode[obs['game_mode']] = 1
        hidden_state.extend(do_flatten(game_mode))

        return np.array(hidden_state, dtype=np.float32)

    def get_score(self, obs):
        score_diff = np.clip(obs['score'][0] - obs['score'][1], -2, 2) + 2
        score = np.zeros(5, dtype=np.float32)
        score[score_diff] = 1
        return score

    def get_avail_action(self, obs, ball_distance):
        avail = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (
            NO_OP,
            LEFT,
            TOP_LEFT,
            TOP,
            TOP_RIGHT,
            RIGHT,
            BOTTOM_RIGHT,
            BOTTOM,
            BOTTOM_LEFT,
            LONG_PASS,
            HIGH_PASS,
            SHORT_PASS,
            SHOT,
            SPRINT,
            RELEASE_MOVE,
            RELEASE_SPRINT,
            SLIDE,
            DRIBBLE,
            RELEASE_DRIBBLE,
        ) = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

        if obs["ball_owned_team"] == 1:  # opponents owning ball
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)
            if ball_distance > 0.03:
                avail[SLIDE] = 0
        elif (
            obs["ball_owned_team"] == -1
            and ball_distance > 0.03
            and obs["game_mode"] == 0
        ):  # Ground ball  and far from me
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
                avail[SLIDE],
            ) = (0, 0, 0, 0, 0, 0)
        else:  # my team owning ball
            avail[SLIDE] = 0
            if ball_distance > 0.03:
                (
                    avail[LONG_PASS],
                    avail[HIGH_PASS],
                    avail[SHORT_PASS],
                    avail[SHOT],
                    avail[DRIBBLE],
                ) = (0, 0, 0, 0, 0)

        # Dealing with sticky actions
        sticky_actions = obs["sticky_actions"]
        if sticky_actions[8] == 0:  # sprinting
            avail[RELEASE_SPRINT] = 0

        if sticky_actions[9] == 1:  # dribbling
            avail[SLIDE] = 0
        else:
            avail[RELEASE_DRIBBLE] = 0

        if np.sum(sticky_actions[:8]) == 0:
            avail[RELEASE_MOVE] = 0

        # if too far, no shot
        ball_x, ball_y, _ = obs["ball"]
        if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
            avail[SHOT] = 0
        elif (0.64 <= ball_x and ball_x <= 1.0) and (
            -0.27 <= ball_y and ball_y <= 0.27
        ):
            avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

        if obs["game_mode"] == 2 and ball_x < -0.7:  # Our GoalKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 4 and ball_x > 0.9:  # Our CornerKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
            return np.array(avail)

        elif obs["game_mode"] == 6 and ball_x > 0.6:  # Our PenaltyKick
            avail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            avail[SHOT] = 1
            return np.array(avail)

        return np.array(avail)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            ball_zone = [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_zone = [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_zone = [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            ball_zone = [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            ball_zone = [0, 0, 0, 0, 1.0, 0]
        else:
            ball_zone = [0, 0, 0, 0, 0, 1.0]
        return np.array(ball_zone, dtype=np.float32)

    def _encode_role_onehot(self, i):
        role = np.zeros(10, np.float32)
        role[i] = 1.0
        return role
