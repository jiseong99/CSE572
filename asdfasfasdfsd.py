#!/usr/bin/env python3
# encoding: utf-8

__copyright__ = "Copyright 2019, AAIR Lab, ASU"
__authors__ = ["Naman Shah", "Rushang Karia"]
__credits__ = ["Siddharth Srivastava"]
__license__ = "MIT"
__version__ = "1.0"
__maintainers__ = ["Naman Shah"]
__contact__ = "aair.lab@asu.edu"
__docformat__ = 'reStructuredText'

import sys
import problem
import json
import os
import random
import utils
from tqdm.auto import trange
import time

from parser import parse_args
from server import initialize_planning_server
from server import generate_maze
from utils import initialize_ros
from utils import cleanup_ros


class QLearning:

    def __init__(self, objtypes, objcount, seed, file_path, alpha, gamma,
        episodes, max_steps, epsilon_task, env, clean):
        
        self.objtypes = objtypes
        self.objcount = objcount
        self.seed = seed
        self.epsilon_task = epsilon_task
        self.env = env
        self.obj_json_file = utils.ROOT_DIR + "/objects.json"
        self.obj = json.load(open(self.obj_json_file))
        self.helper = problem.Helper()
        self.helper.reset_world()
        
        assert not os.path.exists(file_path) or not os.path.isdir(file_path)
        
        self.file_path = file_path
        if clean:
            self.file_handle = open(file_path, "w")
            self.write_file_header(file_path)
        else:
            self.file_handle = open(file_path, "a")

        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps

        q_values = self.learn(episodes)

        with open(utils.ROOT_DIR + "/q_values.json", "w") as fout:
            json.dump(q_values, fout)
            
    def write_file_header(self, file_path):
       
        with open(file_path, "w") as f:
            f.write("Env;Object Types;Num of Objects;Seed;Gamma;Episode #;Alpha;Epsilon;Cumulative Reward;Total Steps;Goal Reached\n")

    def write_to_file(self, file_path, episode_num, alpha, epsilon,
        cumulative_reward, total_steps, is_goal_satisfied):

        with open(file_path, "a") as f:
            f.write("%s;%u;%u;%u;%.6f;%u;%.6f;%.6f;%.2f;%u;%s\n" % (
                self.env,
                self.objtypes,
                self.objcount,
                self.seed,
                self.gamma,
                episode_num,
                alpha,
                epsilon,
                cumulative_reward,
                total_steps,
                is_goal_satisfied))

    def get_q_value(self,alpha, gamma, reward, q_s_a, q_s_dash_a_dash):
        '''
        Use the Q-Learning update rule to calculate and return the q-value.

        return type: float
        '''

        '''
        YOUR CODE HERE
        '''
        q = (1 - alpha) * q_s_a + alpha * (reward + gamma * q_s_dash_a_dash)

        return q
    
    def compute_cumulative_reward(self, current_cumulative_reward, gamma, step, reward):
        '''
        Calculate the running cumulative reward at every step using 
        current value of the cumulative reward,
        discount factor (gamma), 
        current step number (step), 
        the rewards for the current state (reward)

        return type: float
        '''


        '''
        YOUR CODE HERE
        '''
        cumulative_reward = current_cumulative_reward + (gamma ** step) * reward

        return cumulative_reward

    def get_epsilon(self, current_epsilon, episode):
        '''
        Calculate the value for decaying epsilon/
        
        Input: 
        current_epsilon: current value for the epsilon.
        episode: episode number

        Output:

        new value for the epsilon

        return type: float 
        '''

        '''
        YOUR CODE HERE

        '''
        new_epsilon = current_epsilon * 0.99
        if new_epsilon < 0.01:
            new_epsilon = 0.01
        return new_epsilon


    def alpha(self, current_alpha, episode, step):
        return current_alpha

    def learn(self, episodes):
        q_values = {} # Use this dictionary to keep track of q values

        root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        actions_config_file = open(root_path + "/action_config.json",'r')
        actions_config = json.load(actions_config_file)

        objects_file = open(root_path + "/objects.json",'r')
        objects = json.load(objects_file)

        pick_loc=[]
        place_loc=[]
        for object in objects['object'].keys():
            for e in objects['object'][object]['load_loc']:
                pick_loc.append(e)
        
        for goal in objects['goal'].keys():
            for e in objects['goal'][goal]['load_loc']:
                place_loc.append(e)


        def get_params_change(act):
            raw = actions_config.get(act, {}).get("params", None)
            if isinstance(raw, dict):
                return raw.copy()
            if isinstance(raw, list):
                return {raw[0]: None} if raw else {}
            return {}

        
        epsilon = 1.0
        for i in trange(0, episodes, desc="Episode", unit="episode"):

            epsilon = self.get_epsilon(epsilon, i) # Complete get_epsilon()
            curr_state = self.helper.get_current_state()
            cumulative_reward = 0
            
            for step in range(self.max_steps):

                if self.helper.is_terminal_state(curr_state):
                    break
                    
                actions_list = self.helper.get_all_actions()
                curr_loc = [curr_state['robot']['x'],curr_state['robot']['y']]    
                possible_actions_list = actions_list
                                    
                '''
                YOUR CODE HERE
                '''
                key = json.dumps(curr_state, sort_keys=True)

                if random.random() < epsilon:
                    action = random.choice(possible_actions_list)

                else:
                    best_q = float("-inf")
                    best_a = None

                    for a in possible_actions_list:
                        q = q_values.get((key, a), 0.0)

                        if q > best_q:
                            best_q = q
                            best_a = a
                    action = best_a if best_a is not None else random.choice(possible_actions_list)


                original_action = action

                params = get_params_change(action) 


                if action not in actions_config and " " in original_action:
                    head, tail = original_action.split(" ", 1)

                    if head in actions_config:
                        action = head
                        params = get_params_change(head)

                        if isinstance(params, dict) and params:
                            first_key = next(iter(params.keys()))
                            params[first_key] = tail


                if action in actions_config and isinstance(params, dict) and params:

                    if " " in original_action:
                        tail = original_action.split(" ", 1)[1]

                        for k, v in list(params.items()):

                            if v in (None, ""):
                                params[k] = tail

                try:
                    result = self.helper.execute_action(action, params)

                    if isinstance(result, tuple) and len(result) == 2:
                        success, next_state = result

                    else:
                        success, next_state = False, curr_state

                except Exception:
                    success, next_state = False, curr_state

                reward = self.helper.get_reward(curr_state, action, next_state)


                next_key = json.dumps(next_state, sort_keys=True)
                next_actions = self.helper.get_all_actions()

                if next_actions:
                    max_next = max(q_values.get((next_key, a), 0.0) for a in next_actions)
                    
                else:
                    max_next = 0.0


                curr_q = q_values.get((key, action), 0.0)
                new_q = self.get_q_value(self.alpha, self.gamma, reward, curr_q, max_next)
                q_values[(key, action)] = new_q

                cumulative_reward = self.compute_cumulative_reward(cumulative_reward, self.gamma, step, reward)
                curr_state = next_state

                if self.helper.is_terminal_state(curr_state):
                    step = step + 1
                    break
                    
                    



                step = step + 1

            self.write_to_file(self.file_path, i, self.alpha, epsilon,
                cumulative_reward, step, 
                self.helper.is_terminal_state(curr_state))
            self.helper.reset_world()

        return q_values

def run_qlearning(objtypes, objcount, seed, file_name, alpha, 
                  gamma, episodes, max_steps, epsilon_task, env, clean):
    
    file_path = utils.ROOT_DIR + "/" + file_name
    
    rosprocess = initialize_ros()
    planserver_process = initialize_planning_server()
    
    # Generate the world.
    generate_maze(objtypes, objcount, seed, 1, env)
   

    QLearning(objtypes, objcount, seed, file_path, alpha, 
        gamma, episodes, max_steps, epsilon_task, env, clean)
    
    cleanup_ros(planserver_process.pid, rosprocess.pid)
    time.sleep(2)


def submit(args):
    task_name = "task2"
    fname = "qlearning.csv"
    for i, env in enumerate(['cafeWorld','bookWorld']):
        print("Submission: running {} for {}".format(task_name, env))
        run_qlearning(objtypes=1, objcount=1, seed=100,
                      file_name=fname, alpha=0.3, gamma=0.9,
                      episodes=500, max_steps=500, epsilon_task=2,
                      env=env, clean=not(i))


if __name__ == "__main__":

    random.seed(111)

    args = parse_args()

    if args.submit:
        submit(args)
    else:
        task_name = "task2"
        fname = "qlearning.csv"
        run_qlearning(objtypes=args.objtypes, objcount=args.objcount, seed=args.seed,
                      file_name=args.file_name, alpha=args.alpha, gamma=args.gamma,
                      episodes=args.episodes, max_steps=args.max_steps, epsilon_task=2,
                      env=args.env, clean=args.clean)
