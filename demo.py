from social_dilemmas.envs import env_creator, agent
import pygame

import numpy as np

_MAP_ENV_ACTIONS = {
    "MOVE_LEFT": [0, -1],  # Move left
    "MOVE_RIGHT": [0, 1],  # Move right
    "MOVE_UP": [-1, 0],  # Move up
    "MOVE_DOWN": [1, 0],  # Move down
    "STAY": [0, 0],  # don't move
    "TURN_CLOCKWISE": [[0, 1], [-1, 0]],  # Clockwise rotation matrix
    "TURN_COUNTERCLOCKWISE": [[0, -1], [1, 0]],
}  # Counter clockwise rotation matrix
# Positive Theta is in the counterclockwise direction

REVERSE_ACTIONS = {
    (0, -1): "MOVE_LEFT",
    (0, 1): "MOVE_RIGHT",
    (-1, 0): "MOVE_UP",
    (1, 0): "MOVE_DOWN",
    (0, 0): "STAY"
}

BASE_ACTIONS_REVERSE = {
    0: "MOVE_LEFT",
    1: "MOVE_RIGHT",
    2: "MOVE_UP",
    3: "MOVE_DOWN",
    4: "STAY",
    5: "TURN_CLOCKWISE",
    6: "TURN_COUNTERCLOCKWISE",
    7: "FIRE",
    8: "CLEAN",
}
BASE_ACTIONS_REVERSE = {v: k for k, v in BASE_ACTIONS_REVERSE.items()}

ORIENTATION_MATRICES = {
    "UP": np.array([[1, 0], [0, 1]]),
    "LEFT": np.array([[0, 1], [-1, 0]]),  
    "DOWN": np.array([[-1, 0], [0, -1]]), 
    "RIGHT": np.array([[0, -1], [1, 0]]),
}

def transform_action(action, orientation):
    action_vector = np.array(_MAP_ENV_ACTIONS[action])
    orientation = ORIENTATION_MATRICES[orientation]
    movement =  orientation @ action_vector
    return REVERSE_ACTIONS[tuple(movement)]

class Strategy:

    def __init__(self, agent: agent.HarvestAgent):
        self.agent = agent

    def action(self, observation):
        raise NotImplementedError
    
class SocailWarfare(Strategy):

    def __init__(self, agent: agent.HarvestAgent):
        super().__init__(agent)
        self.objective = b"H"

    def get_view_slice(self):
        row, col = self.agent.pos
        view_slice = self.agent.full_map[
            row : row + self.agent.view_len * 2 + 1,
            col : col + self.agent.view_len * 2 + 1,
        ]
        return view_slice

    def action(self, observation):
        orientation = self.agent.get_orientation()
        
        full_map: np.array  = self.agent.full_map
        player_pos = self.agent.get_pos()

        objective_pos = np.where(full_map == self.objective)
        if len(objective_pos[0]) == 0:
            return BASE_ACTIONS_REVERSE["STAY"]
        
        objective_pos = np.array(objective_pos).T
        # find the closest objective
        distances = np.linalg.norm(objective_pos - player_pos, axis=1)

        closest_objective_index = np.argmin(distances)
        closest_objective = objective_pos[closest_objective_index]

        if distances[closest_objective_index] < 2:
            if closest_objective[0] > player_pos[0]:
                turn_orientation = "DOWN"
            elif closest_objective[0] < player_pos[0]:
                turn_orientation = "UP"
            elif closest_objective[1] > player_pos[1]:
                turn_orientation = "RIGHT"
            elif closest_objective[1] < player_pos[1]:
                turn_orientation = "LEFT"

            if orientation != turn_orientation:
                return BASE_ACTIONS_REVERSE["TURN_CLOCKWISE"]
            else:
                return BASE_ACTIONS_REVERSE["CLEAN"]

        # select the action that moves the player towards the objective
        if closest_objective[0] > player_pos[0]:
            action = "MOVE_DOWN"
        elif closest_objective[0] < player_pos[0]:
            action = "MOVE_UP"
        elif closest_objective[1] > player_pos[1]:
            action = "MOVE_RIGHT"
        elif closest_objective[1] < player_pos[1]:
            action = "MOVE_LEFT"
        else:
            action = "STAY"

        action = transform_action(action, orientation)
        return BASE_ACTIONS_REVERSE[action]

import time
GRID_SIZE = 30

def draw(rgb_array, screen, height, width):
    for y in range(height):
        for x in range(width):
            color = rgb_array[y, x]  # Get the color for this pixel
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(screen, color, rect)

def main():
    pygame.init()

    env = env_creator.get_env_creator("newcleanup", num_agents=5)(1)
    obs = env.reset()
    strategies = {agent_id: SocailWarfare(agent) for agent_id, agent in env.agents.items()}
    rgb_arr = env.render(mode="array")
    height, width, _ = rgb_arr.shape
    screen = pygame.display.set_mode((width * GRID_SIZE, height * GRID_SIZE))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        actions = {}
        for agent_id, agent in env.agents.items():
            action = strategies[agent_id].action(obs[agent_id])
            actions[agent_id] = action

        screen.fill((0, 0, 0))
        draw(rgb_arr, screen, height, width)
        pygame.display.update()

        env.step(actions)
        rgb_arr = env.render(mode="array")
        time.sleep(0.2)

    pygame.quit()

if __name__ == "__main__":
    main()