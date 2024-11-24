import random
import numpy as np
from numpy.random import rand

from social_dilemmas.envs.agent import CleanupAgent
from social_dilemmas.envs.gym.discrete_with_dtype import DiscreteWithDType
from social_dilemmas.envs.map_env import MapEnv
from social_dilemmas.maps import CLEANUP_MAP

# add engery system cleaning takes energy, so eat apple to regain energy
# add personality (able to clean, moving speed)
# add env effects: rate of pollution

# Add custom actions to the agent
_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}  # length of firing beam, length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    b"S": np.array([99, 156, 194], dtype=np.uint8),  # Light grey-blue stream cell
    b"H": np.array([113, 75, 24], dtype=np.uint8),  # Brown waste cells
    b"R": np.array([99, 156, 194], dtype=np.uint8),  # Light grey-blue river cell
}

# spawns apples relative to the current number of other apples within a radius of 2.
# spawn prob is based on 0, 1, 2, 3 apples within range respectively
SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4 #max waste on river is 0.4 = 40%
thresholdRestoration = 0.0 #when waste is below this number, spanws a bunch of apples
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class NewCleanupEnv(MapEnv):
    def __init__(
        self,
        ascii_map=CLEANUP_MAP,
        num_agents=1,
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.defrost_check = 0
        self.weather_check = 0
        self.turn_counter = 0
        self.weather_counter = 0
        self.weather_points = []
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])

        self.color_map.update(CLEANUP_COLORS)

    @property
    def action_space(self):
        return DiscreteWithDType(9, dtype=np.uint8)

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.compute_probabilities()

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE" and agent.get_energy() > 0:
            #temp soultion until I find out where action is coming from
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        elif action == "CLEAN" and agent.get_energy() > 0:
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
        return updates

    def custom_map_update(self):
        self.turn_counter += 1
        """ "Update the probabilities and then spawn""" 
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())
        if self.turn_counter % 250 == 0:
            self.weather_check = 1
        if self.weather_check == 1:
            self.update_map(self.weather())

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "agent-" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id,
                spawn_point,
                rotation,
                map_with_agents,
                view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            np.random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

    def compute_probabilities(self):
        waste_density = 0
        # calculate waste density. If potential waste area = 0, then waste density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        # if waste density is higher then the max the river can hold waste,
        # do not spawn apples and do not spawn waste
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            #Otherwise, set the prob of waste spawning (global var = 0.5 currently)
            self.current_waste_spawn_prob = wasteSpawnProbability
            # if the waste density is less then the minimum needed to spawn apples
            # sapwn apples based on apple spawn chance (currently 0.125)
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            #otherwise, spawn the apples on a less rate depending on how dirt the river is
            else:
                spawn_prob = ( 1 - (waste_density - thresholdRestoration) / (thresholdDepletion - thresholdRestoration) ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0) + counts_dict.get(b"8", 0) #we shall count ice as waste for computing apple probability
        free_area = self.potential_waste_area - current_area
        return free_area
    
    def weather(self):
        spawn_point = [] # list of all points on map to change
        # if weather is activated
        if self.weather_check == 1:
            self.weather_counter += 1 # counter to see how long weather has been going on
            # randomly change the river into ice (8)
            for i in range(len(self.river_points)):
                row, col = self.river_points[i]
                if self.world_map[row, col] == b"R":
                    if (random.randint(0,15) == 1):
                        self.weather_points.append([row, col])
                        spawn_point.append((row, col, b"8"))
            # randomly change the stream into ice (8)
            for i in range(len(self.stream_points)):
                row, col = self.stream_points[i]
                if self.world_map[row, col] == b"S":
                    if (random.randint(0,20) == 1):
                        self.weather_points.append([row, col])
                        spawn_point.append((row, col, b"8"))
            # if 50 frames has past, slowly defrost the ice
            if self.weather_counter % 50 == 0:
                self.defrost_check = 1
            if self.defrost_check == 1:
                for i in range(len(self.weather_points)):
                    row, col = self.weather_points[i]
                    if (random.randint(0,15) == 1):
                        spawn_point.append((row, col, b"S"))
            # if 75 frames has past, defrost all the ice
            if self.weather_counter % 125 == 0:
                self.weather_counter = 0 # reset weather counter 
                self.weather_check = 0 # turn off weather
                self.defrost_check = 0 # turn off defrost
                for i in range(len(self.weather_points)):  
                    row, col = self.weather_points[i]
                    spawn_point.append((row, col, b"S"))

        return spawn_point
                

        
