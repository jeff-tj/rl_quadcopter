import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 100
        self.action_high = 800
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        # Calculate distance from target
        # dist_from_target = np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())
        z_dist = np.absolute(self.sim.pose[2] - self.target_pos[2])**2
        non_z_dist = np.sqrt(np.square(self.sim.pose[:2] - self.target_pos[:2]).sum())
        # Want the quadcopter to hover straight up, measure angular deviation
        angular_dev = np.sqrt(np.square(self.sim.pose[3:]).sum())
        # Also want to penalise movement not in z-axis (DeepMind Locomotion)
        non_z_v = np.square(self.sim.v[:2]).sum()
        # Also penalise movement downwards
        z_v = np.absolute(np.minimum(self.sim.v[2],0))
        # Penalty term
        penalty = .000001*z_dist + .001*non_z_dist + .0005*angular_dev + .0005*non_z_v + 0.1*z_v
        reward = 1. - penalty
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state