import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class MPEVisualizer(object):
    def __init__(
        self,
        env,
        state_seq: list,
        reward_seq=None,
    ):
        self.env = env

        self.interval = 100
        self.state_seq = state_seq
        self.reward_seq = reward_seq
        
        
        self.comm_active = not np.all(self.env.silent)
        print('Comm active? ', self.comm_active)
        
        self.init_render()

    def animate(
        self,
        save_fname: Optional[str] = None,
        view: bool = True,
    ):
        """Anim for 2D fct - x (#steps, #pop, 2) & fitness (#steps, #pop)"""
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=len(self.state_seq),
            blit=False,
            interval=self.interval,
        )
        # Save the animation to a gif
        if save_fname is not None:
            ani.save(save_fname)

        if view:
            plt.show(block=True)

    def init_render(self):
        from matplotlib.patches import Circle, Rectangle
        state = self.state_seq[0]
        
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        
        if hasattr(self.env, 'map_size_horizontal') and hasattr(self.env, 'map_size_vertical'):
            x_lim = self.env.map_size_horizontal
            y_lim = self.env.map_size_vertical
        elif hasattr(self.env, 'map_size') and self.env.map_size is not None:
            x_lim = self.env.map_size
            y_lim = self.env.map_size
        else:
            x_lim = 2
            y_lim = 2
        self.ax.set_xlim([-x_lim, x_lim])
        self.ax.set_ylim([-y_lim, y_lim])
        
        self.entity_artists = []
        has_walls = hasattr(self.env, 'wall_indices') and hasattr(self.env, 'wall_widths') and hasattr(self.env, 'wall_heights')
        if has_walls and self.env.wall_indices.shape[0] > 0:
            wall_indices = np.array(self.env.wall_indices)
            wall_entity_indices = set(self.env.num_agents + wall_indices)
        else:
            wall_entity_indices = set()
        
        for i in range(self.env.num_entities):
            is_wall = i in wall_entity_indices
            
            if is_wall and has_walls:
                landmark_idx = i - self.env.num_agents
                wall_idx_in_walls = np.where(wall_indices == landmark_idx)[0]
                if len(wall_idx_in_walls) > 0:
                    wall_idx_in_walls = wall_idx_in_walls[0]
                    wall_width = float(self.env.wall_widths[wall_idx_in_walls])
                    wall_height = float(self.env.wall_heights[wall_idx_in_walls])
                    wall_pos = state.p_pos[i]
                    rect = Rectangle(
                        (float(wall_pos[0]) - wall_width/2, float(wall_pos[1]) - wall_height/2),
                        wall_width,
                        wall_height,
                        color=np.array(self.env.colour[i]) / 255
                    )
                    self.ax.add_patch(rect)
                    self.entity_artists.append(rect)
                else:
                    c = Circle(
                        state.p_pos[i], self.env.rad[i], color=np.array(self.env.colour[i]) / 255
                    )
                    self.ax.add_patch(c)
                    self.entity_artists.append(c)
            else:
                c = Circle(
                    state.p_pos[i], self.env.rad[i], color=np.array(self.env.colour[i]) / 255
                )
                self.ax.add_patch(c)
                self.entity_artists.append(c)
            
        self.step_counter = self.ax.text(-1.95, 1.95, f"Step: {state.step}", va="top")
        
        if self.comm_active:
            self.comm_idx = np.where(self.env.silent == 0)[0]
            print('comm idx', self.comm_idx)
            self.comm_artists = []
            i = 0
            for idx in self.comm_idx:
                
                letter = ALPHABET[np.argmax(state.c[idx])]
                a = self.ax.text(-1.95, -1.95 + i*0.17, f"{self.env.agents[idx]} sends {letter}")
                
                self.comm_artists.append(a)
                i += 1
            
    def update(self, frame):
        from matplotlib.patches import Rectangle
        state = self.state_seq[frame]
        
        has_walls = hasattr(self.env, 'wall_indices') and hasattr(self.env, 'wall_widths') and hasattr(self.env, 'wall_heights')
        if has_walls and self.env.wall_indices.shape[0] > 0:
            wall_indices = np.array(self.env.wall_indices)
            wall_entity_indices = set(self.env.num_agents + wall_indices)
        else:
            wall_entity_indices = set()
        
        for i, artist in enumerate(self.entity_artists):
            is_wall = i in wall_entity_indices
            
            if is_wall and has_walls:
                landmark_idx = i - self.env.num_agents
                wall_idx_in_walls = np.where(wall_indices == landmark_idx)[0]
                if len(wall_idx_in_walls) > 0:
                    wall_idx_in_walls = wall_idx_in_walls[0]
                    wall_width = float(self.env.wall_widths[wall_idx_in_walls])
                    wall_height = float(self.env.wall_heights[wall_idx_in_walls])
                    wall_pos = state.p_pos[i]
                    artist.set_xy((float(wall_pos[0]) - wall_width/2, float(wall_pos[1]) - wall_height/2))
                else:
                    artist.center = state.p_pos[i]
            else:
                artist.center = state.p_pos[i]
            
        self.step_counter.set_text(f"Step: {state.step}")
        
        if self.comm_active:
            for i, a in enumerate(self.comm_artists):
                idx = self.comm_idx[i]
                letter = ALPHABET[np.argmax(state.c[idx])]
                a.set_text(f"{self.env.agents[idx]} sends {letter}")
        