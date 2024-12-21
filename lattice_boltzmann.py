import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch

from torch.utils.data import Dataset

class BoltzmannSimulation:
    def __init__(self, nx=128, ny=64, timesteps=500, U=0.1):
        self.nx = nx
        self.ny = ny
        self.timesteps = timesteps
        self.U = U
        
        # D2Q9 model parameters
        self.cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

        self.tau = 0.9
        self.omega = 1.0 / self.tau

        self.rho0 = 1.0
        y_grid = np.arange(self.ny)

        ux_init = (2*self.U/(self.ny-1))*y_grid - self.U

        perturb_amp = 0.05*self.U
        ux_2D = np.tile(ux_init, (self.nx,1)).T + perturb_amp*(2*np.random.rand(self.ny,self.nx)-1)
        uy_2D = perturb_amp*(2*np.random.rand(self.ny,self.nx)-1)

        self.f = self.init_distribution(self.rho0, ux_2D.T, uy_2D.T)

        self.rho_history = np.zeros((self.timesteps, self.ny, self.nx))
        
    def init_distribution(self, rho, ux, uy):
        f_init = np.zeros((self.nx, self.ny, 9))
        usq = ux**2 + uy**2
        for i in range(9):
            cu = self.cxs[i]*ux + self.cys[i]*uy
            f_init[:,:,i] = self.weights[i]*rho*(1.0 + 3.0*cu + 4.5*(cu**2) - 1.5*usq)
        return f_init

    def equilibrium(self, rho, ux, uy):
        feq = np.zeros((self.nx, self.ny, 9))
        usq = ux**2 + uy**2
        for i in range(9):
            cu = self.cxs[i]*ux + self.cys[i]*uy
            feq[:,:,i] = self.weights[i]*rho*(1 + 3*cu + 4.5*(cu**2) - 1.5*usq)
        return feq

    def stream(self):
        f_streamed = np.zeros_like(self.f)
        for i in range(9):
            cx, cy = self.cxs[i], self.cys[i]
            f_streamed[:,:,i] = np.roll(np.roll(self.f[:,:,i], cx, axis=0), cy, axis=1)
        self.f = f_streamed

    def run(self):
        for t in range(self.timesteps):
            rho = np.sum(self.f, axis=2)
            ux = np.sum(self.f * self.cxs[None,None,:], axis=2) / rho
            uy = np.sum(self.f * self.cys[None,None,:], axis=2) / rho

            ux[:, -1] = self.U
            uy[:, -1] = 0.0

            ux[:, 0] = -self.U
            uy[:, 0] = 0.0

            feq_top = self.equilibrium(rho, ux, uy)
            self.f[:, -1, [2,5,6]] = feq_top[:, -1, [2,5,6]] + (self.f[:, -1, [4,7,8]] - feq_top[:, -1, [4,7,8]])

            feq_bottom = self.equilibrium(rho, ux, uy)
            self.f[:, 0, [4,7,8]] = feq_bottom[:, 0, [4,7,8]] + (self.f[:, 0, [2,5,6]] - feq_bottom[:, 0, [2,5,6]])

            self.f[0,:,:] = self.f[-2,:,:]
            self.f[-1,:,:] = self.f[1,:,:]

            feq = self.equilibrium(rho, ux, uy)
            self.f = (1 - self.omega)*self.f + self.omega*feq

            self.stream()

            self.rho_history[t] = rho.T

    def save_data(self, filepath):
        out_dir = os.path.dirname(filepath)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.savez(filepath, density=self.rho_history)
        #print(f"Simulation data saved to {filepath}")

    def visualize(self, interval=50, save_gif=False, gif_name="lb_simulation.gif"):
        fig, ax = plt.subplots()
        ims = []
        vmin = self.rho_history.min()
        vmax = self.rho_history.max()
        im = ax.imshow(self.rho_history[5], cmap='turbo', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title("Density Field at t=5")

        def update(frame):
            im.set_data(self.rho_history[frame])
            ax.set_title(f"Density Field at t={frame}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=self.timesteps, interval=interval, blit=True)

        if save_gif:
            ani.save(gif_name, writer='pillow')
            print(f"Animation saved as {gif_name}")

        plt.show()


class LatticeBoltzmannDataset(Dataset):
    def __init__(self, file_path, input_steps=10, pred_steps=1):
        self.data_list = [np.load(path)['density'] for path in file_path]

        self.input_steps = input_steps
        self.pred_steps = pred_steps

        all_data = np.concatenate(self.data_list, axis=0)
        self.mean = np.mean(all_data)
        self.std = np.std(all_data)

        self.cumulative_length = [0]
        for data in self.data_list:
            self.cumulative_length.append(self.cumulative_length[-1] + len(data) - input_steps - pred_steps)

    def __len__(self):
        return self.cumulative_length[-1]

    def __getitem__(self, idx):
        file_idx = next(i for i, length in enumerate(self.cumulative_length) if idx < length) - 1
        local_idx = idx - self.cumulative_length[file_idx]

        data = self.data_list[file_idx]

        x = data[local_idx:local_idx + self.input_steps]
        y = data[local_idx + self.input_steps:local_idx + self.input_steps + self.pred_steps]
        
        x = (x - self.mean) / self.std  # Normalize input
        y = (y - self.mean) / self.std  # Normalize output
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
