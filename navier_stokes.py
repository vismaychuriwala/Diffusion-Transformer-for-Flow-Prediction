import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from torch.utils.data import Dataset

def forcing_same_as_paper(X, Y, amplitude=0.000005):
    return amplitude * (np.sin(2*np.pi*(X+Y)) + np.cos(2*np.pi*(X+Y)))

class NavierStokesSimulator:
    def __init__(self, nx=64, ny=64, Lx=1.0, Ly=1.0, dt=0.001, nt=1000, nu=1e-5, forcing=None):
        # Grid and parameters
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.dt = dt
        self.nt = nt
        self.nu = nu
        
        # Prepare wavenumbers
        self.kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dy)
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.k_squared = self.kx**2 + self.ky**2
        self.k_squared[0,0] = 1e-14

        # Create physical grid
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Initialize vorticity field w
        self.w = np.zeros((nx, ny), dtype=np.float64)

        self.forcing = forcing
        
    def set_initial_condition(self, w_init_func):
        self.w = w_init_func(self.X, self.Y)
    
    def set_random_initial_condition(self):
        random_phase = np.random.randn(self.nx, self.ny) + 1j*np.random.randn(self.nx, self.ny)

        spectrum = (73.0/2.0) * (self.k_squared + 49)**(-2.5)
        
        w_hat = random_phase * np.sqrt(spectrum)
        w_init = np.fft.ifft2(w_hat).real
        
        self.w = w_init

    
    def compute_velocity_from_vorticity(self, w):
        w_hat = np.fft.fft2(w)
        psi_hat = -w_hat / self.k_squared

        u_hat = (1j * self.ky) * psi_hat
        v_hat = (-1j * self.kx) * psi_hat

        u = np.fft.ifft2(u_hat).real
        v = np.fft.ifft2(v_hat).real
        return u, v

    def rhs(self, w):
        u, v = self.compute_velocity_from_vorticity(w)
        w_hat = np.fft.fft2(w)

        dw_dx_hat = (1j * self.kx) * w_hat
        dw_dy_hat = (1j * self.ky) * w_hat
        dw_dx = np.fft.ifft2(dw_dx_hat).real
        dw_dy = np.fft.ifft2(dw_dy_hat).real

        nonlinear = u * dw_dx + v * dw_dy
        lap_w_hat = -self.k_squared * w_hat
        lap_w = np.fft.ifft2(lap_w_hat).real

        out = -nonlinear + self.nu * lap_w
        if self.forcing:
            force = self.forcing(self.X, self.Y)
            out += force

        return out

    def runge_kutta_step(self, w):
        dt = self.dt
        k1 = self.rhs(w)
        k2 = self.rhs(w + 0.5*dt*k1)
        k3 = self.rhs(w + 0.5*dt*k2)
        k4 = self.rhs(w + dt*k3)

        w_new = w + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        return w_new
    
    def crank_nicolson_step(self, w):
        dt = self.dt
        nu = self.nu

        w_hat = np.fft.fft2(w)

        u, v = self.compute_velocity_from_vorticity(w)

        w_hat_current = w_hat
        dw_dx_hat = (1j * self.kx) * w_hat_current
        dw_dy_hat = (1j * self.ky) * w_hat_current

        dw_dx = np.fft.ifft2(dw_dx_hat).real
        dw_dy = np.fft.ifft2(dw_dy_hat).real

        nonlinear = u * dw_dx + v * dw_dy
        N_hat = np.fft.fft2(nonlinear)

        force = self.forcing(self.X, self.Y) if self.forcing else 0.0
        f_hat = np.fft.fft2(force)
        lap_w_hat = -self.k_squared * w_hat_current

        numerator = w_hat_current + dt * (-N_hat + f_hat + (nu/2.0)*lap_w_hat)
        denominator = 1.0 + (dt * nu * self.k_squared / 2.0)
        w_hat_new = numerator / denominator
        w_new = np.fft.ifft2(w_hat_new).real

        return w_new


    def run(self, store_data=True):
        w_data = None
        u_data = None
        v_data = None
        if store_data:
            w_data = np.zeros((self.nt+1, self.nx, self.ny), dtype=np.float64)
            u_data = np.zeros((self.nt+1, self.nx, self.ny), dtype=np.float64)
            v_data = np.zeros((self.nt+1, self.nx, self.ny), dtype=np.float64)

            # Store initial data
            w_data[0] = self.w.copy()
            u_init, v_init = self.compute_velocity_from_vorticity(self.w)
            u_data[0] = u_init
            v_data[0] = v_init
        else:
            w_data = u_data = v_data = None
        
        for n in range(1, self.nt+1):
            #self.w = self.runge_kutta_step(self.w)
            self.w = self.crank_nicolson_step(self.w)
            if store_data:
                w_data[n] = self.w.copy()
                u, v = self.compute_velocity_from_vorticity(self.w)
                u_data[n] = u
                v_data[n] = v

        return w_data, u_data, v_data

    def visualize_simulation(self, w_data, u_data=None, v_data=None, save=None, interval=50):
        fig, ax = plt.subplots()
        im = ax.imshow(w_data[0].T, origin='lower', extent=[0, self.Lx, 0, self.Ly], cmap='jet')
        ax.set_title("Vorticity Field Evolution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        quiv = None

        if u_data is not None and v_data is not None:
            skip = (slice(None, None, 4), slice(None, None, 4))
            quiv = ax.quiver(self.X[skip].T, self.Y[skip].T, u_data[0][skip].T, v_data[0][skip].T, color='white')

        def update(frame):
            im.set_data(w_data[frame].T)
            ax.set_title(f"Vorticity at step {frame}")
            if quiv is not None:
                quiv.set_UVC(u_data[frame][skip].T, v_data[frame][skip].T)
            return [im, quiv] if u_data is not None else [im]

        ani = animation.FuncAnimation(fig, update, frames=len(w_data), interval=interval, blit=True)
        plt.show()

        if save is not None:
            ani.save(save, writer="ffmpeg")
    
class NavierStokesDataset(Dataset):
    def __init__(self, file_path, input_steps=10, pred_steps=1):
        self.data_list = [np.load(path) for path in file_path]

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