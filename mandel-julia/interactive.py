
import cupy as np
import matplotlib.pyplot as plt

def generate_julia_set(c, xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate the Julia set using vectorized operations.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Create a grid of complex numbers

    # Initialize the output array
    julia = np.zeros(Z.shape, dtype=int)

    # Vectorized computation of the Julia set
    for _ in range(max_iter):
        mask = np.abs(Z) <= 2  # Only update points that haven't escaped
        Z[mask] = Z[mask] ** 2 + c
        julia += mask  # Increment iteration count for points that haven't escaped

    return julia

def mandelbrot(c, iterations):
    """
    Compute the Mandelbrot set for the given complex matrix.
    """
    z = np.zeros_like(c)
    stable = np.zeros(c.shape, dtype=int)
    for i in range(iterations):
        mask = np.abs(z) <= 2
        z[mask] = z[mask] ** 2 + c[mask]

        stable[mask] = i

    return stable

def complex_matrix(xmin, xmax, ymin, ymax, pixel_density):
    re = np.linspace(xmin, xmax, int((xmax-xmin)*pixel_density))
    im = np.linspace(ymin, ymax, int((ymax-ymin)*pixel_density))
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

class InteractiveJulia:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1,2)
        self.control_x = 0
        self.control_y = 0

        self.xmin, self.xmax = -1.5, 1.5
        self.ymin, self.ymax = -1.5, 1.5
        self.max_iter = 100

        # Resolution of the image
        self.width = 256*2
        self.height = 256*2

        # Initialize the Julia set image
        self.image = self.ax1.imshow(
            np.zeros((self.height, self.width)).get(), 
            extent=(self.xmin, self.xmax, self.ymin, self.ymax), 
            cmap="hot", 
            origin="lower"
        )

        self.complex_matrix = complex_matrix(-2, 0.5, -1.5, 1.5, 2**8)

        self.mandelbrot = self.ax2.imshow(
            mandelbrot(self.complex_matrix, 100).get(),
            extent=(-2, 0.5, -1.5, 1.5),
            cmap="hot",
            origin="lower"
        )

        # Add a red dot to indicate the control point inside mandelbrot
        self.dot, = self.ax2.plot([self.control_x], [self.control_y], 'go')

        # Connect event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.dragging = False

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax2:
            return
        self.dragging = True

    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging = False

    def on_move(self, event):
        """Handle mouse movement events."""
        if self.dragging and event.inaxes == self.ax2:
            self.control_x = event.xdata
            self.control_y = event.ydata

            # Update the red dot position
            self.dot.set_data([self.control_x], [self.control_y])

            # Update the Julia set
            self.update()
            self.fig.canvas.draw_idle()

    def update(self):
        """Update the Julia set based on the current control point."""
        c = self.control_x + self.control_y * 1j

        # Generate the Julia set
        julia = generate_julia_set(c, self.xmin, self.xmax, 
                                   self.ymin, self.ymax, 
                                   self.width, self.height, self.max_iter)

        # Update the image data
        self.image.set_data(np.asnumpy(julia))
        self.image.set_clim(vmin=julia.min(), vmax=julia.max())  # Adjust color limits

if __name__ == "__main__":
    interactive_julia = InteractiveJulia()
    interactive_julia.update()
    plt.show()