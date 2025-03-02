
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

class InteractiveJulia:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.control_x = 0
        self.control_y = 0

        self.width = 512
        self.height = 512

        # Initialize the Julia set image
        self.image = self.ax.imshow(
            np.zeros((self.height, self.width)).get(), 
            extent=(-1.5, 1.5, -1.5, 1.5), 
            cmap="hot", 
            origin="lower"
        )

        # Add a red dot to indicate the control point
        self.dot, = self.ax.plot([self.control_x], [self.control_y], 'ro')

        # Connect event handlers
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        self.dragging = False

    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return
        self.dragging = True

    def on_release(self, event):
        """Handle mouse release events."""
        self.dragging = False

    def on_move(self, event):
        """Handle mouse movement events."""
        if self.dragging and event.inaxes == self.ax:
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
        xmin, xmax = -1.5, 1.5
        ymin, ymax = -1.5, 1.5
        max_iter = 100

        # Generate the Julia set
        julia = generate_julia_set(c, xmin, xmax, ymin, ymax, self.width, self.height, max_iter)

        # Update the image data
        self.image.set_data(julia.get())
        self.image.set_clim(vmin=julia.min(), vmax=julia.max())  # Adjust color limits

if __name__ == "__main__":
    interactive_julia = InteractiveJulia()
    interactive_julia.update()
    plt.show()