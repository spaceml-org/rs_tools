from math import floor, ceil

# TODO: Eman to refactor code
def bounds_and_points_to_step(xmin: float, xmax: float, Nx: float) -> float:
    """Calculates the dx from the minmax
    Eq:
        Lx = abs(xmax - xmin)
        dx = Lx / (Nx - 1)

    Args:
        xmin (Array | float): the input start point
        xmax (Array | float): the input end point
        Nx (int | float): the number of points

    Returns:
        dx (Array | float): the distance between each of the
            steps.
    """
    Lx = abs(float(xmax) - float(xmin))
    return float(Lx) / (float(Nx) - 1.0)


def bounds_and_step_to_points(xmin: float, xmax: float, dx: float) -> int:
    """Calculates the number of points from the
    endpoints and the stepsize

    Eq:
        Nx = 1 + floor((xmax-xmin)) / dx)

    Args:
        xmin (Array | float): the input start point
        xmax (Array | float): the input end point
        dx (Array | float): stepsize between each point.

    Returns:
        Nx (Array | int): the number of points
    """
    Lx = abs(float(xmax) - float(xmin))
    return int(floor(1.0 + float(Lx) / float(dx)))