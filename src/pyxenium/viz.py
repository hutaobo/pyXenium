import matplotlib.pyplot as plt

def plot_spots(df, x='x', y='y', c=None):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(df[x], df[y], s=5, c=(df[c] if c else None))
    ax.set_aspect('equal')
    return fig, ax
