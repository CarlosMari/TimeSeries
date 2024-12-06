import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

parser = argparse.ArgumentParser(description="CLI inputs for distirbution of generated data")
parser.add_argument('--file',
                    type=str,
                    default='data/TRAIN_NEG.pkl',
                    help='Route to the dataset')

parser.add_argument('--name',
                    type=str,
                    default='TRAIN',
                    help='Name of the output file')

parser.add_argument('--show',
                    action='store_true',
                    help='Show the plot')
args = parser.parse_args()




def analyze_curves(data):
    """
    Analyze curves and extract key characteristics.
    
    Parameters:
    data: numpy array of shape (n_curves, 1, points_per_curve)
    
    Returns:
    dict: Dictionary containing analysis results
    """
    n_curves = data.shape[0]
    results = {
        'max_minus_start': np.zeros(n_curves),
        'max_minus_end': np.zeros(n_curves),
        'end_minus_start': np.zeros(n_curves)
    }
    
    for i in range(n_curves):
        curve = data[i, 0]  # Remove middle dimension
        start_point = curve[0]
        end_point = curve[-1]
        max_point = np.max(curve)
        
        results['max_minus_start'][i] = max_point - start_point
        results['max_minus_end'][i] = max_point - end_point
        results['end_minus_start'][i] = end_point - start_point
    
    return results

def plot_analysis(results, sample_curves=None):
    """
    Create visualization of the analysis results.
    
    Parameters:
    results: dict containing analysis results
    sample_curves: optional numpy array of original curves to show examples
    """
    # Set up the figure with a grid
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Main scatter plot
    ax1 = fig.add_subplot(gs[0, :2])
    H, xedges, yedges = np.histogram2d(results['max_minus_start'], 
                                  results['max_minus_end'], 
                                  bins=50)
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)
    scatter = ax1.scatter(results['max_minus_start'], 
                     results['max_minus_end'],
                     c=H[np.searchsorted(x_centers, results['max_minus_start'])-1,
                        np.searchsorted(y_centers, results['max_minus_end'])-1],
                     cmap='viridis',
                     alpha=0.6)
    ax1.set_xlabel('Máximo - Inicio')
    ax1.set_ylabel('Máximo - Final')

    plt.colorbar(scatter, ax=ax1, label='Point Density')
    # Histograms
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(results['max_minus_start'], bins=50)
    ax2.set_xlabel('Maximum - Starting Point')
    ax2.set_title('Distribution')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(results['max_minus_end'], bins=50)
    ax3.set_xlabel('Maximum - Ending Point')
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(results['end_minus_start'], bins=50)
    ax4.set_xlabel('Ending Point - Starting Point')
    
    # Statistics table
    stats_ax = fig.add_subplot(gs[0, 2])
    stats_ax.axis('off')
    
    # Calculate statistics
    stats_text = "Statistics:\n\n"
    for key in results.keys():
        values = results[key]
        stats_text += f"{key}:\n"
        stats_text += f"  Mean: {np.mean(values):.2f}\n"
        stats_text += f"  Std: {np.std(values):.2f}\n"
        stats_text += f"  Min: {np.min(values):.2f}\n"
        stats_text += f"  Max: {np.max(values):.2f}\n\n"
    
    stats_ax.text(0, 0.95, stats_text, va='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.title('Old data generation')
    return fig



if __name__ == "__main__":
    file = open(args.file,'rb')
    X = pickle.load(file)
    file.close()
    X = (X - X.min())/(X.max() - X.min())
    
    X = X.reshape(( -1, 1, X.shape[2]))
    print(X.shape) 
    results = analyze_curves(X)
        
    # Create visualization
    fig = plot_analysis(results)
    plt.savefig(f'./data/{args.name}.png')
    if args.show:
        plt.show()