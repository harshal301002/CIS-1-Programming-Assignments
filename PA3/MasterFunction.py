# Import necessary modules and functions
import numpy as np
from computedk import compute_dk
from simple import closest_point_simple
from sorted import closest_point_sorted  # Import the sorted ICP algorithm

def master_function():
    """
    Master function to control the other functions. It computes the tip coordinates,
    finds the closest point on the mesh using both simple and sorted ICP algorithms,
    computes differences, and prints the results.
    """
    # Get file locations
    bodyA = "PADATA/Problem3-BodyA.txt"
    bodyB = "PADATA/Problem3-BodyB.txt"
    meshFile = "PADATA/Problem3MeshFile.sur"
    sampleReadings = "PADATA/PA3-A-Debug-SampleReadingsTest.txt"
    output = "PADATA/PA3-B-Debug-Output.txt"
    output_results = "PADATA/PA4-J-Output.txt"

    # Compute dk
    dk = compute_dk(bodyA, bodyB, sampleReadings)

    # Ensure dk is a 2D array with shape (3, n_frames)
    if dk.ndim == 1:
        dk = dk.reshape(3, -1)

    # Find closest points using simple ICP algorithm
    d_simple, c_simple = closest_point_simple(meshFile, dk)
    diff_simple = d_simple  # Distance between sample points and closest points
    sk_simple = dk  # Sample points
    ck_simple = c_simple  # Closest points

    # Find closest points using sorted ICP algorithm
    d_sorted, c_sorted = closest_point_sorted(meshFile, dk)
    diff_sorted = d_sorted
    sk_sorted = dk
    ck_sorted = c_sorted

    # Print the results for simple ICP
    print("Results using Simple ICP Algorithm:")
    n_frames = sk_simple.shape[1]
    for i in range(n_frames):
        print('Frame {}:'.format(i+1))
        print('Sample Point (sk): {:.4f}, {:.4f}, {:.4f}'.format(*sk_simple[:, i]))
        print('Closest Point (ck): {:.4f}, {:.4f}, {:.4f}'.format(*ck_simple[:, i]))
        print('Difference (diff): {:.4f}'.format(diff_simple[i]))
        print('-----------------------------')

    # Print the results for sorted ICP
    print("\nResults using Sorted ICP Algorithm:")
    for i in range(n_frames):
        print('Frame {}:'.format(i+1))
        print('Sample Point (sk): {:.4f}, {:.4f}, {:.4f}'.format(*sk_sorted[:, i]))
        print('Closest Point (ck): {:.4f}, {:.4f}, {:.4f}'.format(*ck_sorted[:, i]))
        print('Difference (diff): {:.4f}'.format(diff_sorted[i]))
        print('-----------------------------')

if __name__ == "__main__":
    master_function()

