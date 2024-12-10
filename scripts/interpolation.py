import os
import glob
import numpy as np

def interpolate_embeddings(time_series_embeddings, desired_length=80):
    """
    Interpolates time-series embeddings from (768, num_texts) to (768, desired_length) using linear interpolation.

    Parameters
    ----------
    time_series_embeddings : np.ndarray
        A NumPy array of shape (768, num_texts) where `num_texts` is the variable number of time steps.
    desired_length : int, optional
        The fixed number of time steps to interpolate to. Default is 80.

    Returns
    -------
    np.ndarray
        A new array of shape (768, desired_length) with the linearly interpolated embeddings.
    """
    # Current number of time steps
    num_texts = time_series_embeddings.shape[1]

    # If already the desired length, no interpolation needed
    if num_texts == desired_length:
        return time_series_embeddings.copy()
    
    # Create original and new time step indices
    old_steps = np.arange(num_texts)
    new_steps = np.linspace(0, num_texts - 1, desired_length)

    # Initialize the interpolated array
    interpolated = np.zeros((768, desired_length), dtype=time_series_embeddings.dtype)

    # Interpolate each of the 768 embedding dimensions across time
    for d in range(768):
        interpolated[d, :] = np.interp(new_steps, old_steps, time_series_embeddings[d, :])

    return interpolated

def normalize_all_embeddings(parent_dir="/home/hice1/mbibars3/scratch/vlm-debiasing/data/e-daic/text_2d_interp"):
    """
    Iterates through all .npy embedding files in the given directory, interpolates them to (768, 80), 
    and saves them back to the directory.
    """
    # Pattern to find all .npy files in the directory
    file_pattern = os.path.join(parent_dir, "*.npy")
    embedding_files = glob.glob(file_pattern)

    if not embedding_files:
        print(f"No .npy files found in {parent_dir}.")
        return

    for file_path in embedding_files:
        try:
            # Load the embeddings
            embeddings = np.load(file_path)
            
            # Check that embeddings are shaped (768, num_texts)
            if embeddings.shape[0] != 768:
                print(f"Warning: {file_path} does not have 768 as the first dimension. Skipping.")
                continue

            # Interpolate to (768, 80)
            interpolated_embeddings = interpolate_embeddings(embeddings, desired_length=80)

            # Save back to the same file path (overwriting original)
            np.save(file_path, interpolated_embeddings)
            print(f"Normalized and saved {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

