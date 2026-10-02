from scipy.linalg import eigh
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import math
import random

def load_and_center_dataset(filename):
    """
    Load dataset from .npy file and center it by subtracting the mean.

    Args:
        filename (str): Path to the .npy file

    Returns:
        numpy.ndarray: Centered dataset (n x d matrix)
    """
    # Your implementation goes here!

    dataset = np.array(np.load(filename))
    centered_data: npt.NDArray[np.float64] = dataset - np.mean(dataset)
    
    return centered_data

def get_covariance(dataset):
    """
    Calculate the sample covariance matrix of the dataset.

    Args:
        dataset (numpy.ndarray): Centered dataset (n x d matrix)

    Returns:
        numpy.ndarray: Covariance matrix (d x d matrix)
    """

    return np.cov(dataset)

def get_eig(S, k):
    """
    Get the k largest eigenvalues and corresponding eigenvectors.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        k (int): Number of largest eigenvalues/eigenvectors to return

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    # Your implementation goes here!
    eigenvalues, eigenvectors = np.linalg.eigh(S)

    diagonal_eigenvalues = np.diag(eigenvalues)
    return (diagonal_eigenvalues, eigenvectors)

def get_eig_prop(S, prop):
    """
    Get eigenvalues and eigenvectors that explain more than prop proportion of variance.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        prop (float): Minimum proportion of variance to explain (0 <= prop <= 1)

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    # Your implementation goes here!
    raise NotImplementedError

def project_and_reconstruct_image(image, U):
    """
    Project image to PCA subspace and reconstruct it back to original dimension.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)

    Returns:
        numpy.ndarray: Reconstructed image as flattened d x 1 vector
    """
    # Your implementation goes here!
    raise NotImplementedError

def project_reconstruct_with_gaussian_noise(image, U, sigma=0.1, seed=0):
    """
    Project image to PCA subspace, add Gaussian noise to coefficients, and reconstruct.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)
        sigma (float): Standard deviation of Gaussian noise to add to coefficients
        seed (int): Random seed for reproducibility
    Returns:        
        numpy.ndarray: Reconstructed image with noise as flattened d x 1 vector
    """
    # Your implementation goes here!
    raise NotImplementedError

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    """
    Display three images side by side: original high-res, original, and reconstructed.

    Args:
        im_orig_fullres (numpy.ndarray): Original high-resolution image
        im_orig (numpy.ndarray): Original low-resolution image
        im_reconstructed (numpy.ndarray): Reconstructed image from PCA

    Returns:
        tuple: (fig, ax1, ax2, ax3) matplotlib figure and axes objects
    """

    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    # Your implementation goes here!
    # Note: Do NOT include plt.show() in your implementation - it will be called separately for testing

    return fig, ax1, ax2, ax3


# =============================================================================
# N-GRAM LANGUAGE MODEL FUNCTIONS
# =============================================================================

class NGramCharLM:
    """
    N-gram Character Language Model
    
    This class implements an n-gram character-level language model.
    Students should implement the missing methods.
    """
    
    def __init__(self, n=5):
        """
        Initialize the n-gram language model.
        
        Args:
            n (int): The order of the n-gram model (must be >= 1)
        """
        assert n >= 1, "n must be at least 1"
        self.n = n
        self.counts = {}  # dict[str, dict[str, int]] - context -> {char: count}
        self.vocab = set()  # set of all characters seen during training
        self.trained = False
    
    def fit(self, text: str):
        """
        Train the n-gram model on the given text.
        
        Args:
            text (str): Training text
            
        Returns:
            NGramCharLM: self (for method chaining)


        Examples:
        If n=3 and text="banana", then counts should look like:
        {
            ""   : {"b": 1},         # at i=0, ctx="", ch="b"
            "b"  : {"a": 1},         # at i=1, ctx="b", ch="a"
            "ba" : {"n": 1},         # at i=2, ctx="ba", ch="n"
            "an" : {"a": 2},         # at i=3,5, ctx="an", ch="a"
            "na" : {"n": 1}          # at i=4, ctx="na", ch="n"
        }

        If n=2 and text="abab", then counts should look like:
        {
            ""  : {"a": 1},          # at i=0, ctx="", ch="a"
            "a" : {"b": 2},          # at i=1,3, ctx="a", ch="b"
            "b" : {"a": 1}           # at i=2, ctx="b", ch="a"
        }

        Note: Context is always the last (n-1) characters before current position.
          For positions near the beginning, context may be shorter than (n-1).
        """
        # Your implementation goes here!
        raise NotImplementedError
    
    def _probs_for_context(self, context: str):
        """
        Get probability distribution over next characters for a given context.
        
        Args:
            context (str): The context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities
        """
        # Your implementation goes here!
        raise NotImplementedError
    
    def prob(self, s: str) -> float:
        """
        Calculate the probability of a string.
        
        Args:
            s (str): String to calculate probability for
            
        Returns:
            float: Probability of the string
        """
        return math.exp(self.logprob(s))
    
    def logprob(self, s: str) -> float:
        """
        Calculate the log-probability of a string.
        
        Args:
            s (str): String to calculate log-probability for
            
        Returns:
            float: Log-probability of the string
        """
        lp = 0.0
        for i in range(len(s)):
            # Get context for character at position i
            ctx = s[max(0, i - (self.n - 1)):i]
            
            # Get probability of character given context
            p_next = self._probs_for_context(ctx).get(s[i], 0.0)
            
            if p_next <= 0.0:
                return float("-inf")
            
            lp += math.log(p_next)
        
        return lp
    
    def next_char_distribution(self, context: str):
        """
        Get the probability distribution over next characters for given context.
        
        Args:
            context (str): Context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities,
                            sorted by probability in descending order
        """
        d = self._probs_for_context(context)
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    def generate(self, num_chars: int, seed: str = "") -> str:
        """
        Generate text using the trained model.

        Args:
            num_chars (int): Number of characters to generate
            seed (str): Initial string to start generation

        Returns:
            str: Generated text (seed + num_chars new characters)
        """
        out = list(seed)

        for _ in range(num_chars):
            # Get probability distribution for current context
            dist = self._probs_for_context("".join(out))

            if not dist:
                break

            # Extract characters and probabilities
            chars, probs = zip(*dist.items())

            # Cumulative sampling
            r = random.random()
            s = 0.0
            pick = chars[-1]  # fallback to last character

            for ch, p in zip(chars, probs):
                s += p
                if r <= s:
                    pick = ch
                    break

            out.append(pick)

        return "".join(out)

