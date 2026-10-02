from scipy.linalg import eigh
import numpy as np
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
    dataset = np.load(filename).astype(np.float64)

    # Subtract the feature-wise mean so each column (pixel) has mean zero.
    return dataset - np.mean(dataset, axis=0)

def get_covariance(dataset):
    """
    Calculate the sample covariance matrix of the dataset.

    Args:
        dataset (numpy.ndarray): Centered dataset (n x d matrix)

    Returns:
        numpy.ndarray: Covariance matrix (d x d matrix)
    """
    x = np.asarray(dataset, dtype=np.float64)
    n = x.shape[0]

    # S = 1/(n-1) * sum_i x_i x_i^T = 1/(n-1) * X^T X for row-wise data.
    return np.dot(np.transpose(x), x) / (n - 1)

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
    d = S.shape[0]

    # eigh returns eigenvalues in ascending order; grab the top k.
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[d - k, d - 1])

    # Flip to descending order, keeping eigenvectors in matching columns.
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return np.diag(eigenvalues), eigenvectors

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
    # The total variance is the sum of all eigenvalues, i.e. the trace of S.
    total_variance = np.trace(S)

    # subset_by_value uses the half-open interval (a, b], so this keeps every
    # eigenvalue explaining strictly more than prop of the variance.
    eigenvalues, eigenvectors = eigh(
        S, subset_by_value=[prop * total_variance, np.inf]
    )

    # Flip to descending order, keeping eigenvectors in matching columns.
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    return np.diag(eigenvalues), eigenvectors

def project_and_reconstruct_image(image, U):
    """
    Project image to PCA subspace and reconstruct it back to original dimension.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)

    Returns:
        numpy.ndarray: Reconstructed image as flattened d x 1 vector
    """
    x = np.asarray(image, dtype=np.float64).reshape(-1)

    # alpha = U^T x is the m-dimensional projection ("score").
    alpha = np.dot(np.transpose(U), x)

    # x_pca = U alpha brings it back to the original d-dimensional space.
    return np.dot(U, alpha)

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
    x = np.asarray(image, dtype=np.float64).reshape(-1)

    alpha = np.dot(np.transpose(U), x)

    # Perturb the PCA coefficients with i.i.d. noise from N(0, sigma^2).
    rng = np.random.default_rng(seed)
    noisy_alpha = alpha + rng.normal(loc=0.0, scale=sigma, size=alpha.shape)

    return np.dot(U, noisy_alpha)

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

    # Reshape the flattened vectors back into images.
    fullres = np.asarray(im_orig_fullres).reshape(218, 178, 3)
    orig = np.asarray(im_orig, dtype=np.float64).reshape(60, 50)
    reconstructed = np.asarray(im_reconstructed, dtype=np.float64).reshape(60, 50)

    ax1.set_title('Original High Res')
    ax2.set_title('Original')
    ax3.set_title('Reconstructed')

    ax1.imshow(fullres, aspect='equal')
    im2 = ax2.imshow(orig, aspect='equal', cmap='gray')
    im3 = ax3.imshow(reconstructed, aspect='equal', cmap='gray')

    # Colorbars on the right of the two low-resolution grayscale images.
    fig.colorbar(im2, ax=ax2, location='right')
    fig.colorbar(im3, ax=ax3, location='right')

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
        self.counts = {}
        self.vocab = set()

        for i in range(len(text)):
            ch = text[i]

            # Context is the (up to) n-1 characters immediately before position i.
            ctx = text[max(0, i - (self.n - 1)):i]

            self.vocab.add(ch)

            ctx_counts = self.counts.setdefault(ctx, {})
            ctx_counts[ch] = ctx_counts.get(ch, 0) + 1

        self.trained = True
        return self
    
    def _probs_for_context(self, context: str):
        """
        Get probability distribution over next characters for a given context.
        
        Args:
            context (str): The context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities
        """
        # Only the last n-1 characters matter (for n=1 the context is always empty).
        ctx = context[-(self.n - 1):] if self.n > 1 else ""

        ctx_counts = self.counts.get(ctx)
        total = sum(ctx_counts.values()) if ctx_counts else 0

        # Unseen (or empty) context: fall back to a uniform distribution so that
        # generation stays well-defined.
        if total == 0:
            if not self.vocab:
                return {}
            uniform = 1.0 / len(self.vocab)
            return {c: uniform for c in self.vocab}

        # Every vocabulary character is included; unseen ones get probability 0.
        return {c: ctx_counts.get(c, 0) / total for c in self.vocab}
    
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

