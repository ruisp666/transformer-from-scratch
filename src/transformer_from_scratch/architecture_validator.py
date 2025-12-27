
import torch
from transformer_from_scratch.transformer_blk import TransformerBlock
from transformer_from_scratch.positional_embeddings import SinusoidalPositionalEncoding
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

"""
Test the Decay of the gradient throught the observation of the SVDs for both architectures, with and without positional encoding"""
def effective_dimension(s):
    # Participation Ratio: (sum of eigenvalues)^2 / (sum of squared eigenvalues)
    # Note: eigenvalues of the covariance are s^2
    eigs = s**2
    return (eigs.sum()**2) / (eigs.pow(2).sum())


def get_pca_explained_variance(s):
    """Calculates the ratio of variance explained by each singular value."""
    variance = s**2
    return (variance / variance.sum()).detach().numpy()

def test_svds():
    d_model = 32
    n_layers = 40
    seq_len = 128
    transformer_pre_ln = nn.Sequential(*[TransformerBlock(d_model,4,0.1, 'Pre-Ln') for _ in range(n_layers)])
    transformer_post_ln = nn.Sequential(*[TransformerBlock(d_model,4,0.1, 'Post-Ln') for _ in range(n_layers)])

    # Add positional encoding
    transformer_pre_ln_with_positional_enc = nn.Sequential(SinusoidalPositionalEncoding(seq_len, d_model),*[TransformerBlock(d_model,4,0.1, 'Pre-Ln') for _ in range(n_layers)])
    X_input = torch.randn(1,seq_len,d_model)

    with torch.no_grad():
        y_pre = transformer_pre_ln(X_input)
        y_post = transformer_post_ln(X_input)
        y_pre_with_positional_enc = transformer_pre_ln_with_positional_enc(X_input)

    s_pre = torch.linalg.svdvals(y_pre.squeeze(0))
    s_post = torch.linalg.svdvals(y_post.squeeze(0))
    s_pre_with_positional_enc = torch.linalg.svdvals(y_pre_with_positional_enc.squeeze(0))
    print(f"Pre-LN Max Singular Value: {s_pre.max().item():.4f}")
    print(f"Post-LN Max Singular Value: {s_post.max().item():.4f}")
    print(f"Effective Dimension (Pre-LN): {effective_dimension(s_pre):.2f}")
    print(f"Effective Dimension (Post-LN): {effective_dimension(s_post):.2f}")
    print(f"Effective Dimension (Pre-LN-with-positional-encoding): {effective_dimension(s_pre_with_positional_enc):.2f}")

    print("Top 5 Singular Values (Pre-LN): ", s_pre[:5].detach().numpy())
    print("Top 5 Singular Values (Post-LN):", s_post[:5].detach().numpy())
    print("Top 5 Singular Values (Pre-LN-with-positional-encoding):", s_pre_with_positional_enc[:5].detach().numpy())

    # 2. Compute Cumulative Variance Explained
    var_pre = get_pca_explained_variance(s_pre)
    var_post = get_pca_explained_variance(s_post)
    var_pre_with_positional_enc = get_pca_explained_variance(s_pre_with_positional_enc)
    cum_var_pre = np.cumsum(var_pre)
    cum_var_post = np.cumsum(var_post)
    cum_var_pre_with_positional_enc = np.cumsum(var_pre_with_positional_enc)

    # 3. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, d_model + 1), cum_var_pre, marker='o', label=f'Pre-LN (Depth {n_layers})', color='blue')
    plt.plot(range(1, d_model + 1), cum_var_post, marker='x', label=f'Post-LN (Depth {n_layers})', color='red')

    plt.plot(range(1, d_model + 1), cum_var_pre_with_positional_enc, marker='x', label=f'Pre-LN-with-positional-enc (Depth {n_layers})', color='green')

    plt.axhline(y=0.90, color='gray', linestyle='--', label='90% Variance Explained')
    plt.title("PCA: Cumulative Variance Explained by Component")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Cumulative Proportion of Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Log the number of components needed for 90% variance
    comp_pre = np.argmax(cum_var_pre >= 0.90) + 1
    comp_post = np.argmax(cum_var_post >= 0.90) + 1
    comp_pre_with_positional_enc = np.argmax(cum_var_pre_with_positional_enc >= 0.9) + 1
    print(f"Components for 90% var (Pre): {comp_pre} | (Post): {comp_post}")
    print(f"Components for 90% var (Pre with positional encoding): {comp_pre_with_positional_enc} ")

if __name__=='__main__':
    test_svds()
