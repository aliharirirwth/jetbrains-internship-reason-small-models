# Gradient derivation: Skip-gram with negative sampling

This document derives the gradients used in the NumPy implementation so you can defend them in follow-up.

## Setup

- **Input (center) embedding** \( v_w = W_{\text{in}}[w] \in \mathbb{R}^D \)
- **Output (context) embedding** \( u_c = W_{\text{out}}[c] \in \mathbb{R}^D \)
- **Score** for pair \((w, c)\): \( s(w,c) = u_c^\top v_w \)

For a batch we have:
- One **positive** pair \((w, c)\) per example: score \( \sigma_{\text{pos}} = u_c^\top v_w \)
- **K negative** pairs \((w, n_k)\) per example: scores \( \sigma_{\text{neg},k} = u_{n_k}^\top v_w \)

## Loss

Per example (one center \(w\), one positive \(c\), K negatives \(n_1,\ldots,n_K\)):

\[
\mathcal{L} = -\log \sigma(\sigma_{\text{pos}}) - \sum_{k=1}^{K} \log \sigma(-\sigma_{\text{neg},k})
\]

where \(\sigma(x) = 1/(1+e^{-x})\).

We minimise the **mean** over the batch, so \(\mathcal{L}_{\text{batch}} = \frac{1}{B}\sum_b \mathcal{L}_b\). All gradients below are per-example; we sum over the batch and divide by \(B\) when updating (see code).

## Derivative of \(\log \sigma(x)\)

\[
\frac{d}{dx} \log \sigma(x) = \frac{\sigma'(x)}{\sigma(x)} = \frac{\sigma(x)(1-\sigma(x))}{\sigma(x)} = 1 - \sigma(x)
\]

So:
\[
\frac{d}{dx}\bigl[-\log \sigma(x)\bigr] = \sigma(x) - 1
\]

## Derivative of \(\log \sigma(-x)\) (negative sample term)

\[
\frac{d}{dx}\bigl[-\log \sigma(-x)\bigr] = \frac{d}{dx}\bigl[\log(1-\sigma(x))\bigr] = \frac{-\sigma'(x)}{1-\sigma(x)} = \sigma(x)
\]

So the gradient of the negative-sample term with respect to the **score** \(x = u_n^\top v_w\) is \(\sigma(x)\).

## Gradients w.r.t. embeddings

Let \(\sigma_{\text{pos}} = u_c^\top v_w\) and \(\sigma_{\text{neg},k} = u_{n_k}^\top v_w\).

- **Positive term** \(-\log \sigma(\sigma_{\text{pos}})\):
  - \(\displaystyle \frac{\partial \mathcal{L}}{\partial \sigma_{\text{pos}}} = \sigma(\sigma_{\text{pos}}) - 1 = g_{\text{pos}}\)
  - \(\displaystyle \frac{\partial \sigma_{\text{pos}}}{\partial v_w} = u_c \Rightarrow \frac{\partial \mathcal{L}}{\partial v_w} \mathrel{+}= g_{\text{pos}}\, u_c\)
  - \(\displaystyle \frac{\partial \sigma_{\text{pos}}}{\partial u_c} = v_w \Rightarrow \frac{\partial \mathcal{L}}{\partial u_c} \mathrel{+}= g_{\text{pos}}\, v_w\)

- **Negative term** \(-\log \sigma(-\sigma_{\text{neg},k})\):
  - \(\displaystyle \frac{\partial \mathcal{L}}{\partial \sigma_{\text{neg},k}} = \sigma(\sigma_{\text{neg},k}) = g_{\text{neg},k}\)
  - \(\displaystyle \frac{\partial \mathcal{L}}{\partial v_w} \mathrel{+}= g_{\text{neg},k}\, u_{n_k}\)
  - \(\displaystyle \frac{\partial \mathcal{L}}{\partial u_{n_k}} \mathrel{+}= g_{\text{neg},k}\, v_w\)

So **per example**:
- **\(v_w\)**: \(g_{\text{pos}} u_c + \sum_k g_{\text{neg},k} u_{n_k}\)
- **\(u_c\)**: \(g_{\text{pos}} v_w\)
- **\(u_{n_k}\)**: \(g_{\text{neg},k} v_w\)

The code implements exactly this and uses `np.add.at` to scatter the gradients into the full \(W_{\text{in}}\) and \(W_{\text{out}}\) matrices (because the same word type can appear in many batch positions).

## Subtle point: same word as positive and negative

If in the same batch the same word \(c\) appears both as a positive context and as a negative sample for another example, we update \(u_c\) twice: once with \(g_{\text{pos}} v_w\) and once with \(g_{\text{neg}} v_w\). The **correct** handling is to **add** these gradients (which we do naturally by accumulating with `np.add.at`). This is the part many implementations get wrong when they overwrite instead of add.

## Implementation convention (gradient sign)

The formulas above give \(\partial\mathcal{L}/\partial W\). The training loop does **W -= lr * grad**, so we need **grad = \(\partial\mathcal{L}/\partial W\)** for gradient descent. The code instead computes the *descent direction* \(-\partial\mathcal{L}/\partial W\) in the internal variables, then returns **grad = \(-(-\partial\mathcal{L}/\partial W) = \partial\mathcal{L}/\partial W\)**. So the caller receives the true gradient and **W -= lr * grad** correctly decreases \(\mathcal{L}\). When tracing live: internal dW is the descent direction; the final `return loss, -dW_in, -dW_out` converts it to the gradient.

## Numerical stability

- **Sigmoid**: we clip the argument to \([-500, 500]\) so \(\exp(-x)\) does not overflow.
- **\(\log \sigma(x)\)**: we use \(\log \sigma(x) = -\max(x,0) - \log(1+e^{-|x|})\) so we never compute \(\exp(x)\) for large positive \(x\) (which would overflow). This is the standard stable formula.
