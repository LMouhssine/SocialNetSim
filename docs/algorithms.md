# SocialNetSim Algorithm Documentation

This document describes the key algorithms and mathematical models used in SocialNetSim.

## Table of Contents

1. [Utility-Based Decision Model](#utility-based-decision-model)
2. [Hawkes Process for Virality](#hawkes-process-for-virality)
3. [Bounded Confidence Opinion Dynamics](#bounded-confidence-opinion-dynamics)
4. [Bandit-Optimized Feed Ranking](#bandit-optimized-feed-ranking)
5. [Information Diffusion Model](#information-diffusion-model)
6. [Assumptions and Limitations](#assumptions-and-limitations)

---

## Utility-Based Decision Model

### Overview

The utility-based decision model replaces simple probabilistic engagement with a principled framework based on user utility maximization. Users evaluate content based on multiple utility components and engage when expected utility exceeds their threshold.

### Mathematical Formulation

Total utility for user $u$ engaging with post $p$:

$$U(u, p) = \sum_{i} w_i \cdot U_i(u, p)$$

Where:

- $w_i$ = weight for utility component $i$
- $U_i$ = individual utility component

### Utility Components

#### 1. Information Gain

Measures novelty of content relative to user's exposure history:

$$U_{info}(u, p) = 1 - \exp(-\alpha \cdot novelty(p, u))$$

Where $novelty(p, u) = 1 - \frac{exposure\_count(topic)}{max\_exposure}$

#### 2. Social Utility

Based on author relationship and social factors:

$$U_{social}(u, p) = \beta_1 \cdot following(u, author) + \beta_2 \cdot influence(author)$$

#### 3. Emotional Resonance

Alignment between content emotion and user's emotional state:

$$U_{emotion}(u, p) = reactivity(u) \cdot |emotion(p)| \cdot alignment(valence_u, sentiment_p)$$

#### 4. Cognitive Cost

Attention depletion penalty:

$$U_{cognitive}(u, p) = -\gamma \cdot (1 - attention\_budget(u))$$

#### 5. Ideological Alignment

Based on confirmation bias and opinion matching:

$$U_{ideology}(u, p) = bias(u) \cdot similarity(ideology_u, stance_p) - (1 - bias(u)) \cdot |ideology_u - stance_p|$$

### Decision Process

Engagement probability is computed via softmax:

$$P(engage) = \sigma(U(u, p) - threshold)$$

Where $\sigma$ is the sigmoid function and $threshold$ is user-specific.

---

## Hawkes Process for Virality

### Overview

Hawkes processes model self-exciting point processes where past events increase the likelihood of future events. This captures the "momentum" of viral cascades.

### Mathematical Formulation

The intensity function at time $t$:

$$\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$$

Where:

- $\mu$ = baseline intensity (spontaneous events)
- $\alpha$ = excitation parameter (how much each event increases intensity)
- $\beta$ = decay rate (how quickly excitement fades)
- $t_i$ = times of previous events

### Branching Ratio

The expected number of child events per parent event:

$$R = \frac{\alpha}{\beta}$$

- $R < 1$: Subcritical (cascade dies out)
- $R = 1$: Critical (borderline viral)
- $R > 1$: Supercritical (explosive growth)

### Event Sampling

Events are sampled using Ogata's thinning algorithm:

1. Compute intensity upper bound $\lambda^*$
2. Propose event at time $t + \Delta$ where $\Delta \sim Exp(\lambda^*)$
3. Accept with probability $\lambda(t + \Delta) / \lambda^*$
4. Repeat until termination

### Cascade-Specific Parameters

Virality score modulates intensity:

$$\lambda_{cascade}(t) = \mu \cdot (1 + virality\_score) + \sum_{t_i < t} \alpha_{cascade} \cdot e^{-\beta(t - t_i)}$$

---

## Bounded Confidence Opinion Dynamics

### Overview

The Deffuant-Weisbuch bounded confidence model simulates opinion formation where agents only interact with others whose opinions are sufficiently similar.

### Mathematical Formulation

For two users $i$ and $j$ with opinions $o_i$ and $o_j$:

**Interaction condition:**
$$|o_i - o_j| < \epsilon$$

Where $\epsilon$ is the confidence bound (tolerance threshold).

**Opinion update (if condition met):**
$$o_i' = o_i + \mu \cdot (o_j - o_i)$$
$$o_j' = o_j + \mu \cdot (o_i - o_j)$$

Where $\mu$ is the convergence rate (typically 0.1-0.5).

### Content Influence

Content exposure also affects opinions:

$$o_i' = o_i + \mu_{content} \cdot (stance_p - o_i) \cdot receptivity(i)$$

Where $receptivity(i) = 1 - confidence(i)$

### Stubbornness

Users with high opinion confidence become stubborn:

$$effective\_\mu = \mu \cdot (1 - stubbornness(i))$$

$$stubbornness(i) = \max(0, confidence(i) - threshold)$$

### Polarization Metrics

**Bimodality Coefficient (Sarle's):**

$$BC = \frac{skewness^2 + 1}{kurtosis}$$

- $BC > 0.555$: Bimodal distribution (polarized)
- $BC < 0.555$: Unimodal distribution

**Echo Chamber Index:**

$$ECI = \frac{E_{within}}{E_{within} + E_{between}}$$

Where $E_{within}$ = edges within opinion clusters, $E_{between}$ = edges between clusters.

---

## Bandit-Optimized Feed Ranking

### Overview

Feed ranking weights are optimized using multi-armed bandit algorithms to maximize engagement while exploring different ranking strategies.

### Thompson Sampling

For each arm (weight configuration) $k$:

1. Maintain Beta distribution parameters: $(\alpha_k, \beta_k)$
2. Sample $\theta_k \sim Beta(\alpha_k, \beta_k)$
3. Select arm with highest $\theta_k$
4. Update: $\alpha_k += reward$, $\beta_k += (1 - reward)$

### LinUCB (Contextual Bandit)

For user context $x$ and arm $a$:

$$score_a = x^T \theta_a + \alpha \sqrt{x^T A_a^{-1} x}$$

Where:

- $\theta_a$ = estimated parameter vector for arm $a$
- $A_a$ = design matrix for arm $a$
- $\alpha$ = exploration parameter

**Parameter update after reward $r$:**
$$A_a = A_a + x x^T$$
$$b_a = b_a + r \cdot x$$
$$\theta_a = A_a^{-1} b_a$$

### Ranking Score Function

$$score(u, p) = w_1 \cdot recency(p) + w_2 \cdot velocity(p) + w_3 \cdot relevance(u, p) + w_4 \cdot controversy(p) + w_5 \cdot social\_proximity(u, p)$$

Where weights $w_i$ are selected by the bandit algorithm.

---

## Information Diffusion Model

### Overview

An SIR-like model for information spread with fatigue, saturation, and backlash effects.

### States

- **Susceptible (S)**: Not yet exposed to information
- **Exposed (E)**: Exposed but not yet sharing
- **Infected (I)**: Actively sharing
- **Recovered (R)**: No longer sharing (lost interest)
- **Immune (I)**: Cannot be re-infected (fatigue/backlash)

### Transition Rates

**S → E (Exposure):**
$$rate_{SE} = \beta \cdot \frac{I}{N} \cdot (1 - saturation)$$

**E → I (Adoption):**
$$rate_{EI} = \gamma \cdot base\_share\_prob \cdot threshold\_factor$$

**I → R (Recovery):**
$$rate_{IR} = \delta \cdot time\_infected$$

### Saturation Effect

As content spreads, novelty decreases:

$$saturation = 1 - e^{-share\_count / K}$$

Where $K$ is the saturation constant (shares for 50% saturation).

### Backlash Effect

Overexposure leads to negative reactions:

$$backlash\_prob = \max(0, exposure\_count - threshold) \cdot backlash\_rate$$

### Effective Reproduction Number

$$R_t = R_0 \cdot (1 - saturation) \cdot susceptible\_fraction$$

Where $R_0$ is the basic reproduction number.

---

## Assumptions and Limitations

### Model Assumptions

1. **User Rationality**: Users approximately maximize utility (bounded rationality)
2. **Homogeneous Time Steps**: Discrete time steps of equal duration
3. **Complete Information**: Users can observe post characteristics
4. **Network Stability**: Social network structure is static during simulation
5. **Independent Decisions**: User decisions are conditionally independent given state

### Known Limitations

1. **Scale**: Full simulation tested up to 100k users; larger scales may require distributed computing
2. **Real-time**: Not suitable for real-time applications; designed for research simulations
3. **Content Understanding**: No actual NLP; content features are simulated
4. **Platform Specificity**: Generic social network model, not platform-specific

### Parameter Sensitivity

Critical parameters requiring careful calibration:

| Parameter | Typical Range | Impact |
|-----------|---------------|--------|
| Hawkes $\alpha/\beta$ | 0.5-1.0 | Cascade size distribution |
| Confidence bound $\epsilon$ | 0.2-0.5 | Polarization emergence |
| Utility weights | 0.0-1.0 | Engagement patterns |
| Saturation constant $K$ | 50-200 | Information saturation speed |

### Validation Recommendations

1. Compare cascade size distributions to power-law expectations
2. Verify engagement rates are 1-5% (typical for social platforms)
3. Check opinion dynamics don't converge too quickly or slowly
4. Ensure no single cascade reaches >50% of users (unrealistic)

---

## References

1. Hawkes, A. G. (1971). "Spectra of some self-exciting and mutually exciting point processes."
2. Deffuant, G., et al. (2000). "Mixing beliefs among interacting agents."
3. Li, L., et al. (2010). "A contextual-bandit approach to personalized news article recommendation."
4. Goel, S., et al. (2016). "The structural virality of online diffusion."
5. Sîrbu, A., et al. (2017). "Opinion dynamics: models, extensions and external effects."
