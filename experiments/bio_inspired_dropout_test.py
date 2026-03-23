"""
Test whether bio-inspired dropout (activity-dependent) outperforms standard dropout.

Hypothesis: Standard dropout is uniform. Biological synaptic release is NOT uniform -
high-activity synapses show use-dependent depression (release less). This should
improve generalization by preventing any single feature from dominating.

Test: Train a simple network on a classification task with:
1. No dropout (baseline)
2. Standard dropout (uniform p=0.5)
3. Bio-inspired dropout (p increases with activity)

Measure: Generalization gap (train acc - test acc). Lower = better generalization.
"""

import math
import random
from typing import List, Tuple

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def relu(x: float) -> float:
    return max(0, x)

class SimpleNetwork:
    """2-layer network with configurable dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        random.seed(seed)

        # Initialize weights
        scale1 = math.sqrt(2.0 / input_dim)
        scale2 = math.sqrt(2.0 / hidden_dim)

        self.W1 = [[random.gauss(0, scale1) for _ in range(input_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim

        self.W2 = [[random.gauss(0, scale2) for _ in range(hidden_dim)]
                   for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

        # Activity tracking for bio-inspired dropout
        self.hidden_activity = [0.0] * hidden_dim
        self.activity_decay = 0.9

    def forward(self, x: List[float], dropout_mode: str = "none",
                p_dropout: float = 0.5, training: bool = True) -> List[float]:
        """
        Forward pass with configurable dropout.

        dropout_mode:
            "none": No dropout
            "standard": Uniform dropout with probability p
            "bio": Activity-dependent dropout (high activity → more dropout)
        """
        # Hidden layer
        hidden = []
        for i in range(len(self.W1)):
            h = sum(self.W1[i][j] * x[j] for j in range(len(x))) + self.b1[i]
            h = relu(h)
            hidden.append(h)

        # Update activity tracking
        for i in range(len(hidden)):
            self.hidden_activity[i] = (self.activity_decay * self.hidden_activity[i] +
                                       (1 - self.activity_decay) * abs(hidden[i]))

        # Apply dropout
        if training and dropout_mode != "none":
            if dropout_mode == "standard":
                # Uniform dropout
                for i in range(len(hidden)):
                    if random.random() < p_dropout:
                        hidden[i] = 0.0
                    else:
                        hidden[i] /= (1 - p_dropout)  # Scale to maintain expected value

            elif dropout_mode == "bio":
                # Activity-dependent dropout
                # High activity → higher dropout probability (but gentler)
                max_activity = max(self.hidden_activity) + 1e-6
                for i in range(len(hidden)):
                    # Normalize activity to [0, 1]
                    normalized_activity = self.hidden_activity[i] / max_activity
                    # Adaptive dropout: mostly base, small activity-dependent term
                    p_adaptive = p_dropout * 0.8 + p_dropout * 0.4 * normalized_activity
                    p_adaptive = min(0.7, p_adaptive)  # Lower cap

                    if random.random() < p_adaptive:
                        hidden[i] = 0.0
                    else:
                        hidden[i] /= (1 - p_adaptive)

        # Output layer (no dropout)
        output = []
        for i in range(len(self.W2)):
            o = sum(self.W2[i][j] * hidden[j] for j in range(len(hidden))) + self.b2[i]
            output.append(o)

        return output

    def predict(self, x: List[float]) -> int:
        """Get predicted class."""
        output = self.forward(x, dropout_mode="none", training=False)
        return output.index(max(output))

    def train_step(self, x: List[float], y: int, lr: float,
                   dropout_mode: str, p_dropout: float) -> float:
        """Single training step with gradient descent. Returns loss."""
        # Forward pass
        # Hidden layer
        hidden_pre = []
        for i in range(len(self.W1)):
            h = sum(self.W1[i][j] * x[j] for j in range(len(x))) + self.b1[i]
            hidden_pre.append(h)

        hidden = [relu(h) for h in hidden_pre]

        # Track and apply dropout
        for i in range(len(hidden)):
            self.hidden_activity[i] = (self.activity_decay * self.hidden_activity[i] +
                                       (1 - self.activity_decay) * abs(hidden[i]))

        dropout_mask = [1.0] * len(hidden)
        if dropout_mode != "none":
            if dropout_mode == "standard":
                for i in range(len(hidden)):
                    if random.random() < p_dropout:
                        dropout_mask[i] = 0.0
                    else:
                        dropout_mask[i] = 1.0 / (1 - p_dropout)
            elif dropout_mode == "bio":
                max_activity = max(self.hidden_activity) + 1e-6
                for i in range(len(hidden)):
                    normalized_activity = self.hidden_activity[i] / max_activity
                    p_adaptive = p_dropout * 0.8 + p_dropout * 0.4 * normalized_activity
                    p_adaptive = min(0.7, p_adaptive)
                    if random.random() < p_adaptive:
                        dropout_mask[i] = 0.0
                    else:
                        dropout_mask[i] = 1.0 / (1 - p_adaptive)

        hidden_dropped = [h * m for h, m in zip(hidden, dropout_mask)]

        # Output layer
        output = []
        for i in range(len(self.W2)):
            o = sum(self.W2[i][j] * hidden_dropped[j] for j in range(len(hidden_dropped))) + self.b2[i]
            output.append(o)

        # Softmax
        max_o = max(output)
        exp_output = [math.exp(o - max_o) for o in output]
        sum_exp = sum(exp_output)
        probs = [e / sum_exp for e in exp_output]

        # Cross-entropy loss
        loss = -math.log(probs[y] + 1e-10)

        # Backward pass
        # Output gradient
        d_output = probs.copy()
        d_output[y] -= 1.0

        # Gradient for W2, b2
        for i in range(len(self.W2)):
            for j in range(len(hidden_dropped)):
                self.W2[i][j] -= lr * d_output[i] * hidden_dropped[j]
            self.b2[i] -= lr * d_output[i]

        # Backprop through hidden
        d_hidden_dropped = [0.0] * len(hidden_dropped)
        for j in range(len(hidden_dropped)):
            for i in range(len(self.W2)):
                d_hidden_dropped[j] += self.W2[i][j] * d_output[i]

        # Backprop through dropout
        d_hidden = [d * m for d, m in zip(d_hidden_dropped, dropout_mask)]

        # Backprop through ReLU
        d_hidden_pre = [d if h > 0 else 0 for d, h in zip(d_hidden, hidden_pre)]

        # Gradient for W1, b1
        for i in range(len(self.W1)):
            for j in range(len(x)):
                self.W1[i][j] -= lr * d_hidden_pre[i] * x[j]
            self.b1[i] -= lr * d_hidden_pre[i]

        return loss


def generate_data(n_samples: int, n_features: int, n_classes: int,
                  noise: float = 0.1, seed: int = 42) -> Tuple[List[List[float]], List[int]]:
    """Generate synthetic classification data with some dominant features."""
    random.seed(seed)

    X = []
    y = []

    # Each class has a "dominant" feature that strongly indicates it
    # This tests whether dropout prevents over-reliance on single features

    for _ in range(n_samples):
        label = random.randint(0, n_classes - 1)

        # Base features (noise)
        x = [random.gauss(0, noise) for _ in range(n_features)]

        # Dominant feature for this class (medium signal - make it harder)
        dominant_idx = label * (n_features // n_classes)
        x[dominant_idx] += 1.0  # Medium signal (was 2.0)

        # Secondary features (weaker signal)
        for i in range(n_features):
            if i // (n_features // n_classes) == label:
                x[i] += 0.2  # Weak class signal (was 0.5)

        X.append(x)
        y.append(label)

    return X, y


def accuracy(net: SimpleNetwork, X: List[List[float]], y: List[int]) -> float:
    correct = sum(1 for xi, yi in zip(X, y) if net.predict(xi) == yi)
    return correct / len(y)


def run_experiment(n_trials: int = 5, n_epochs: int = 50) -> dict:
    """Compare dropout methods across multiple trials."""

    results = {
        "none": {"train_acc": [], "test_acc": [], "gen_gap": []},
        "standard": {"train_acc": [], "test_acc": [], "gen_gap": []},
        "bio": {"train_acc": [], "test_acc": [], "gen_gap": []}
    }

    for trial in range(n_trials):
        seed = 42 + trial

        # Generate data - HARDER: more classes, more noise, weaker signal
        X_train, y_train = generate_data(300, 30, 6, noise=0.8, seed=seed)
        X_test, y_test = generate_data(150, 30, 6, noise=0.8, seed=seed + 1000)

        for dropout_mode in ["none", "standard", "bio"]:
            # Fresh network for each mode - wider network more prone to overfit
            net = SimpleNetwork(30, 64, 6, seed=seed)

            # Train
            for epoch in range(n_epochs):
                indices = list(range(len(X_train)))
                random.shuffle(indices)

                for i in indices:
                    net.train_step(X_train[i], y_train[i], lr=0.01,
                                   dropout_mode=dropout_mode, p_dropout=0.5)

            # Evaluate
            train_acc = accuracy(net, X_train, y_train)
            test_acc = accuracy(net, X_test, y_test)
            gen_gap = train_acc - test_acc

            results[dropout_mode]["train_acc"].append(train_acc)
            results[dropout_mode]["test_acc"].append(test_acc)
            results[dropout_mode]["gen_gap"].append(gen_gap)

    return results


def main():
    print("=" * 60)
    print("BIO-INSPIRED DROPOUT TEST")
    print("=" * 60)
    print()
    print("Testing whether activity-dependent dropout improves generalization")
    print("compared to standard uniform dropout.")
    print()

    results = run_experiment(n_trials=30, n_epochs=100)

    print("Results (mean ± std across 10 trials):")
    print("-" * 60)
    print(f"{'Method':<15} {'Train Acc':<15} {'Test Acc':<15} {'Gen Gap':<15}")
    print("-" * 60)

    for mode in ["none", "standard", "bio"]:
        train_mean = sum(results[mode]["train_acc"]) / len(results[mode]["train_acc"])
        test_mean = sum(results[mode]["test_acc"]) / len(results[mode]["test_acc"])
        gap_mean = sum(results[mode]["gen_gap"]) / len(results[mode]["gen_gap"])

        train_std = math.sqrt(sum((x - train_mean)**2 for x in results[mode]["train_acc"]) / len(results[mode]["train_acc"]))
        test_std = math.sqrt(sum((x - test_mean)**2 for x in results[mode]["test_acc"]) / len(results[mode]["test_acc"]))
        gap_std = math.sqrt(sum((x - gap_mean)**2 for x in results[mode]["gen_gap"]) / len(results[mode]["gen_gap"]))

        print(f"{mode:<15} {train_mean:.3f}±{train_std:.3f}     {test_mean:.3f}±{test_std:.3f}     {gap_mean:.3f}±{gap_std:.3f}")

    print()

    # Statistical comparison: bio vs standard
    bio_test = results["bio"]["test_acc"]
    std_test = results["standard"]["test_acc"]

    bio_mean = sum(bio_test) / len(bio_test)
    std_mean = sum(std_test) / len(std_test)

    # Paired difference
    diffs = [b - s for b, s in zip(bio_test, std_test)]
    diff_mean = sum(diffs) / len(diffs)
    diff_std = math.sqrt(sum((d - diff_mean)**2 for d in diffs) / len(diffs))

    # t-statistic
    t_stat = diff_mean / (diff_std / math.sqrt(len(diffs)) + 1e-10)

    print("Bio vs Standard comparison:")
    print(f"  Mean difference in test acc: {diff_mean:+.4f}")
    print(f"  t-statistic: {t_stat:.3f}")

    if abs(t_stat) > 2.26:  # ~p<0.05 for 9 df
        if diff_mean > 0:
            print("  → Bio-inspired dropout significantly BETTER")
        else:
            print("  → Standard dropout significantly BETTER")
    else:
        print("  → No significant difference")

    print()

    # Generalization gap comparison
    bio_gap = results["bio"]["gen_gap"]
    std_gap = results["standard"]["gen_gap"]

    gap_diffs = [s - b for b, s in zip(bio_gap, std_gap)]  # Positive = bio has smaller gap
    gap_diff_mean = sum(gap_diffs) / len(gap_diffs)

    print("Generalization gap comparison (lower = better):")
    print(f"  Bio gap mean: {sum(bio_gap)/len(bio_gap):.4f}")
    print(f"  Standard gap mean: {sum(std_gap)/len(std_gap):.4f}")
    print(f"  Improvement: {gap_diff_mean:+.4f}")

    return results


if __name__ == "__main__":
    results = main()
