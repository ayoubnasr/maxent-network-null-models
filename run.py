import argparse

import numpy as np
from data.loaders import load_karate
from models.soft_configuration_model import fit, link_probabilities, sample

def run_soft_config_analysis(G, sample_nb = 10_000, rng=42):
    # Define the constraint vector:
    k = np.array([G.degree[i] for i in G.nodes])

    # initialize multipliers:
    lam, P = fit(k)

    # How far the nodes' expected degrees are from the constraints we set:
    print("Fit max degree error = ", np.max(np.abs(P.sum(axis=1) - k)))

    A = sample(P, rng)
    print("\nCompare expected total number of edges to the total number of edges in our sample:")
    print("Expected edges:", P.sum() // 2, "Edges in one sample:", A.sum() // 2, "\n")

    sample_deg = np.zeros(len(k))

    for i in range(sample_nb):
        A = sample(P, rng)
        sample_deg += A.sum(axis=1)
    
    print("Avg degrees:", sample_deg / sample_nb)
    print("Max error:", np.max(np.abs(sample_deg / sample_nb - k)))

DATASETS = {
    "karate": load_karate,
    # "facebook": load_facebook, (later)
}

ANALYSES = {
    "soft_config": run_soft_config_analysis,
}

def main():
    parser = argparse.ArgumentParser(description="Max-entropy null model analysis")
    parser.add_argument("model", choices=ANALYSES.keys(), help="which null model to run")
    parser.add_argument("dataset", choices=DATASETS.keys(), help="which network to analyse")
    parser.add_argument("--samples", type=int, default=1000, help="number of sampled graphs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    G = DATASETS[args.dataset]()
    rng = np.random.default_rng(args.seed)
    ANALYSES[args.model](G,args.samples, rng)


if __name__ == "__main__":
    main()