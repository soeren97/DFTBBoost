"""Explore eigenenergies to confirm distribution is calculated correctly. """

import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


def load_eigenvalues(path: str) -> NDArray:
    """Load Eigenvalues from folder of G16 calculations.

    Args:
        path (str): Path to folder

    Returns:
        NDArray: Eigenenergies.
    """
    eigenvalue_list = []
    generator = os.walk(path)
    regex_string = (
        r"(?<=Population analysis using the SCF density"
        + r"\.\n\n \*{70}\n)\n([\S\s]*)Condensed to atoms"
    )

    # skip first iteration of the generator
    next(generator)

    for root, _, _ in tqdm(
        generator,
        total=len(os.listdir(path)),
        mininterval=0.2,
        desc="Loading eigenvalues",
    ):
        log_path = os.path.join(root, "mol.log")
        with open(log_path, "r") as log_file:
            log_text = log_file.read()
            eigenvalues_pattern = re.compile(
                regex_string,
            )
            match = eigenvalues_pattern.search(log_text)
            if match:
                eigenvalues_lines = match.group(1).strip().split("\n")
                eigenvalues = []
                for line in eigenvalues_lines:
                    line = re.sub(r"[^0-9\s.-]", "", line)
                    line = re.sub(r"--", "", line)
                    line = re.sub(r"\s\.\s", "", line)
                    eigenvalues += [float(x) for x in line.strip().split()]
                eigenvalue_list.extend(eigenvalues)
            else:
                print("Eigenvalues not found in log file")
    return np.array(eigenvalue_list)


dir = os.path.join(os.getcwd(), "Data/G16_test_eigenvalues/")
pickle_location = os.path.join(os.getcwd(), "Data/G16_eigenvalues.pkl")

if os.path.exists(pickle_location):
    with open(pickle_location, "rb") as f:
        eigenvalues = pickle.load(f)
else:
    eigenvalues = load_eigenvalues(dir)
    with open(pickle_location, "wb") as f:
        pickle.dump(eigenvalues, f)

# convert to ev
eigenvalues *= 27.2114

bins = 1000  # freedman_diaconis_bins(eigenvalues) * 2
plt.hist(eigenvalues, bins=bins)
plt.xlabel("Eigen energies [eV]")
plt.ylabel("Frequency")
# plt.yscale("log")
plt.xlim(left=-20, right=20)
plt.tight_layout()
plt.savefig("Figures/energy_from_g16_eV")
