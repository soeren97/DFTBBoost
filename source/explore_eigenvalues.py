import re
import os
from tqdm import tqdm
import numpy as np
from source.utils import freedman_diaconis_bins
import matplotlib.pyplot as plt
import pickle


def load_eigenvalues(path):
    eigenvalue_list = []
    generator = os.walk(path)

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
                r"(?<=Population analysis using the SCF density\.\n\n \*{70}\n)\n([\S\s]*)Condensed to atoms",
            )
            eigenvalues_match = eigenvalues_pattern.search(log_text)
            if eigenvalues_match:
                eigenvalues_lines = eigenvalues_match.group(1).strip().split("\n")
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
