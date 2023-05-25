"""Transform the data from DFTB and DFT calculations into graphs."""
import gc
import logging
import os
import shutil
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pysmiles import read_smiles
from torch.nn.functional import pad
from torch_geometric.data import Data as GraphData
from torch_geometric.utils import from_networkx, from_smiles
from tqdm import tqdm
from utils import create_savefolder, find_eigenvalues_true

# Anything higher than warning is logged.
logging.getLogger("pysmiles").setLevel(logging.CRITICAL)


class DataTransformer:
    """Class used to hold variables for datatransformation."""

    def __init__(self) -> None:
        self.device = None

        self.data_location = None
        self.data_location_dftb = None
        self.data_location_g16 = None
        self.data_interval = None

        self.max_dim = 65
        self.elements = None

        self.dftb_data = None
        self.g16_data = None
        self.data_unpadded = None
        self.data = pd.DataFrame()

    def extract_data_dftb(self) -> None:
        """Extract data from DFTB calculations.

        Files are read and names, coordinates, overlap and Fock are extracted.
        """
        generator = os.walk(self.data_location_dftb)

        # skip first iteration of the generator
        next(generator)

        data_list = []

        for root, _, _ in tqdm(
            generator,
            total=len(os.listdir(self.data_location_dftb)),
            mininterval=0.2,
            desc="Loading DFTB+ data",
        ):
            try:
                # Extract input

                file_location = root + "/mol.xyz"

                file = open(file_location)

                num = int(file.readlines()[0])

                file.close()

                try:
                    xyz_file = np.genfromtxt(
                        fname=file_location, skip_header=2, dtype="unicode"
                    )
                except ValueError:
                    xyz_file = np.genfromtxt(
                        fname=file_location, skip_header=3, dtype="unicode"
                    )

                xyz_file = np.char.replace(xyz_file, "*^", "e")

                xyz_file[:, 1:] = xyz_file[:, 1:].astype("float64")

                xyz_file[:, 0] = xyz_file[:, 0].astype(str)

                title = str(
                    np.genfromtxt(
                        fname=file_location,
                        comments=None,
                        skip_header=1,
                        skip_footer=num,
                        dtype="unicode",
                    )
                )

                # Extract Hamiltonian
                file_location = root + "/hamsqr1.dat"

                if os.path.exists(file_location):
                    ham_file = np.genfromtxt(
                        fname=file_location, skip_header=4, dtype="unicode"
                    )

                    ham_file = ham_file.astype("float64")

                else:
                    ham_file = None

                # Extract Overlap
                file_location = root + "/oversqr.dat"

                if os.path.exists(file_location):
                    over_file = np.genfromtxt(
                        fname=file_location, skip_header=4, dtype="unicode"
                    )

                    over_file = over_file.astype("float64")

                else:
                    over_file = None

                data = [title, xyz_file, ham_file, over_file]

                data_list.append(data)

            except:
                pass
        self.dftb_data = pd.DataFrame(
            data_list,
            columns=["SMILES", "Coordinates", "Hamiltonian", "Overlap"],
        )
        self.dftb_data.dropna(inplace=True)
        gc.collect()

    def matrix_from_log(self, root: str, nbas: int, identifier: str) -> NDArray:
        """Extract the Fock and overlap matrices from the respective files.

        Args:
            root (str): Path to file.
            nbas (int): Number of basis sets.
            identifier (str): Name of file.

        Returns:
            NDArray: Extracted matrix.
        """
        try:
            matrix = []
            with open(root + identifier) as f:
                matrix_line = []
                for idx, line in enumerate(f.readlines()):
                    row_arr = line.split()
                    matrix_line.append(row_arr)

                    test_length = np.concatenate(matrix_line)
                    if test_length.shape[0] == nbas:
                        temp = [float(x.replace("D", "e")) for x in test_length]
                        matrix.append(temp)
                        matrix_line = []
            matrix = np.array(matrix)

        except:
            matrix = None
        return matrix

    def extract_data_g16(self) -> None:
        """Extract data from DFTB calculations.

        Files are read and names, coordinates, overlap and Fock are extracted.
        """
        generator = os.walk(self.data_location_g16)

        # skip first iteration of the generator
        next(generator)

        try:
            g16_data = pd.read_pickle("Data/G16_batched/" + self.data_interval + ".pkl")
            data_list = g16_data.values.tolist()
            del g16_data

        except FileNotFoundError:
            data_list = []

        for root, _, _ in tqdm(
            generator,
            total=len(os.listdir(self.data_location_g16)),
            mininterval=0.2,
            desc="Loading G16 data",
        ):
            try:
                # Extract input
                file_location = root + "/mol.com"

                file = open(file_location)

                title = file.readlines()[7]

                title = title[:-1]

                file.close()

                nbas = 0
                with open(root + "/mol.log") as f:
                    for line in f.readlines():
                        if "basis functions," in line:
                            nbas = int(line.split()[0])

                hamiltonian = self.matrix_from_log(root, nbas, "/hamiltonian.1")

                overlap = self.matrix_from_log(root, nbas, "/overlap")

                if len(hamiltonian) == 0:
                    hamiltonian = None

                if len(overlap) == 0:
                    overlap = None

                data = [title, hamiltonian, overlap]

                data_list.append(data)

            except:
                pass

        data_list = np.array(data_list, dtype=object)
        self.g16_data = pd.DataFrame(
            data_list, columns=["SMILES", "Hamiltonian_g16", "Overlap_g16"]
        )

        self.g16_data.drop_duplicates(subset="SMILES", inplace=True, ignore_index=True)

        self.g16_data.dropna(inplace=True)

        self.g16_data.to_pickle("Data/G16_batched/" + self.data_interval + ".pkl")

        shutil.rmtree(self.data_location_g16)

        os.mkdir(self.data_location_g16)

        gc.collect()

    def pad_and_tril(
        self, array: NDArray, overlap: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[None, None]:
        """Pad the matrix and converts it to a lower triangle array.

        Args:
            array (NDArray): The matrix to be converted.
            overlap (Optional[bool], optional): Is matrix overlap. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] | Tuple[None, None]:
            Padded and lower triangle matrix.
        """
        tensor = torch.from_numpy(array)

        padding = self.max_dim - tensor.shape[0]

        try:
            tensor_padded = pad(tensor, [0, padding, 0, padding])
        except:
            return None, None

        if overlap:
            tensor_padded.fill_diagonal_(1)

        trilled = tensor_padded[np.tril_indices(self.max_dim)]

        return tensor_padded, trilled

    def element_to_onehot(self, element: List[str]) -> NDArray:
        """Convert elements to numbers.

        Args:
            element (List[str]): Elements in a molecule.

        Returns:
            NDArray: Converted elements.
        """
        indices = np.array([self.elements.index(e) for e in element])
        onehot = np.eye(len(self.elements))[indices]
        return np.argmax(onehot, axis=1)

    def pad_edge(self, edge: NDArray) -> NDArray:
        """Pad the edge of graphs.

        Args:
            edge (NDArray): Edge of graph.

        Returns:
            NDArray: Padded edge.
        """
        padding = 3 - len(edge)
        edge = np.pad(edge, ((0, padding), (0, 0)), constant_values=0)

        return edge

    def remove_diagonal(self, row: NDArray, diagonal: NDArray) -> NDArray:
        """Delete the diagonal element from the row

        Args:
            row (NDArray): Row that needs to be deleted.
            diagonal (NDArray): Mask of what needs to be removed.

        Returns:
            NDArray: Cleaned row.
        """
        return np.delete(row, np.where(row == diagonal)[0])

    def extract_data_from_matrices(
        self,
        graph: GraphData,
        fock: torch.Tensor,
        overlap: torch.Tensor,
        bond_attributes,
    ) -> Tuple[List[NDArray], torch.Tensor, int]:
        """Extract padded node features, edge attributes and number of electrons.

        Args:
            graph (GraphData): Molecule represented as a graph.
            fock (torch.Tensor): Fock matrix.
            overlap (torch.Tensor): Overlap matrix.
            bond_attributes (_type_): Bond attributes.

        Returns:
            Tuple[List[NDArray], torch.Tensor, int]:
            Padded node features, edge attributes and number of electrons.
        """
        atom_to_orbitals = {0: 1, 1: 4, 2: 4, 3: 4, 4: 4}

        atom_to_electron = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}

        bond_to_orbitals = {0: 1, 1: 1, 2: 2, 3: 3, 12: 2}

        atoms = graph.x[:, 0].clone()

        diagonal_fock = fock.diagonal()
        diagonal_over = overlap.diagonal()

        nodes_fock = diagonal_fock.clone()

        position = 0
        n_electrons = 0
        nodes_fock_split = []

        for atom in atoms:
            n_electrons += atom_to_electron[int(atom)]

            nr_orbitals = atom_to_orbitals[int(atom)]
            new_position = position + nr_orbitals
            node = nodes_fock[position:new_position]
            position = new_position
            if not atom == 0:
                pass
            else:
                node = np.pad(node, (0, 3), constant_values=0)

            nodes_fock_split.append(node)

        edges_hamiltonian = np.apply_along_axis(
            self.remove_diagonal, 1, fock, diagonal_fock
        )

        edges_overlap = np.apply_along_axis(
            self.remove_diagonal, 1, overlap, diagonal_over
        )

        edges = [
            [[int(j), int(k)], int(bond_attributes[i])]
            for i, (j, k) in enumerate(graph.edge_index.T)
        ]

        edge_list = []

        position = 0
        for bond in edges:
            new_position = position + bond_to_orbitals[bond[1]]

            ham = self.pad_edge(edges_hamiltonian[position:new_position])

            over = self.pad_edge(edges_overlap[position:new_position])

            edge_attr = np.concatenate((ham, over), axis=0).T

            edge_list.append(edge_attr)

            position = new_position

        edge_list = torch.tensor(np.array(edge_list))

        return nodes_fock_split, edge_list, n_electrons

    def smiles2graph(self) -> None:
        """Transforms smiles to graph
        https://github.com/pckroon/pysmiles
        https://sullyfchen.medium.com/predicting-drug-solubility-with-deep-learning-b5e48ff61206
        https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac
        https://programtalk.com/vs4/python/BNN-UPC/ignnition/examples/QM9/generate_dataset.py/
        """

        generator = self.data.iterrows()
        end = len(self.data)

        del self.data

        save_location = "Data/datasets/"

        create_savefolder(os.path.join(save_location, "NN"))

        create_savefolder(os.path.join(save_location, "GNN"))

        create_savefolder(os.path.join(save_location, "GNN_MG"))

        create_savefolder(os.path.join(save_location, "GNN_MG_FO"))

        GNN_data = []
        GNN_MG_data = []
        GNN_MG_FO_data = []

        X_list = []
        Y_list = []
        smiles = []
        electron_list = []

        for index, data in tqdm(
            generator, total=end, mininterval=0.2, desc="Saving datasets"
        ):
            smile, xyz = data["SMILES"], data["Coordinates"]

            hamdftb_pad, hamdftb_pad_tril = self.pad_and_tril(data["Hamiltonian"])
            overdftb_pad, overdftb_pad_tril = self.pad_and_tril(
                data["Overlap"], overlap=True
            )
            hamg16_pad, hamg16_pad_tril = self.pad_and_tril(data["Hamiltonian_g16"])
            overg16_pad, overg16_pad_tril = self.pad_and_tril(
                data["Overlap_g16"], overlap=True
            )

            if None in [hamdftb_pad, overdftb_pad, hamg16_pad, overg16_pad]:
                continue

            n_orbitals = len(data["Hamiltonian"])

            NN_X = torch.concat((hamdftb_pad_tril, overdftb_pad_tril))

            coord = xyz[:, 1:].astype("float64")

            graph_MO = read_smiles(smile, explicit_hydrogen=True)

            graph_MO_FO = graph_MO.copy()

            graph = graph_MO.copy()

            feature = self.element_to_onehot(
                np.asarray(graph_MO.nodes(data="element"))[:, 1]
            )

            # Includes type of bond connects two nodes and is used as edge_attr
            bond_attributes = from_smiles(smile, with_hydrogen=True).edge_attr

            # GNN_MO
            nx.set_node_attributes(
                graph_MO,
                {
                    k: {
                        "x": [
                            feature[k],
                            coord[k][0],
                            coord[k][1],
                            coord[k][2],
                        ]
                    }
                    for k, d in dict(graph_MO.nodes(data=True)).items()
                },
            )

            graph_MO = from_networkx(graph_MO)

            del graph_MO["element"], graph_MO["aromatic"], graph_MO["charge"]

            # [:,0] describes what type of bond the edge is
            edge_attributes = bond_attributes[:, 0].clone()

            data_graph_MO = GraphData(
                x=graph_MO.x,
                edge_index=graph_MO.edge_index,
                edge_attr=edge_attributes,
            )

            num_nodes = graph_MO.num_nodes

            data_graph_MO.num_nodes = num_nodes

            # GNN
            nx.set_node_attributes(
                graph,
                {
                    k: {"x": [feature[k]]}
                    for k, d in dict(graph.nodes(data=True)).items()
                },
            )

            graph = from_networkx(graph)

            del graph["element"], graph["aromatic"], graph["charge"]

            data_graph = GraphData(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=edge_attributes,
            )

            num_nodes = graph.num_nodes

            data_graph.num_nodes = num_nodes

            # GNN_MG_FO
            (
                nodes_ham,
                edge_attributes,
                n_electrons,
            ) = self.extract_data_from_matrices(
                graph_MO, hamdftb_pad, overdftb_pad, edge_attributes
            )

            try:
                Y = find_eigenvalues_true(hamg16_pad_tril, overg16_pad_tril, n_orbitals)
            except:
                print(f"Molecule {smile} failed")
                continue

            Y.append(torch.concat((hamg16_pad_tril, overg16_pad_tril)))

            nx.set_node_attributes(
                graph_MO_FO,
                {
                    k: {
                        "x": [
                            feature[k],
                            coord[k][0],
                            coord[k][1],
                            coord[k][2],
                            nodes_ham[i][0],
                            nodes_ham[i][1],
                            nodes_ham[i][2],
                            nodes_ham[i][3],
                        ]
                    }
                    for i, (k, d) in enumerate(
                        dict(graph_MO_FO.nodes(data=True)).items()
                    )
                },
            )

            graph_MO_FO = from_networkx(graph_MO_FO)

            del graph_MO["element"], graph_MO["aromatic"], graph_MO["charge"]

            data_graph_MO_FO = GraphData(
                x=graph_MO_FO.x,
                edge_index=graph_MO_FO.edge_index,
                edge_attr=edge_attributes,
                y=Y,
            )

            data_graph_MO_FO.num_nodes = num_nodes

            smiles.append(smile)

            GNN_data.append(data_graph)

            GNN_MG_data.append(data_graph_MO)

            GNN_MG_FO_data.append(data_graph_MO_FO)

            X_list.append(NN_X)

            Y_list.append(Y)

            electron_list.append(n_electrons)

            if (index % 32 == 0 and index != 0) or index == end - 1:
                file_name = (
                    self.data_interval
                    + "_"
                    + str(index - 32)
                    + "-"
                    + str(index - 1)
                    + ".pkl"
                )

                NN = {
                    "SMILES": smiles,
                    "X": X_list,
                    "Y": Y_list,
                    "N_electrons": electron_list,
                }

                NN_data = pd.DataFrame(NN)

                NN_data.to_pickle(save_location + "NN/" + file_name)

                GNN = {
                    "SMILES": smiles,
                    "X": GNN_data,
                    "Y": Y_list,
                    "N_electrons": electron_list,
                }
                GNN_data = pd.DataFrame(GNN)
                GNN_data.to_pickle(save_location + "GNN/" + file_name)

                GNN_MG = {
                    "SMILES": smiles,
                    "X": GNN_MG_data,
                    "Y": Y_list,
                    "N_electrons": electron_list,
                }

                GNN_MG_data = pd.DataFrame(GNN_MG)

                GNN_MG_data.to_pickle(save_location + "GNN_MG/" + file_name)

                GNN_MG_FO = {
                    "SMILES": smiles,
                    "X": GNN_MG_FO_data,
                    "Y": Y_list,
                    "N_electrons": electron_list,
                }
                GNN_MG_FO_data = pd.DataFrame(GNN_MG_FO)
                GNN_MG_FO_data.to_pickle(save_location + "GNN_MG_FO/" + file_name)

                GNN_data = []
                GNN_MG_data = []
                GNN_MG_FO_data = []
                X_list = []
                Y_list = []
                smiles = []
                electron_list = []

    def load_dftb(self) -> None:
        """Load or transform DFTB data."""
        if os.path.exists("Data/dftb.pkl"):
            self.dftb_data = pd.read_pickle("Data/dftb.pkl")
        else:
            self.extract_data_dftb()

            self.dftb_data.to_pickle("Data/dftb.pkl")

    def create_dataset(self) -> None:
        """Create dataset from DFTB and DFT calculations."""
        self.data_location = os.getcwd() + "/Data/"
        self.data_location_dftb = self.data_location + "slurm_ready/"
        self.data_location_g16 = self.data_location + "G16_zip/"

        # Only elements in the dataset
        self.elements = ["H", "C", "N", "O", "F"]

        self.load_dftb()

        self.extract_data_g16()

        self.dftb_data.reset_index(drop=True, inplace=True)

        self.g16_data.reset_index(drop=True, inplace=True)

        self.data = pd.merge(self.g16_data, self.dftb_data, on=["SMILES"])

        del self.g16_data, self.dftb_data

        self.smiles2graph()


def main() -> None:
    """Run entire script."""
    datatransformer = DataTransformer()
    data_intervals = ["1-10000"]
    for i in range(10001, 100001, 10000):
        data_intervals.append(f"{i}-{i + 10000 - 1}")

    for i in data_intervals:
        datatransformer.data_interval = i
        datatransformer.create_dataset()
        gc.collect()


if __name__ == "__main__":
    main()
