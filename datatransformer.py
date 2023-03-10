import numpy as np
import pandas as pd
from tqdm import tqdm

import logging
import os
import shutil
import gc

import torch
from torch.nn.functional import pad
from torch_geometric.utils import from_networkx, from_smiles, sort_edge_index
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch

from pysmiles import read_smiles
import networkx as nx

import utils

from typing import Tuple, List, Union
from numpy.typing import NDArray

logging.getLogger("pysmiles").setLevel(logging.CRITICAL)  # Anything higher than warning


class DataTransformer:
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
            data_list, columns=["SMILES", "Coordinates", "Hamiltonian", "Overlap"]
        )
        self.dftb_data.dropna(inplace=True)
        gc.collect()

    def extract_data_g16(self) -> None:
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
                try:
                    hamiltonian = []
                    with open(root + "/hamiltonian.1") as f:
                        matrix_line = []
                        for idx, line in enumerate(f.readlines()):
                            row_arr = line.split()
                            matrix_line.append(row_arr)

                            test_length = np.concatenate(matrix_line)
                            if test_length.shape[0] == nbas:
                                temp = [float(x.replace("D", "e")) for x in test_length]
                                hamiltonian.append(temp)
                                matrix_line = []
                    hamiltonian = np.array(hamiltonian)

                except:
                    hamiltonian = None

                try:
                    overlap = []
                    with open(root + "/overlap") as f:
                        matrix_line = []
                        for idx, line in enumerate(f.readlines()):
                            row_arr = line.split()
                            matrix_line.append(row_arr)

                            test_length = np.concatenate(matrix_line)
                            if test_length.shape[0] == nbas:
                                temp = [float(x.replace("D", "e")) for x in test_length]
                                overlap.append(temp)
                                matrix_line = []
                    overlap = np.array(overlap)

                except:
                    overlap = None

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
        self, array: NDArray
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        tensor = torch.from_numpy(array)

        padding = self.max_dim - tensor.shape[0]

        try:
            tensor_padded = pad(tensor, [0, padding, 0, padding])
        except:
            return None, None

        tensor = tensor_padded[np.tril_indices(self.max_dim)]

        return tensor, tensor_padded

    def element_to_onehot(self, element: List[str]):
        out = []
        for i in range(0, len(element)):
            v = np.zeros(len(self.elements))
            v[self.elements.index(element[i])] = 1.0
            out.append(v.argmax())
        return np.asarray(out)

    def pad_edge(self, edge: NDArray) -> NDArray:
        padding = 3 - len(edge)
        edge = np.pad(edge, ((0, padding), (0, 0)), constant_values=0)

        return edge

    def remove_diagonal(self, row: NDArray, diagonal: NDArray) -> NDArray:
        """Delete the diagonal element from the row

        Args:
            row (_type_): _description_
            diagonal (_type_): _description_

        Returns:
            np.array: _description_
        """
        return np.delete(row, np.where(row == diagonal)[0])

    def extract_data_from_matrices(
        self,
        graph: GraphData,
        hamiltonian: torch.Tensor,
        overlap: torch.Tensor,
        bond_attributes,
    ) -> Tuple[List[NDArray], torch.Tensor, int]:

        atom_to_orbitals = {0: 1, 1: 4, 2: 4, 3: 4, 4: 4}

        atom_to_electron = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}

        bond_to_orbitals = {0: 1, 1: 1, 2: 2, 3: 3, 12: 2}

        atoms = graph.x[:, 0].clone()

        diagonal_ham = hamiltonian.diagonal()
        diagonal_over = overlap.diagonal()

        nodes_ham = diagonal_ham.clone()

        position = 0
        n_electrons = 0
        nodes_ham_split = []

        for atom in atoms:
            n_electrons += atom_to_electron[int(atom)]

            nr_orbitals = atom_to_orbitals[int(atom)]
            new_position = position + nr_orbitals
            node = nodes_ham[position:new_position]
            position = new_position
            if not atom == 0:
                pass
            else:
                node = np.pad(node, (0, 3), constant_values=0)

            nodes_ham_split.append(node)

        edges_hamiltonian = np.apply_along_axis(
            self.remove_diagonal, 1, hamiltonian, diagonal_ham
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

        return nodes_ham_split, edge_list, n_electrons

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

            hamiltonian_dftb, hamdftb_pad = self.pad_and_tril(data["Hamiltonian"])
            overlap_dftb, overdftb_pad = self.pad_and_tril(data["Overlap"])
            hamiltonian_g16, hamg16_pad = self.pad_and_tril(data["Hamiltonian_g16"])
            overlap_g16, overg16_pad = self.pad_and_tril(data["Overlap_g16"])

            if None in [hamiltonian_dftb, overlap_dftb, hamiltonian_g16, overlap_g16]:
                continue

            NN_X = torch.concat((hamiltonian_dftb, overlap_dftb))

            coord = xyz[:, 1:].astype("float64")

            graph_MO = read_smiles(smile, explicit_hydrogen=True)

            graph_MO_FO = graph_MO.copy()

            graph = graph_MO.copy()

            feature = self.element_to_onehot(
                np.asarray(graph_MO.nodes(data="element"))[:, 1]
            )

            # This function includes what type of bond connects two nodes and is used as edge_attr
            bond_attributes = from_smiles(smile, with_hydrogen=True).edge_attr

            # GNN_MO
            nx.set_node_attributes(
                graph_MO,
                {
                    k: {"x": [feature[k], coord[k][0], coord[k][1], coord[k][2]]}
                    for k, d in dict(graph_MO.nodes(data=True)).items()
                },
            )

            graph_MO = from_networkx(graph_MO)

            del graph_MO["element"], graph_MO["aromatic"], graph_MO["charge"]

            # [:,0] describes what type of bond the edge is, ie a single, double or triple bond
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
            nodes_ham, edge_attributes, n_electrons = self.extract_data_from_matrices(
                graph_MO, hamdftb_pad, overdftb_pad, edge_attributes
            )

            try:
                Y = utils.find_eigenvalues_true(
                    data["Hamiltonian_g16"], data["Overlap_g16"], n_electrons
                )
            except:
                print(f"Molecule {smile} failed")
                continue

            Y.append(torch.concat((hamiltonian_g16, overlap_g16)))

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
        if os.path.exists("Data/dftb.pkl"):
            self.dftb_data = pd.read_pickle("Data/dftb.pkl")
        else:
            self.extract_data_dftb()

            self.dftb_data.to_pickle("Data/dftb.pkl")

    def load_g16(self) -> None:
        self.extract_data_g16()

    def create_dataset(self) -> None:
        self.data_location = os.getcwd() + "/Data/"
        self.data_location_dftb = self.data_location + "slurm_ready/"
        self.data_location_g16 = self.data_location + "G16_zip/"

        # Only elements in the dataset
        self.elements = ["H", "C", "N", "O", "F"]

        self.load_dftb()

        self.load_g16()

        self.dftb_data.reset_index(drop=True, inplace=True)

        self.g16_data.reset_index(drop=True, inplace=True)

        self.data = pd.merge(self.g16_data, self.dftb_data, on=["SMILES"])

        del self.g16_data, self.dftb_data

        self.smiles2graph()


def main() -> None:
    datatransformer = DataTransformer()
    data_intervals = ["0-10000"]
    for i in range(10001, 100001, 10000):
        data_intervals.append(f"{i}-{i + 10000 - 1}")

    for i in data_intervals:
        datatransformer.data_interval = i
        datatransformer.create_dataset()
        gc.collect()


if __name__ == "__main__":
    main()
