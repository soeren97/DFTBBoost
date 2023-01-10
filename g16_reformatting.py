import numpy as np
import os
import pandas as pd


def extract_from_xyz(file_nr):

    file_name = os.listdir(data_location)[file_nr]

    folder_name = file_name[:-10] + str(file_nr + 1)

    if os.path.exists(os.getcwd() + "/Data/G16/" + folder_name + "/mol.com"):
        return

    file_location = data_location + file_name

    file = open(file_location)

    num = int(file.readlines()[0])

    file.close()

    xyz_file = np.genfromtxt(
        fname=file_location, skip_header=2, skip_footer=3, dtype="unicode"
    )

    xyz_file = xyz_file[:, :-1]

    xyz_file = np.char.replace(xyz_file, "*^", "e")

    title = np.genfromtxt(
        fname=file_location,
        comments=None,
        skip_header=num + 3,
        skip_footer=1,
        dtype="unicode",
    )[0]

    os.makedirs("Data/G16/" + folder_name, exist_ok=True)

    atom_list = []

    with open("Data/G16/" + folder_name + "/mol.com", "w+") as com_writer:
        com_writer.write("%NprocShared=4\n")
        com_writer.write("%Mem=8GB\n")
        com_writer.write("%chk=mol.chk\n")
        com_writer.write("# nosym PBEPBE/sto-3g\n")
        com_writer.write("# iop(5/33=3)\n")
        com_writer.write("# iop(3/33=4)\n\n")
        com_writer.write(title)
        com_writer.write("\n\n")
        com_writer.write("0  1\n")
        for atom in xyz_file:
            com_writer.write(
                "{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(
                    atom[0], float(atom[1]), float(atom[2]), float(atom[3])
                )
            )
            atom_list.append(atom[0])

        com_writer.write("\n\n")


data_location = os.getcwd() + "/Data/xyz_wrong_format/"

file_numbers = pd.DataFrame(range(0, 100001))

file_numbers[0].apply(extract_from_xyz)
