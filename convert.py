import pandas as pd
from tqdm import tqdm
from architector.io_obabel import get_obmol_smiles, get_OBMol_coords_anums_graph
import architector.io_ptable as io_ptable
from architector import convert_io_molecule, view_structures
from architector.io_obabel import generate_obmol_conformers
from executorlib import Executor
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv('AqSolDB_v1.0_min.csv')
    f1 = (df.SMILES.str.count('\.') < 1)
    f2 = pd.Series([True]*df.shape[0])
    natoms = pd.Series([0.0]*df.shape[0])
    for i,row in tqdm(df.iterrows(), total=df.shape[0]):
        obmol = get_obmol_smiles(row['SMILES'], build=False)
        _ , syms, _  = get_OBMol_coords_anums_graph(obmol, get_types=True)
        natoms.loc[i] = len(syms)
        mets = [x for x in syms if x in io_ptable.all_metals]
        if len(mets) > 0:
            f2[i] = False

    df['natoms'] = natoms
    f3 = df.natoms < 400
    fdf = df[(f2) & (f1) & (f3)]
    print(fdf.shape)
    fdf.to_csv('filtered_aq_soldb.csv')
