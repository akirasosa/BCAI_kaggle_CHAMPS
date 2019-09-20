from sklearn.preprocessing import LabelEncoder
import pandas as pd

from proj import const


def get_scc_type_encoder():
    enc = LabelEncoder()
    enc.fit(const.TYPES)
    return enc


def get_atom_encoder():
    enc = LabelEncoder()
    enc.fit(const.ATOMS)
    return enc


def get_all_df():
    df_train = pd.read_csv(const.DATA_DIR / 'input' / 'train.csv')
    df_test = pd.read_csv(const.DATA_DIR / 'input' / 'test.csv')
    df = pd.concat((df_train, df_test), sort=True, ignore_index=True)

    type_encoder = get_scc_type_encoder()
    df['type_encoded'] = type_encoder.transform(df.type) + 1

    return df


def get_structures_df():
    df = pd.read_csv(const.DATA_DIR / 'input' / 'structures.csv')

    atom_encoder = get_atom_encoder()
    df['atom_encoded'] = atom_encoder.transform(df.atom) + 1

    return df
