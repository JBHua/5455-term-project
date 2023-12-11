import pandas as pd

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_colwidth', None)  # Show full width of columns
pd.set_option('display.width', None)  # Auto-detect the width of the terminal

def search_string_in_tsv(file_path, search_string, chunk_size=10000, max_results=1000000):
    result_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, delimiter='\t'):
        filtered_chunk = chunk[chunk.apply(lambda row: row.astype(str).str.contains(search_string).any(), axis=1)]
        if not filtered_chunk.empty:
            for _, row in filtered_chunk.iterrows():
                # Yield or print only the 'sentence' and 'path' columns
                yield row[['sentence', 'path']]
                result_count += 1
                if result_count >= max_results:
                    return

# Usage
file_path = 'other.tsv'
search_string = '51dff465f517584a4cb3f174052968195d18a0a94c67d6b5ac8badbc5d01466c2bf9bb0c42a2c3fea68e889b33e208be8c182beadc19da897da164bc8cf21e1b'
for found_rows in search_string_in_tsv(file_path, search_string):
    print(found_rows)

# Reference for US Male
# a lemon tea could be great to drink this summer
# 9cb53a64c5a239520c6888ec48e67ca7963bcb5fc4a2673645c0bdaf58a89649be88bb44935a1df354c4648ddd78d88142c8e32694ca5c2baf0e41f0c2ceb4fd

# Refernce for Indian Female
# Are you still using your computer for the research?
# b0ad186dd1d0187426a1b546db6ea0a43b11dd9957a4579016f528850af60b98c8c7f8761be1e418c56609897972b13892c33f3a0b74420d5f745dbca1b7b000