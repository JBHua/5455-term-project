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
search_string = '9ec24631628c749a38d8747ba8607deecd412924d7dbaeee16a52215295e742252ec3f953f3c98c005fa09343c6f3ed72a8fdb4d8bea9571b6acb9200d6c3b45'
for found_rows in search_string_in_tsv(file_path, search_string):
    print(found_rows)

# Reference for US Male
# a lemon tea could be great to drink this summer
# 9cb53a64c5a239520c6888ec48e67ca7963bcb5fc4a2673645c0bdaf58a89649be88bb44935a1df354c4648ddd78d88142c8e32694ca5c2baf0e41f0c2ceb4fd

# Refernce for Indian Female
# Are you still using your computer for the research?
# b0ad186dd1d0187426a1b546db6ea0a43b11dd9957a4579016f528850af60b98c8c7f8761be1e418c56609897972b13892c33f3a0b74420d5f745dbca1b7b000

# Refernce for England Male
# When the judge spoke the death sentence, the defendant showed no emotion
# 84e17ef32894333ef219a150cecd3ed9393dc7457f44cff98b705366659081c5d3dfb51719db88ce7ef065d609d375f600e124c1ffdb500c8d410ee70b3991dc

# Refernce for England Female
# Thousands of years ago, this town was the center of an ancient civilisation
# 1144a255bc3728ee1ad447b2c5daa8c360951de987676ea3ec54c9365f6a2ac105e5b7456062ca7918e59aa4b6351c5185d4c0645a00f7e0f324ab01a8af3a4e

# Refernce for HongKong Male
# The boy was cuddling with his fluffy teddy bear
# 70c4bca75aef11e142f55b3ad8ab5932cf1985ec94151a3f7ab25946752df57c2ba38b7fb0904278d854efbdc8ca9da8e692d9be20bc36ae8dd111ee915d9ad4

# Reference for Philippines Male
# It seems that the elderly are having difficulties in using the Internet
# f557d16817532d17167053abf86b9901daeb4c29042f4d37fa42ec57fe8f94babb00b99a0c3d9de47e6f4a049152b3121e6782a41bb9a00411202a0862479380
