import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

# List of notebook file paths to merge
notebooks = [
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/0_instacart_pre_processing.ipynb',
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/1_instacart_data_profiling.ipynb',
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/1_instacart_data_profiling USER.ipynb',    
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/2_instacart_clustering.ipynb',
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/2_instacart_clustering_keep_outliers.ipynb',
    r'C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/3_instacart_pattern_analysis.ipynb'
]

# Create a new notebook
merged_notebook = new_notebook()

# Iterate through each notebook and append its cells to the merged notebook
for notebook_path in notebooks:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        # Add a markdown cell to indicate the start of a new notebook
        merged_notebook.cells.append(new_markdown_cell(f'# Merged from {notebook_path}'))
        merged_notebook.cells.extend(nb.cells)

# Save the merged notebook
with open('C:/Data/Coding/Python/ist_dash_2024_rec/non_supervised_analysis/notebooks/dataset_3/G02_notebook_dataset3.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(merged_notebook, f)

print('Notebooks merged successfully into merged_notebook.ipynb')