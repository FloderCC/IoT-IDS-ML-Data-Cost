"""File description:

Script to generate the readme file from the results stored in CSV files.
"""
import pandas as pd
def csv_to_markdown_table(csv_file_path):
    df = pd.read_csv(csv_file_path)
    markdown_table = df.to_markdown(index=False)
    return markdown_table


# create the Readme file
readme_file = open(f'results/README.md', 'w')
readme_file.write('# All results\n')

readme_file.write('## Top score summary.\n')
readme_file.write('This table contains the maximum value (MAX) of MCC reached for each model-dataset pair. It also includes the values for which 98.5% (CP) of the MCC was reached.\n\n')
readme_file.write(csv_to_markdown_table('results/top scores summary.csv')+'\n')

readme_file.write('## Spearman Coefficients\n')
readme_file.write('This table contains all Spearman\'s correlations (SRCC) computed by each model-dataset pair.\n\n')
readme_file.write(csv_to_markdown_table('results/spearman coefficients.csv')+'\n')


readme_file.write('## Raw results of the experiment\n')
readme_file.write(csv_to_markdown_table('results/results_by_rows_and_features.csv')+'\n')

readme_file.close()
