import pandas as pd

# Load the CSV file
df = pd.read_csv('cv_validated.tsv', delimiter='\t')

# Filter out rows where 'gender' or 'accent' is NaN
filtered_df = df.dropna(subset=['gender', 'accent'])
filtered_df = filtered_df[(filtered_df['gender'] != 'other') & (filtered_df['accent'] != 'other')]

# Group by 'gender' and 'accent'
grouped = filtered_df.groupby(['gender', 'accent'])


def get_top_client_id_and_count(group):
    top_client_id = group['client_id'].value_counts().idxmax()
    count = group['client_id'].value_counts().max()
    return top_client_id, count


top_client_ids_and_counts = grouped.apply(get_top_client_id_and_count)


output_lines = []

print("Top client_id and count for each gender and accent combination:\n\n\n")
for (gender, accent), (client_id, count) in top_client_ids_and_counts.items():
    print(f"Gender: {gender}, Accent: {accent}, Count: {count}")
    associated_line = \
        grouped.get_group((gender, accent))[grouped.get_group((gender, accent))['client_id'] == client_id].iloc[0]

    print(f"Associated line for '{gender} {accent}' combination:")
    print(associated_line)
    print("\n")

    output_lines.append(accent)
    output_lines.append(gender)
    output_lines.append(client_id)
    output_lines.append(str(count))

output_file = 'top_client_id.txt'
with open(output_file, 'w') as file:
    for line in output_lines:
        file.write(line + "\n")

print(f"Output written to {output_file}")
