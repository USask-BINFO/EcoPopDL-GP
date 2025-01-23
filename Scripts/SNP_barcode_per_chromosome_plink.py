import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import dask.dataframe as dd
import logging
from tqdm import tqdm
import psutil
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_log.txt"),
        logging.StreamHandler()
    ]
)

# Color map and chromosome info definitions
chr_info = {
    '1': 45050000,
    '2': 36780000,
    '3': 37370000,
    '4': 36150000,
    '5': 30000000,
    '6': 31600000,
    '7': 30280000,
    '8': 28570000,
    '9': 30530000,
    '10': 23960000,
    '11': 30760000,
    '12': 27770000
}

color_map = {
    '00': [0, 0, 0],      # Black for major allele (homozygous)
    '11': [255, 0, 0],    # Red for minor allele 1 (homozygous)
    '22': [0, 255, 0],    # Green for minor allele 2 (homozygous)
    '33': [0, 0, 255],    # Blue for minor allele 3 (homozygous)
    '01': [0, 0, 128],    # Dark blue (heterozygous with major allele)
    '10': [0, 0, 128],
    '02': [0, 128, 0],
    '20': [0, 128, 0],
    '03': [128, 0, 0],
    '30': [128, 0, 0],
    '12': [255, 255, 0],  # Yellow (heterozygous minor alleles)
    '21': [255, 255, 0],
    '13': [255, 0, 255],  # Purple (heterozygous minor alleles)
    '31': [255, 0, 255],
    '23': [0, 255, 255],  # Cyan (heterozygous minor alleles)
    '32': [0, 255, 255],
    '-1': [255, 255, 255] # White for missing data
}

def encode_alleles_nucl(alleles):
    encoded_alleles = []
    for allele in alleles:
        if len(allele) != 2:
            logging.warning(f"Unexpected allele length: {allele}")
            encoded_alleles.append('-1')  # Use string '-1' for missing/invalid data
            continue
        encoding_map = {'A': '1', 'T': '2', 'C': '3', 'G': '4'}
        try:
            encoded_str = encoding_map[allele[0]] + encoding_map[allele[1]]  # Encode both characters
            encoded_alleles.append(encoded_str)
        except KeyError:
            logging.warning(f"Unknown character in allele: {allele}")
            encoded_alleles.append('-1')  # Use string '-1' for missing/invalid data
    return encoded_alleles


import pandas as pd
import dask.dataframe as dd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_plink_data(ped_file, map_file, chunk_size=10_000):
    """
    Load PLINK .ped and .map files, merge SNP, Chromosome, and Position data,
    and save the final dataframe with the required column order.
    """
    try:
        # Check if files exist
        if not os.path.exists(ped_file):
            raise FileNotFoundError(f"The .ped file {ped_file} does not exist.")
        if not os.path.exists(map_file):
            raise FileNotFoundError(f"The .map file {map_file} does not exist.")

        # Load .map file
        logging.info("Loading .map file...")
        map_df = pd.read_csv(map_file, sep="\t", header=None, names=['Chromosome', 'SNP', 'Genetic_Distance', 'Position'])
        logging.info(f"Loaded .map file with {len(map_df)} SNPs.")

        # Read .ped file in chunks
        genotype_columns_start = 6
        genotype_columns_end = 6 + 2 * len(map_df)

        logging.info("Loading .ped file in chunks...")
        sample_ids = []
        genotype_chunks = []
        with pd.read_csv(
            ped_file,
            delim_whitespace=True,
            header=None,
            usecols=[1] + list(range(genotype_columns_start, genotype_columns_end)),
            chunksize=chunk_size,
            dtype=str
        ) as reader:
            for i, chunk in enumerate(reader):
                sample_ids.extend(chunk.iloc[:, 0].values)
                genotype_chunks.append(chunk.iloc[:, 1:])
                logging.info(f"Processed chunk {i + 1} with {len(chunk)} rows.")

        logging.info("Concatenating chunks...")
        genotype_data = pd.concat(genotype_chunks, axis=0)
        genotype_data.index = sample_ids
        genotype_data.columns = map_df['SNP'].repeat(2).values  # Duplicate SNPs for pairs

        # Encode alleles row by row
        logging.info("Starting allele encoding...")
        encoded_genotypes = pd.DataFrame.from_dict(
            {idx: row for idx, row in encode_alleles_row_by_row(genotype_data)},
            orient='index'
        )

        # Merge SNP information (Chromosome, Position)
        logging.info("Merging SNP, Chromosome, and Position data...")
        encoded_genotypes = encoded_genotypes.T  # Transpose for merging
        encoded_genotypes['Chromosome'] = map_df['Chromosome'].values
        encoded_genotypes['Position'] = map_df['Position'].values
        encoded_genotypes.index = map_df['SNP'].values  # Set SNP as row names

        # Reorder columns: Sample columns, Chromosome, Position
        sample_columns = list(encoded_genotypes.columns[:-2])  # Exclude Chromosome and Position
        final_df = encoded_genotypes[sample_columns + ['Chromosome', 'Position']]

        # Save final dataframe to CSV
        final_output_path = "merged_genotype_data.csv"
        final_df.to_csv(final_output_path)
        logging.info(f"Final dataframe saved at {final_output_path}")

        return final_df, map_df, genotype_data

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
        raise
    except pd.errors.EmptyDataError:
        logging.error("The .ped file appears empty or is improperly formatted.", exc_info=True)
        raise
    except Exception as e:
        logging.error("An unexpected error occurred.", exc_info=True)
        raise



def encode_alleles_row_by_row(genotype_df):
    logging.info("Starting allele encoding row by row.")
    for idx, row in genotype_df.iterrows():
        encoded_row = []
        # Process columns in pairs
        for col1, col2 in zip(row[::2], row[1::2]):
            if pd.isna(col1) or pd.isna(col2):
                encoded_row.append("-1")  # Handle missing data
                continue
            allele_pair = f"{col1}{col2}"
            encoded_allele = encode_alleles_nucl([allele_pair])[0]
            encoded_row.append(encoded_allele)
        if len(row) // 2 != len(encoded_row):
            logging.warning(f"Row {idx}: SNP count mismatch. Expected: {len(row) // 2}, Encoded: {len(encoded_row)}")
        yield idx, encoded_row
    logging.info("Finished encoding alleles.")


def calculate_maf_per_snp(encoded_alleles_df):
    maf_per_snp = {}
    for index, row in (encoded_alleles_df.transpose()).iterrows():
        allele_counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
        for allele in row:
            allele_str = str(allele)
            for char in allele_str:
                if char == '1':
                    allele_counts['A'] += 1
                elif char == '2':
                    allele_counts['T'] += 1
                elif char == '3':
                    allele_counts['C'] += 1
                elif char == '4':
                    allele_counts['G'] += 1
        total_alleles = len(row) * 2
        allele_frequencies = {allele: count / total_alleles for allele, count in allele_counts.items()}
        sorted_frequencies = sorted(allele_frequencies.values())
        maf = next((freq for freq in sorted_frequencies if freq > 0), None)
        maf_per_snp[index] = maf
    return maf_per_snp

def filter_snps_by_maf_in_batches(encoded_alleles_df, maf_threshold=0, batch_size=1000):
    maf_filtered_batches = []
    for start in range(0, encoded_alleles_df.shape[0], batch_size):
        batch = encoded_alleles_df.iloc[start:start + batch_size]
        maf_per_snp = calculate_maf_per_snp(batch)
        filtered_snps = {snp: maf for snp, maf in maf_per_snp.items() if maf is not None and maf >= maf_threshold}
        filtered_df = batch.loc[batch.index.intersection(filtered_snps.keys())]
        maf_filtered_batches.append(filtered_df)
    return pd.concat(maf_filtered_batches, axis=0)

def major_minor_allele(encoded_alleles_df):
    logging.info("Starting major/minor allele encoding...")
    
    # Create a copy to avoid modifying the original DataFrame
    encoded_df = encoded_alleles_df.copy()
    total_rows = len(encoded_df)

    for row_idx, (index, row) in enumerate(encoded_df.iterrows(), start=1):
        logging.info(f"Processing row {row_idx}/{total_rows}...")
        
        allele_counts = {'1': 0, '2': 0, '3': 0, '4': 0}
        
        # Count alleles in the row
        for allele in row:
            allele_str = str(allele)
            if allele_str == '-1':
                continue
            for char in allele_str:
                if char in allele_counts:
                    allele_counts[char] += 1
        logging.debug(f"Row {row_idx}: Allele counts: {allele_counts}")
        
        total_alleles = sum(allele_counts.values())
        if total_alleles == 0:
            logging.warning(f"Row {row_idx}: No valid alleles found. Skipping row.")
            continue
        
        # Calculate allele frequencies
        allele_frequencies = {allele: count / total_alleles for allele, count in allele_counts.items()}
        logging.debug(f"Row {row_idx}: Allele frequencies: {allele_frequencies}")
        
        # Identify major and minor alleles
        sorted_alleles = sorted(allele_counts.items(), key=lambda item: item[1], reverse=True)
        major_allele = sorted_alleles[0][0] if sorted_alleles[0][1] > 0 else None
        minor_alleles = [allele for allele, count in sorted_alleles[1:] if count > 0]
        logging.debug(f"Row {row_idx}: Major allele: {major_allele}, Minor alleles: {minor_alleles}")
        
        # Create encoding map
        encoding_map = {major_allele: '0'}
        for i, allele in enumerate(minor_alleles):
            encoding_map[allele] = str(i + 1)
        logging.debug(f"Row {row_idx}: Encoding map: {encoding_map}")
        
        # Update encoded DataFrame
        for i, allele in enumerate(row):
            allele_str = str(allele)
            if allele_str == '-1':
                encoded_df.loc[index, encoded_df.columns[i]] = '-1'
            else:
                if len(allele_str) == 2:
                    encoded_str = encoding_map.get(allele_str[0], '9') + encoding_map.get(allele_str[1], '9')
                else:
                    encoded_str = '99'
                encoded_df.loc[index, encoded_df.columns[i]] = encoded_str
        
        # Log progress every 100 rows
        if row_idx % 100 == 0:
            logging.info(f"Processed {row_idx}/{total_rows} rows.")
    
    logging.info("Major/minor allele encoding complete.")
    return encoded_df


def save_sample_as_image(sample_data, sample_name, folder, height=1):
    """Save the sample data as an image in the specified folder with reduced height."""
    # Convert the sample_data to a 1D array
    image_data = np.array(sample_data)

    # Define color mapping: major allele (0) is black, minor alleles have distinct colors
    color_map = {
        '00': [0, 0, 0],      # Black for major allele (homozygous)
        '11': [255, 0, 0],    # Red for minor allele 1 (homozygous)
        '22': [0, 255, 0],    # Green for minor allele 2 (homozygous)
        '33': [0, 0, 255],    # Blue for minor allele 3 (homozygous)
        '01': [0, 0, 128],    # Dark blue (heterozygous with major allele)
        '10': [0, 0, 128],    # Dark blue (heterozygous with major allele)
        '02': [0, 128, 0],    # Dark green (heterozygous with major allele)
        '20': [0, 128, 0],    # Dark green (heterozygous with major allele)
        '03': [128, 0, 0],    # Dark red (heterozygous with major allele)
        '30': [128, 0, 0],    # Dark red (heterozygous with major allele)
        '12': [255, 255, 0],  # Yellow (heterozygous minor alleles)
        '21': [255, 255, 0],  # Yellow (heterozygous minor alleles)
        '13': [255, 0, 255],  # Purple (heterozygous minor alleles)
        '31': [255, 0, 255],  # Purple (heterozygous minor alleles)
        '23': [0, 255, 255],  # Cyan (heterozygous minor alleles)
        '32': [0, 255, 255],  # Cyan (heterozygous minor alleles)
        '-1': [255, 255, 255] # White for missing data
    }

    # Define the maximum allowed width
    max_width = 65535
    width = min(len(image_data), max_width)  # Limit the width to the maximum allowed size

    # Scale down the sample data if it exceeds the maximum width
    if len(image_data) > max_width:
        scaling_factor = len(image_data) / max_width
        indices = np.linspace(0, len(image_data) - 1, max_width).astype(int)
        image_data = image_data[indices]  # Downsample data to fit

    # Generate a color matrix for the sample data
    color_matrix = np.array([color_map.get(str(allele), [255, 255, 255]) for allele in image_data])
    color_matrix = color_matrix.reshape(1, -1, 3)  # Reshape to 1 row, X columns, 3 color channels (RGB)

    # Create the plot
    plt.figure(figsize=(width / 1000, height))  # Scale width for visualization
    plt.imshow(color_matrix, aspect='auto')
    plt.axis('off')  # No axis needed

    # Save the image
    image_path = os.path.join(folder, f'{sample_name}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    logging.info(f"Image saved for {sample_name} at {image_path}")

# Define a consistent color mapping for alleles
color_map = {
    '00': [0, 0, 0],      # Black for major allele (homozygous)
    '11': [255, 0, 0],    # Red for minor allele 1 (homozygous)
    '22': [0, 255, 0],    # Green for minor allele 2 (homozygous)
    '33': [0, 0, 255],    # Blue for minor allele 3 (homozygous)
    '01': [0, 0, 128],    # Dark blue (heterozygous with major allele)
    '10': [0, 0, 128],    
    '02': [0, 128, 0],    
    '20': [0, 128, 0],    
    '03': [128, 0, 0],    
    '30': [128, 0, 0],    
    '12': [255, 255, 0],  # Yellow (heterozygous minor alleles)
    '21': [255, 255, 0],  
    '13': [255, 0, 255],  # Purple (heterozygous minor alleles)
    '31': [255, 0, 255],  
    '23': [0, 255, 255],  # Cyan (heterozygous minor alleles)
    '32': [0, 255, 255],  
    '-1': [255, 255, 255] # White for missing data
}

def save_sample_as_image_chr(sample_data, sample_name, folder, chr_id_row, locus_row):
    """
    Generates separate images for each chromosome in a sample with SNPs displayed as vertical lines by locus.
    """
    # Iterate through the chromosomes in chr_info
    for chr_name, chr_len in chr_info.items():
        chromosome_id = str(chr_name)  # Ensure ID is string
        print(f"Processing chromosome {chromosome_id} for sample {sample_name}")

        # Filter SNPs for the current chromosome
        chr_snps = [
            (locus, allele)
            for locus, allele, chr_id in zip(locus_row, sample_data, chr_id_row)
            if str(chr_id) == chromosome_id
        ]

        # Check if SNPs exist for the current chromosome
        if chr_snps:
            print(f"Saving image for chromosome {chromosome_id} (sample {sample_name})")

            # Create output folder for the chromosome
            output_folder = os.path.join(folder, f'{sample_name}/chr_{chr_name}')
            os.makedirs(output_folder, exist_ok=True)

            # Call helper function to generate and save the image
            save_sample_as_image_for_chr(chr_snps, sample_name, output_folder, chr_name, chr_len)
        else:
            print(f"No SNPs found for chromosome {chromosome_id} (sample {sample_name})")


def save_sample_as_image_for_chr(chr_snps, sample_name, output_folder, chr_name, chr_len):
    """
    Save chromosome data as a horizontal strip with SNPs positioned as vertical lines by locus.
    """
    # Fixed height for all chromosome strips
    strip_height = 0.5

    # Set up plot dimensions
    fig, ax = plt.subplots(figsize=(chr_len / 1e7, strip_height))  # Fixed height, scaled width based on chromosome length

    # Draw chromosome as a white rectangle (strip)
    ax.add_patch(plt.Rectangle((0, 0), chr_len, strip_height, color='white', ec='black', lw=1))

    # Place each SNP as a vertical line inside the chromosome strip at the correct locus position
    for locus, allele_code in chr_snps:
        if locus > chr_len:
            print(f"SNP at locus {locus} is out of range for chromosome {chr_name} (length {chr_len})")
            continue  # Skip SNPs that are out of range

        color = np.array(color_map.get(str(allele_code), [255, 255, 255])) / 255  # Normalize color to [0,1]
        x_pos = (locus / chr_len) * chr_len  # Scale locus to chromosome length
        ax.plot([x_pos, x_pos], [0, strip_height], color=color, lw=1)  # Vertical line for SNP

    # Set plot limits and hide axes for a clean look
    ax.set_xlim(0, chr_len)
    ax.set_ylim(0, strip_height)
    ax.axis('off')  # Remove axes for a clean figure

    # Save the image without title or labels
    image_path = os.path.join(output_folder, f'{sample_name}_chr_{chr_name}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()



def process_and_transpose_snp(file_name, map_file_name):
    # Load the SNP markers from the .map file's second column
    try:
        map_df = pd.read_csv(map_file_name, sep="\t", header=None)
        snp_markers = set(map_df.iloc[:, 1])  # Assuming the second column contains SNP Marker names
    except FileNotFoundError:
        print("Map file not found. Please check the file path.")
        return

    # Read the SNP data file into a DataFrame, skipping the first row
    df = pd.read_csv(file_name, skiprows=[0])
    
    # Drop the first column (probeset_id or any metadata column)
    df = df.drop(df.columns[0], axis=1)

    # Ensure the target column ('WCC237') exists
    if 'WCC237' in df.columns:
        last_valid_column = df.columns.get_loc('WCC237') + 1
    else:
        print("Column 'WCC237' not found.")
        return

    # Extract relevant columns up to 'WCC237'
    df = df.iloc[:, :last_valid_column]

    # Filter rows to include only those matching SNP markers from the map file
    if 'SNP Marker' in df.columns:
        df = df[df['SNP Marker'].isin(snp_markers)]

        # Set 'SNP Marker' as the row index and transpose the DataFrame
        df_mod = df.set_index('SNP Marker')
        
        # Apply encoding to each row in the DataFrame
        encoded_df = df_mod.apply(lambda row: encode_alleles_nucl(row.values), axis=1)
        encoded_df = pd.DataFrame(encoded_df.tolist(), index=encoded_df.index)

        # Filter SNPs based on MAF
        # filtered_df = encoded_df[encoded_df.apply(lambda row: calculate_maf(row) >= maf_threshold, axis=1)]
        # print(f"Filtered SNPs based on MAF threshold of {maf_threshold}: {filtered_df.shape[0]} SNPs retained.")

        # Transpose the DataFrame and set the row names to sample names
        filtered_df = encoded_df.transpose()
        filtered_df.index = df.columns[1:]  # Assuming sample names are in the second row

    else:
        print("SNP Marker column not found. Please check column names.")
        return

    return filtered_df

def save_all_chromosomes_as_image(sample_data, sample_name, folder, chr_id_row, locus_row):
    """
    Save all chromosomes for a sample in a single image.
    """
    # Determine the width of the image based on the longest chromosome
    max_chr_len = max(chr_info.values())
    strip_height = 0.5  # Fixed height for each chromosome strip
    fig_height = strip_height * len(chr_info)  # Height for all chromosomes without padding

    # Set up plot dimensions
    fig, ax = plt.subplots(figsize=(max_chr_len / 1e7, fig_height))

    # Plot each chromosome from top (chromosome 1) to bottom (last chromosome)
    y_offset = 0
    for chr_name in sorted(chr_info.keys(), reverse=True):  # Reverse order to have chromosome 1 on top
        chr_len = chr_info[chr_name]
        chromosome_id = f'{chr_name}'  # Expected identifier format
        print(f"Processing chromosome {chromosome_id} for sample {sample_name}")

        # Select SNPs that belong to the current chromosome
        chr_snps = [(locus, allele) for locus, allele, chr_id in zip(locus_row, sample_data, chr_id_row) if chr_id == chromosome_id]

        # Draw the chromosome strip as a white rectangle
        ax.add_patch(plt.Rectangle((0, y_offset), chr_len, strip_height, color='white', ec='black', lw=1))

        # Plot each SNP as a vertical line within the chromosome strip
        for locus, allele_code in chr_snps:
            if locus > chr_len:
                print(f"SNP at locus {locus} is out of range for chromosome {chr_name} (length {chr_len})")
                continue  # Skip SNPs that are out of range

            color = np.array(color_map.get(str(allele_code), [255, 255, 255])) / 255  # Normalize color to [0,1]
            x_pos = (locus / chr_len) * chr_len  # Scale locus to chromosome length
            ax.plot([x_pos, x_pos], [y_offset, y_offset + strip_height], color=color, lw=1)  # Vertical line for SNP

        # Move y_offset down for the next chromosome without extra space
        y_offset += strip_height

    # Set plot limits and hide axes for a clean look
    ax.set_xlim(0, max_chr_len)
    ax.set_ylim(0, y_offset)
    ax.axis('off')  # Remove axes for a clean figure

    # Save the image without title or labels
    output_folder = os.path.join(folder, 'combined')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_path = os.path.join(output_folder, f'{sample_name}_all_chromosomes.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Combined image saved for {sample_name} at {image_path}")


def main():
    ped_file = "pruned_data_admixture_norm.ped"
    map_file = "pruned_data_admixture.map"

  
    # Load PLINK data
    major_minor_df_chr, map_df, genotype_df = load_plink_data(ped_file, map_file)
    logging.info(f"Initial SNP count from .map file: {len(map_df)}")

    # Encode alleles
    logging.info("Starting allele encoding...")
    encoded_genotypes = pd.DataFrame.from_dict(
        {idx: row for idx, row in encode_alleles_row_by_row(genotype_df)},
        orient='index'
    )
    logging.info(f"Allele encoding complete. Encoded genotype DataFrame shape: {encoded_genotypes.shape}")

    # Log unexpected columns during encoding
    if encoded_genotypes.shape[1] != len(map_df):
        logging.warning(f"Expected SNPs: {len(map_df)}, but encoded SNP count is: {encoded_genotypes.shape[1]}")

    # Process major/minor allele encoding
    major_minor_encoded = major_minor_allele(encoded_genotypes)
    logging.info(f"Major/minor allele encoding complete. DataFrame shape: {major_minor_encoded.shape}")

    # Save images based on MAF filtering
    output_folder = os.path.expanduser('images_AF')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if major_minor_encoded is not None:
        for sample_name, sample_data in major_minor_encoded.iterrows():
            save_sample_as_image(sample_data, sample_name, output_folder)
    

# Extract the last row (chr_id_row) for chromosome data
    # chr_id_row = major_minor_df_chr.iloc[-2]  # Assuming this is the chromosome data row
    # locus_row = major_minor_df_chr.iloc[-1]  # Assuming this is the locus data row

    # Add row names (SNPs) from `major_minor_df_chr` to `major_minor_encoded`
    logging.info("Adding SNP row names to major_minor_encoded...")
    major_minor_encoded_T = major_minor_encoded.T
    major_minor_encoded_T.index = major_minor_df_chr.index
    logging.info("Row names added to major_minor_encoded.")

    # Add Chromosome and Position columns from `map_df` to `major_minor_encoded`
    logging.info("Adding Chromosome and Position columns to major_minor_encoded...")
    major_minor_encoded_T['Chromosome'] = map_df['Chromosome'].values
    major_minor_encoded_T['Position'] = map_df['Position'].values
    logging.info("Chromosome and Position columns added to major_minor_encoded.")

    # Save the updated dataframe to verify
    major_minor_encoded_T.to_csv('major_minor_encoded_with_chr_pos.csv')
    logging.info("Saved updated major_minor_encoded dataframe with Chromosome and Position columns.")


    output_folder_chr = os.path.expanduser('images_AF_chr')
    if not os.path.exists(output_folder_chr):
        os.makedirs(output_folder_chr)

    if major_minor_encoded_T is not None:
        # Iterate over each sample row (excluding the chr_id_row)
        for sample_name in major_minor_encoded_T.columns[:-2]:  # Exclude Chromosome and Position columns
            sample_data =  major_minor_encoded_T[sample_name]  # Get the sample data (column values)

            # Call the save_sample_as_image_chr function for each sample
            save_sample_as_image_chr(
                sample_data=sample_data,
                sample_name=sample_name,
                folder=output_folder_chr,
                chr_id_row= major_minor_encoded_T['Chromosome'],
                locus_row= major_minor_encoded_T['Position']
            )

    output_folder = os.path.expanduser('images_AF_combined')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for sample_name, sample_data in major_minor_encoded_T.columns[:-2]:
        sample_data =  major_minor_encoded_T[sample_name]  # Get the sample data (column values)

        # Call the save_sample_as_image_chr function for each sample
        save_all_chromosomes_as_image(
            sample_data=sample_data,
            sample_name=sample_name,
            folder=output_folder_chr,
            chr_id_row= major_minor_encoded_T['Chromosome'],
            locus_row= major_minor_encoded_T['Position']
        )

if __name__ == "__main__":
    main()
