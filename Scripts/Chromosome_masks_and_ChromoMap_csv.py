

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import allel
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def encode_alleles_nucl(alleles):
    # if A we code it as 1, if T we code it as 2, if C we code it as 3, if G we code it as 4.
    # depending on homozygous or heterozygous we will have a two-digit code
    encoded_alleles = []
    for allele in alleles:
        encoded_str = ""
        unknown_character = False
        for char in allele:
            if char == 'A':
                encoded_str += '1'
            elif char == 'T':
                encoded_str += '2'
            elif char == 'C':
                encoded_str += '3'
            elif char == 'G':
                encoded_str += '4'
            else:
                unknown_character = True
                break  # Exit the loop if an unknown character is found
        if unknown_character:
            encoded_alleles.append(-1)
        else:
            encoded_alleles.append(int(encoded_str))
    return encoded_alleles

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

def calculate_maf_per_snp(encoded_alleles_df):
    """Calculate the Minor Allele Frequency (MAF) for each SNP in encoded SNP data."""
    maf_per_snp = {}
    
    # Iterate over each row in the DataFrame
    for index, row in (encoded_alleles_df.transpose()).iterrows():
        # Initialize counts for each nucleotide
        allele_counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
        
        for allele in row:
            # Convert the allele to string and iterate over each character
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
        
        # Calculate the total number of alleles for the row
        total_alleles = len(row) * 2

        
        # Calculate the frequency of each allele for the row
        allele_frequencies = {allele: count / total_alleles for allele, count in allele_counts.items()}
        # Identify the minor allele frequency (MAF). If the smallest value is 0, find the next smallest non-zero value
        sorted_frequencies = sorted(allele_frequencies.values())
        
        maf = None
        for freq in sorted_frequencies:
            if freq > 0:
                maf = freq
                break
        
        # Store the MAF for the current SNP
        maf_per_snp[index] = maf
 
    return maf_per_snp

def filter_snps_by_maf(encoded_alleles_df, maf_threshold=0):
    """Filter SNPs based on Minor Allele Frequency (MAF) threshold."""
    maf_per_snp = calculate_maf_per_snp(encoded_alleles_df)
    
    
    # Filter SNPs based on MAF threshold
    filtered_snps = {snp: maf for snp, maf in maf_per_snp.items() if maf is not None and maf >= maf_threshold}
    
    encoded_alleles_df = encoded_alleles_df.transpose()
    # Drop rows that do not meet the MAF threshold
    filtered_df = encoded_alleles_df.loc[encoded_alleles_df.index.intersection(filtered_snps.keys())]
    
    return filtered_df

def major_minor_allele(encoded_alleles_df):
    encoded_df = encoded_alleles_df.copy()
    
    # Iterate over each row in the DataFrame
    for index, row in encoded_df.iterrows():
        # Initialize counts for each nucleotide
        allele_counts = {'1': 0, '2': 0, '3': 0, '4': 0}
        
        for allele in row:
            # Convert the allele to string and iterate over each character
            allele_str = str(allele)
            if allele_str == '-1':
                continue  # Skip missing data
            for char in allele_str:
                if char in allele_counts:
                    allele_counts[char] += 1
        
        # Calculate the total number of alleles for the row
        total_alleles = sum(allele_counts.values())
        
        # Calculate the frequency of each allele for the row
        allele_frequencies = {allele: count / total_alleles for allele, count in allele_counts.items()}
        
        # Identify the major and minor alleles
        sorted_alleles = sorted(allele_counts.items(), key=lambda item: item[1], reverse=True)
        
        major_allele = sorted_alleles[0][0] if sorted_alleles[0][1] > 0 else None
        minor_alleles = [allele for allele, count in sorted_alleles[1:] if count > 0]
        
        # Create a mapping for encoding
        encoding_map = {major_allele: '0'}
        for i, allele in enumerate(minor_alleles):
            encoding_map[allele] = str(i + 1)
        
        # Update the row with the new encoded values
        for i, allele in enumerate(row):
            allele_str = str(allele)
            if allele_str == '-1':
                encoded_df.loc[index, encoded_df.columns[i]] = '-1'  # Keep missing data as '-1'
            else:
                # Encode the entire allele as a two-digit string
                if len(allele_str) == 2:
                    encoded_str = encoding_map.get(allele_str[0], '9') + encoding_map.get(allele_str[1], '9')
                else:
                    encoded_str = '99'  # Use '99' for unknown characters
                encoded_df.loc[index, encoded_df.columns[i]] = encoded_str
    
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

    # Generate a color matrix for the sample data
    color_matrix = np.array([color_map.get(str(allele), [255, 255, 255]) for allele in image_data])

    # Reshape the color matrix for visualization
    color_matrix = color_matrix.reshape(1, -1, 3)  # Reshape to 1 row, X columns, 3 color channels (RGB)

    # Create the plot with reduced height
    plt.figure(figsize=(len(sample_data) / 100, height))  # Adjust width and height
    plt.imshow(color_matrix, aspect='auto')
    plt.axis('off')  # No axis needed

    # Save the image
    image_path = os.path.join(folder, f'{sample_name}.png')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # print(f"Image saved for {sample_name} at {image_path}")
    

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

# Chromosome lengths (example values in base pairs)
chr_info = {
    '1': 48360000,
    '2': 36630000,
    '3': 39990000,
    '4': 49190000,
    '5': 48170000,
    '6': 59460000,
    '7': 48960000,
    '8': 16480000
}

def save_sample_as_image_chr(sample_data, sample_name, folder, chr_id_row, locus_row):
    """
    Generates separate images for each chromosome in a sample with SNPs displayed as vertical lines by locus.
    """
    for chr_name, chr_len in chr_info.items():
        chromosome_id = f'Ca{chr_name}'  # Expected identifier format
        print(f"Processing chromosome {chromosome_id} for sample {sample_name}")

        # Select SNPs that belong to the current chromosome
        chr_snps = [(locus, allele) for locus, allele, chr_id in zip(locus_row, sample_data, chr_id_row) if chr_id == chromosome_id]

        if chr_snps:
            print(f"Saving image for {chromosome_id}")
            output_folder = os.path.join(folder, f'{sample_name}/chr_{chr_name}')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            save_sample_as_image_for_chr(chr_snps, sample_name, output_folder, chr_name, chr_len)
        else:
            print(f"No SNPs found for chromosome {chromosome_id}")

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

def process_and_add_chr_id_locus(file_name, major_minor_df_T):
    """
    Reads SNP data, adds Chr_id and loci, and merges with major_minor_df_T.
    """
    # Read SNP data file
    original_df = pd.read_csv(file_name, skiprows=[0])
    original_df = original_df.drop(original_df.columns[0], axis=1)  # Drop metadata column

    # Ensure required columns exist
    if 'SNP Marker' not in original_df.columns or 'Chr_id' not in original_df.columns:
        print("Columns 'SNP Marker' or 'Chr_id' not found in the original file.")
        return major_minor_df_T

    # Filter to retain only 'SNP Marker', 'Chr_id', and 'Start' columns
    snp_chr_df = original_df[['SNP Marker', 'Chr_id', 'Start']]
    merged_df = major_minor_df_T.merge(snp_chr_df, left_index=True, right_on='SNP Marker', how='inner')
    merged_df.set_index('SNP Marker', inplace=True)
    
    return merged_df

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


# Define color mapping for alleles
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

# Chromosome lengths (example values in base pairs)
chr_info = {
    '1': 48360000,
    '2': 36630000,
    '3': 39990000,
    '4': 49190000,
    '5': 48170000,
    '6': 59460000,
    '7': 48960000,
    '8': 16480000
}

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
        chromosome_id = f'Ca{chr_name}'  # Expected identifier format
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
    
def save_chromosome_masks(sample_data, sample_name, folder, chr_id_row, locus_row):
    """
    Generate and save masks for each chromosome in a sample image.
    """
    max_chr_len = max(chr_info.values())
    strip_height = 0.5  # Fixed height for each chromosome strip
    mask_height = int(strip_height * len(chr_info) * 100)  # Scale height for resolution
    mask_width = int(max_chr_len / 1e7 * 100)  # Scale width for resolution

    # Initialize an empty mask with background label 0
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    # Debugging: Print chromosome order
    print(f"Generating mask for sample {sample_name}")
    print(f"Chromosome order: {sorted(chr_info.keys(), reverse=True)}")

    # Reverse the chromosome order for masks (bottom-up instead of top-down)
    y_offset = len(chr_info) * strip_height  # Start from the bottom
    for chr_name in sorted(chr_info.keys(), reverse=True):  # Reverse order for chromosome 1 at the top
        chr_len = chr_info[chr_name]
        chromosome_id = f'Ca{chr_name}'  # Expected identifier format

        print(f"Processing mask for chromosome {chromosome_id}")

        # Update mask for this chromosome
        y_start = int((y_offset - strip_height) * 100)  # Start from bottom and move up
        y_end = int(y_offset * 100)

        # Scale chromosome length to mask width
        chr_width = int(chr_len / max_chr_len * mask_width)

        # Assign the chromosome ID to the mask region
        mask[y_start:y_end, :chr_width] = int(chr_name)

        # Move y_offset up for the next chromosome
        y_offset -= strip_height

    # Save the mask as an image
    mask_folder = os.path.join(folder, 'masks')
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    mask_path = os.path.join(mask_folder, f'{sample_name}_mask.png')
    plt.imsave(mask_path, mask, cmap='gray')
    print(f"Mask saved for {sample_name} at {mask_path}")



def main():
    # Path to SNP file
    file_name = os.path.expanduser('~/Downloads/WCC_SNP_file.csv')
    pruned_data_admixture = 'pruned_data_admixture.map'

    # Process SNP data and transpose
    encoded_df = process_and_transpose_snp(file_name, pruned_data_admixture)
    filtered_snps = filter_snps_by_maf(encoded_df, maf_threshold=0)
    major_minor_df = major_minor_allele(filtered_snps)
    major_minor_df_chr = process_and_add_chr_id_locus(file_name, major_minor_df)
    major_minor_df_chr_T = major_minor_df_chr.transpose()

    chr_id_row = major_minor_df_chr_T.iloc[-2]  # Chromosome ID row
    locus_row = major_minor_df_chr_T.iloc[-1]   # Locus row

    # Exclude last two rows for SNP data
    sample_data_df = major_minor_df_chr_T.iloc[:-2]

    output_folder = os.path.expanduser('images_AF_combined')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each sample
    for sample_name, sample_data in sample_data_df.iterrows():
        save_all_chromosomes_as_image(sample_data, sample_name, output_folder, chr_id_row, locus_row)
        save_chromosome_masks(sample_data, sample_name, output_folder, chr_id_row, locus_row)

if __name__ == '__main__':
    main()
