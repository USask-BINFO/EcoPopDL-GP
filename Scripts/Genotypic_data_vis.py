# STEP I

## Genotypic data visualization

# i. SNP % across chromosomes
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

def read_csv_file(file_name):
    try:
        data = pd.read_csv(file_name, encoding='latin1')
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def snp_count_chr(data):
    if data is not None:
        # Count the SNPs for each chromosome
        snp_counts = data['Chr_id'].value_counts().sort_index()
        
        # Calculate the percentage of SNPs for each chromosome
        total_snps = snp_counts.sum()
        snp_percentage = (snp_counts / total_snps) * 100
        
        # Plot the SNP percentage distribution
        plt.figure(figsize=(10, 6))
        ax = snp_percentage.plot(kind='bar', color='green')
        plt.xlabel('Chromosome')
        plt.ylabel('Percentage of SNPs (%)')
        plt.title('SNP Percentage Distribution Across Chromosomes')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        # Annotate the bars with the actual SNP counts
        for p, count in zip(ax.patches, snp_counts):
            ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom')
        
        plt.show()
    else:
        print("No data to plot.")


# ii. SNP count distribution across chromsomes

def read_csv_file(file_name):
    try:
        data = pd.read_csv(file_name, encoding='latin1')
        return data
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def snp_position_distribution(data, bin_size=1000000):
    if data is not None:
        # Ensure 'Chr_id' and 'Start' columns are present
        if 'Chr_id' not in data.columns or 'Start' not in data.columns:
            print("Missing required columns 'Chr_id' or 'Start'.")
            return
        
        # Chromosome lengths (example values, replace with actual lengths)
        chr_lengths = {
            'Ca1': 48360000,
            'Ca2': 36630000,
            'Ca3': 39990000,
            'Ca4': 49190000,
            'Ca5': 48170000,
            'Ca6': 59460000,
            'Ca7': 48960000,
            'Ca8': 16480000
        }
        
        # Create bins for SNP positions
        data['Position_Bin'] = (data['Start'] // bin_size) * bin_size
        
        # Count the SNPs in each bin for each chromosome
        snp_counts = data.groupby(['Chr_id', 'Position_Bin']).size().reset_index(name='SNP_Count')
        
        # Plot the frequency distribution with separation between chromosomes
        plt.figure(figsize=(15, 10))
        x_offset = 0  # Track x position offset for each chromosome
        
        for chr_id, chr_length in chr_lengths.items():
            chr_data = snp_counts[snp_counts['Chr_id'] == chr_id]
            # Adjust x positions by adding an offset based on cumulative chromosome length
            plt.bar(chr_data['Position_Bin'] + x_offset, chr_data['SNP_Count'], width=bin_size, label=f'{chr_id}')
            x_offset += chr_length  # Update offset for the next chromosome
            
        # Set x-ticks at approximate centers for each chromosome
        chr_centers = [(sum(list(chr_lengths.values())[:i]) + chr_lengths[chr] / 2) for i, chr in enumerate(chr_lengths.keys())]
        plt.xticks(chr_centers, [f'{chr}' for chr in chr_lengths.keys()])
        
        plt.xlabel('Chromosome')
        plt.ylabel('SNP Count')
        plt.title('SNP Frequency Distribution Across Chromosomes')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("No data to plot.")


# iii. SNP type vs. count

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def classify_snp_type(ref, alt):
    """Classifies SNPs based on reference and alternate alleles."""
    # Standardize allele pair order for consistent categorization
    snp_pair = f"{ref}/{alt}" if ref < alt else f"{alt}/{ref}"
    return snp_pair if snp_pair in {'A/G', 'C/T', 'A/C', 'A/T', 'G/C', 'G/T'} else None

def snp_type_distribution(data):
    # Ensure 'SNP Marker' and SNP columns are present
    if 'SNP Marker' not in data.columns:
        print("Missing required column 'SNP Marker'.")
        return

    # Define columns representing individuals
    individual_columns = data.columns[2:]
    
    # Initialize counts for each SNP type
    snp_type_counts = Counter({'A/G': 0, 'C/T': 0, 'A/C': 0, 'A/T': 0, 'G/C': 0, 'G/T': 0})
    
    # Loop over each SNP position
    for _, row in data.iterrows():
        alleles = row[individual_columns].apply(lambda x: x[0]).tolist()  # Extract first letter of each genotype
        ref_allele = max(set(alleles), key=alleles.count)  # Choose the most common allele as reference
        
        # Compare each allele with the reference to determine SNP type
        for alt_allele in alleles:
            if alt_allele != ref_allele:  # Only consider different alleles
                snp_type = classify_snp_type(ref_allele, alt_allele)
                if snp_type:
                    snp_type_counts[snp_type] += 1

    # Prepare data for plotting
    snp_types = list(snp_type_counts.keys())
    snp_counts = list(snp_type_counts.values())
    colors = ['blue' if snp in {'A/G', 'C/T'} else '#FFD700' for snp in snp_types]  # Transition/Transversion colors
    
    # Plot SNP type distribution
    plt.figure(figsize=(10, 6))
    bars = plt.bar(snp_types, snp_counts, color=colors)
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, snp_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(count), ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('SNP Type')
    plt.ylabel('Count')
    plt.title('Frequency of Specific SNP Types')
    
    # Create custom legend handles for transitions and transversions
    transition_patch = mpatches.Patch(color='blue', label='Transition (A/G, C/T)')
    transversion_patch = mpatches.Patch(color='#FFD700', label='Transversion (A/C, A/T, G/C, G/T)')
    plt.legend(handles=[transition_patch, transversion_patch], loc='upper right', title="SNP Type")

    plt.show()

# iv. Allele type

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patches as mpatches

def classify_allele_type(alleles):
    """Classifies allele type as homozygous or heterozygous with sub-categories."""
    if alleles[0] == alleles[1]:  # Homozygous
        return alleles  # e.g., 'AA', 'GG'
    else:  # Heterozygous
        # Standardize the order for heterozygous pairs
        allele_pair = ''.join(sorted(alleles))
        return allele_pair  # e.g., 'AG', 'CT'

def allele_type_distribution(data):
    # Ensure SNP columns are present
    individual_columns = data.columns[2:]
    
    # Initialize counts for each allele type
    allele_type_counts = Counter({
        'AA': 0, 'GG': 0, 'CC': 0, 'TT': 0,
        'AG': 0, 'CT': 0, 'AC': 0, 'AT': 0, 'GC': 0, 'GT': 0
    })
    
    # Loop over each SNP position for each individual
    for _, row in data.iterrows():
        for col in individual_columns:
            genotype = row[col]
            if len(genotype) == 2:  # Ensure genotype has two alleles
                allele_type = classify_allele_type(genotype)
                if allele_type:
                    allele_type_counts[allele_type] += 1

    # Prepare data for plotting
    allele_types = list(allele_type_counts.keys())
    allele_counts = list(allele_type_counts.values())
    
    # Color coding: dark yellow for homozygous, teal for heterozygous
    colors = ['#FFD700' if at in {'AA', 'GG', 'CC', 'TT'} else '#2CA02C' for at in allele_types]
    
    # Plot allele type distribution
    plt.figure(figsize=(12, 8))
    bars = plt.bar(allele_types, allele_counts, color=colors)
    
    # Add count labels on top of each bar
    for bar, count in zip(bars, allele_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, str(count), ha='center', va='bottom')
    
    # Add labels and title
    plt.xlabel('Allele Type')
    plt.ylabel('Count')
    plt.title('Frequency of Specific Allele Types')
    
    # Create custom legend handles for homozygous and heterozygous
    homozygous_patch = mpatches.Patch(color='#FFD700', label='Homozygous (e.g., AA, GG)')
    heterozygous_patch = mpatches.Patch(color='#2CA02C', label='Heterozygous (e.g., AG, CT)')
    plt.legend(handles=[homozygous_patch, heterozygous_patch], loc='upper right', title="Allele Type")

    plt.show()



def main():

    # Expanding the path to handle home directory
    file_name = os.path.expanduser('~/Downloads/WCC_SNP_file.csv')
    data = pd.read_csv(file_name, skiprows=[0])
    
    if 'WCC237' in data.columns:
        last_valid_column = data.columns.get_loc('WCC237') + 1
    else:
        print("Column 'WCC237' not found.")
        return
    
    data = data.iloc[:, :last_valid_column]
    print(data.head())
    
    snp_count_chr(data)

    snp_position_distribution(data)

    snp_type_distribution(data)
    
    allele_type_distribution(data)

if __name__ == '__main__':
    main()


