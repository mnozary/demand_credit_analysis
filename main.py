import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def read_regional_data(file_name):
    # Load the Excel file
    xl = pd.ExcelFile(file_name)

    # Read the 'Sectoral Reductions' sheet
    sheet_name = 'Regional Reductions'
    df = xl.parse(sheet_name, header=1)

    # Extract columns B:M from row 2 onwards
    start_row = 2
    end_row = df.shape[0]
    # columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    start_col = 1  # Index 1 corresponds to column B
    end_col = 13  # Index 12 corresponds to column M
    selected_data = df.iloc[start_row:end_row, start_col:end_col]
    # Remove columns 8, 9
    columns_to_remove = [8, 9]
    selected_data = selected_data.drop(selected_data.columns[columns_to_remove], axis=1)

    return selected_data


def plot_scatter_for_regional_data(df_regional):
    plt.clf()
    # Get the column names as x-axis values
    x = df_regional.columns.tolist()
    x_indices = range(len(x))
    # Define the order of indices and their distances
    # index_order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1]
    index_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    distances = [0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4]
    for i, index in enumerate(index_order):
        col = df_regional.columns[index]
        y = df_regional[col].dropna().values * 100
        if len(y) > 0:
            color = '#646B86' if i % 2 == 0 else '#6B204B'  # Specify the colors for odd and even indices
            plt.scatter([i - distances[i]] * len(y), y, s=1, color=color, alpha=0.1)
            median_val = np.median(y)
            avg_val = np.mean(y)
            plt.plot([i - distances[i] - 0.2, i - distances[i] + 0.2],
                     [median_val, median_val], color=color, marker="_", markersize=2, linewidth=2)
            plt.plot([i - distances[i] - 0.2, i - distances[i] + 0.2],
                     [avg_val, avg_val], color=color, marker="x", markersize=2, linewidth=2)
    # Format y-axis ticks as percentages
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(yticks)

    # Set y-axis limits and ticks
    plt.ylim(-40, 40)
    plt.yticks(np.arange(-40, 41, 10))
    plt.xticks([])  # Remove the x-axis labels
    plt.gca().spines['left'].set_color('gray')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.tight_layout()
    # Save the plot as a high-resolution JPEG image
    plt.savefig('Regional_Reductions.jpg', dpi=400, bbox_inches='tight')
    return plt


def read_sectoral_data(file_name):
    # Load the Excel file
    xl = pd.ExcelFile(file_name)

    # Read the 'Sectoral Reductions' sheet
    sheet_name = 'Sectoral Reductions'
    df = xl.parse(sheet_name, header=1)

    # Extract columns B:M from row 2 onwards
    start_row = 3
    end_row = df.shape[0]
    # columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    start_col = 1  # Index 1 corresponds to column B
    end_col = 29  # Index 28 corresponds to column AC
    selected_data = df.iloc[start_row:end_row, start_col:end_col]
    # Remove columns 10, 11, 20, 21
    columns_to_remove = [10, 11, 20, 21]
    selected_data = selected_data.drop(selected_data.columns[columns_to_remove], axis=1)

    return selected_data


def plot_scatter_for_sectoral_data(df_sectoral):
    plt.clf()

    # Get the column names as x-axis values
    x = df_sectoral.columns.tolist()
    x_indices = range(len(x))
    # Define the order of indices and their distances
    # index_order = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 1]
    index_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    distances = [0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4, 0, 0.4]
    # Create a scatter plot for each column's non-null values
    for i, index in enumerate(index_order):
        col = df_sectoral.columns[index]
        y = df_sectoral[col].dropna().values * 100
        if len(y) > 0:
            color = '#646B86' if i % 2 == 0 else '#6B204B'
            plt.scatter([i - distances[i]] * len(y), y, s=1, color=color, alpha=0.1)
            median_val = np.median(y)
            avg_val = np.mean(y)
            plt.plot([i - distances[i] - 0.2, i - distances[i] + 0.2],
                     [median_val, median_val], color=color, marker="_", markersize=2, linewidth=2)
            plt.plot([i - distances[i] - 0.2, i - distances[i] + 0.2],
                     [avg_val, avg_val], color=color, marker="x", markersize=2, linewidth=2)
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(yticks)

    # Set y-axis limits and ticks
    plt.ylim(-40, 40)
    plt.yticks(np.arange(-40, 41, 10))
    plt.xticks([])
    plt.gca().spines['left'].set_color('gray')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['bottom'].set_linewidth(1)
    plt.tight_layout()
    # Save the plot as a high-resolution JPEG image
    plt.savefig('Sectoral_Reductions.jpg', dpi=400, bbox_inches='tight')
    return plt


if __name__ == '__main__':
    file_path = 'JP Reduction vs Credits analysis v7.9 (CDO 18.05).xlsm'
    sectoral_df = read_sectoral_data(file_path)
    plt_regional = plot_scatter_for_sectoral_data(sectoral_df)
    regional_df = read_regional_data(file_path)
    plt2 = plot_scatter_for_regional_data(regional_df)