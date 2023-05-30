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
    start_row = 2  # Indexing starts from 0, so row 3 is at index 2
    end_row = df.shape[0]  # Number of rows in the dataframe
    # columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    start_col = 1  # Index 1 corresponds to column B
    end_col = 12  # Index 12 corresponds to column M
    selected_data = df.iloc[start_row:end_row, start_col:end_col]
    # Remove columns 11, 12, 21, and 22
    columns_to_remove = [7, 8]
    selected_data = selected_data.drop(selected_data.columns[columns_to_remove], axis=1)

    return selected_data


def read_sectoral_data(file_name):
    # Load the Excel file
    xl = pd.ExcelFile(file_name)

    # Read the 'Sectoral Reductions' sheet
    sheet_name = 'Sectoral Reductions'
    df = xl.parse(sheet_name, header=1)

    # Extract columns B:M from row 2 onwards
    start_row = 2  # Indexing starts from 0, so row 3 is at index 2
    end_row = df.shape[0]  # Number of rows in the dataframe
    # columns = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    start_col = 1  # Index 1 corresponds to column B
    end_col = 28  # Index 12 corresponds to column M
    selected_data = df.iloc[start_row:end_row, start_col:end_col]
    # Remove columns 11, 12, 21, and 22
    columns_to_remove = [10, 11, 20, 21]
    selected_data = selected_data.drop(selected_data.columns[columns_to_remove], axis=1)

    return selected_data


def plot_scatter_from_excel(df):
    # Get the column names as x-axis values
    x = df.columns.tolist()
    x_indices = range(len(x))
    # Create a scatter plot for each column's non-null values
    spacing = 0.4  # Adjust the spacing between scatter plots
    for col, index in zip(x[::2], x_indices[::2]):
        y = df[col].dropna().values
        if len(y) > 0:
            if index % 2 == 0:  # Assign color based on even or odd index
                color = '#646B86'  # Specify color for even index scatter plots
            else:
                color = '#6B204B'  # Specify color for odd index scatter plots
            plt.scatter([index - spacing] * len(y), y, s=1, color=color)  # , label=col)
            median_val = df[col].median()
            avg_val = df[col].mean()
            plt.plot([index - 0.2, index + 0.2], [median_val, median_val], color='r', linestyle='-',
                     linewidth=1)
            plt.plot([index], [median_val - 0.2], color='r', marker='x', markersize=5)
            plt.plot([index], [median_val + 0.2], color='r', marker='x', markersize=5)


    spacing += 0.2  # Increase the spacing before plotting odd-indexed columns
    # Create scatter plots for odd-indexed columns (remaining columns)
    for col, index in zip(x[1::2], x_indices[1::2]):
        y = df[col].dropna().values
        if len(y) > 0:
            if index % 2 == 0:  # Assign color based on even or odd index
                color = '#646B86'  # Specify color for even index scatter plots
            else:
                color = '#6B204B'  # Specify color for odd index scatter plots
            plt.scatter([index - spacing] * len(y), y, s=1, color=color)  # , label=col)
            median_val = df[col].median()
            avg_val = df[col].mean()
            plt.plot([index - spacing - 0.2, index - spacing + 0.2], [median_val, median_val], color='r',
                     linestyle='-', linewidth=1)
            plt.plot([index - spacing - 0.2, index - spacing + 0.2], [avg_val, avg_val], color='g', marker='x',
                     markersize=1)

    # Configure plot
    # font_size = 6
    # plt.xlabel('Sectoral Reductions', fontsize=font_size)
    # plt.ylabel('% Reduction', fontsize=font_size)
    # Set the y-axis tick formatter
    # Display the legend outside of the plot area
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    # plt.title('Available_Range Scope 1+2+3 % Reduction (2017-2022)', fontsize=font_size+2)
    # plt.xticks(x_indices, x, rotation=35, fontsize=font_size)
    # Add manual legend entries for average and median
    # plt.plot([], [], color='r', linestyle='-', label='Median')
    # plt.plot([], [], color='g', linestyle='x', label='Average')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=font_size)

    # Format y-axis ticks as percentages
    fmt = '%.0f%%'  # Set the format to display as percentage
    yticks = mtick.FormatStrFormatter(fmt)
    plt.gca().yaxis.set_major_formatter(yticks)

    # Set y-axis limits and ticks
    plt.ylim(-40, 40)
    plt.yticks(np.arange(-40, 41, 10))

    # Change the color of the borderlines to gray
    plt.plot([-0.2, len(x_indices) - 0.8], [0, 0], color='#A6A6A6', linewidth=1)
    plt.plot([-0.2, len(x_indices) - 0.8], [0, 0], color='#A6A6A6', linewidth=1)

    plt.tight_layout()
    # Save the plot as a high-resolution JPEG image
    plt.savefig('Sectoral_Reductions.jpg', dpi=400, bbox_inches='tight')
    return plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = 'JP Reduction vs Credits analysis v7.9 (CDO 18.05).xlsm'
    sectoral_df = read_sectoral_data(file_path)
    print(sectoral_df.shape)

    # For Regions - please do not include ME&A
    # For Sectors - please do not include Fossil Fuels and Power Generation
    # make each sector near each other and bigger gap to the other
    # total at the right-side
    # the same color for
    # change the borderlines gray
    # x-axis without label
    # y-axis percentage -40 to 40% by 10%
    plt = plot_scatter_from_excel(sectoral_df)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
