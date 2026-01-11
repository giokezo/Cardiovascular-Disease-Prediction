import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplots(cols_to_plot,df,nrows,ncols,figsize,suptitle = 'Boxplots'):
    """
    Plot boxplots for specified columns
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    cols_to_plot : list
        List of column names to plot
    nrows : int, optional
        Number of rows (auto-calculated if None)
    ncols : int, default=3
        Number of columns
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    suptitle : str, default='Boxplots'
        Super title for the figure
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, var in enumerate(cols_to_plot):
        # Plot Boxplot
        sns.boxplot(data=df,x=var, ax=axes[i], fill=True, color='steelblue')
        
        axes[i].set_title(f'Boxplot of {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Density')
        axes[i].grid(True, alpha=0.3)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_kdes(cols_to_plot,df,nrows,ncols,figsize,suptitle = 'KDE Distributions'):
    """
    Plot KDE distributions for specified columns
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    cols_to_plot : list
        List of column names to plot
    nrows : int, optional
        Number of rows (auto-calculated if None)
    ncols : int, default=3
        Number of columns
    figsize : tuple, optional
        Figure size (auto-calculated if None)
    suptitle : str, default='KDE Distributions'
        Super title for the figure
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, var in enumerate(cols_to_plot):
        # Plot KDE
        sns.kdeplot(data=df[var], ax=axes[i], fill=True, alpha=0.6, color='steelblue')
        
        # Add mean line
        mean_val = df[var].mean()
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {mean_val:.2f}')
        
        axes[i].set_title(f'KDE Distribution of {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_piecharts(cols_to_plot, df, nrows, ncols, figsize, suptitle='Piecharts'):
    """
    Plot Piecharts for specified columns
    
    Parameters:
    -----------
    cols_to_plot : list
        List of column names to plot
    df : DataFrame
        Input dataframe
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize : tuple
        Figure size
    suptitle : str, default='Piecharts'
        Super title for the figure
    """
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, var in enumerate(cols_to_plot):
        # Get value counts for the variable
        value_counts = df[var].value_counts()
        
        # Plot Piechart
        axes[i].pie(value_counts, 
                   labels=value_counts.index,
                   autopct='%1.1f%%',  # Show percentages
                   startangle=90,
                   colors=sns.color_palette('pastel'))
        
        axes[i].set_title(f'Distribution of {var}', fontweight='bold')

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_scatterplot(df,x,y,hue = None,title = 'Scatterplot'):
    """
    Plot Scatterplot for specified columns
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    x : Column Name
        Column to Plot on X Axis
    y : Column Name
        Column to Plot on Y Axis
    hue : Hue
        Column to color the points by
    title : str, default='Scatterplot'
        Title for the figure
    """
    plt.figure(figsize= (9,4))

    sns.scatterplot(data = df, x= x, y = y, hue = hue)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_crosstab(df,x,y):
    # crosstab fo the housing x target variable
    crosstab = pd.crosstab(df[y], df[x])

    plt.figure(figsize=(8, 5))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Crosttab of {x} by {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def plot_bar_by_target(df, columns_to_plot,target,suptitle = 'Barplots by target variable'):
        n_cols = 2
        n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(columns_to_plot):
            crosstab = pd.crosstab(df[col], df[target])
            crosstab.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'Frequency of {col} by {target}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend(title=target)

        for j in range(len(columns_to_plot), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.show()