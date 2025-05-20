"""
Reusable plots for score-based deep-fake evaluation notebooks
=============================================================
All functions accept a pandas DataFrame with at least

  df_class   – the ground–truth label  (bona-fide / spoof)
  score      – system score            (high = bona-fide)

plus any additional columns the plot needs (e.g. 'tts').
Nothing in here touches notebook-specific globals.
"""

from __future__ import annotations
import pathlib
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
#  Global style helper  (use once at the top of every notebook)
# ------------------------------------------------------------------
def set_style(style: str = "seaborn-v0_8-darkgrid",
              figsize: tuple[int,int] = (12, 8),
              font_family: str = "sans-serif",
              grid_color: str = 'white') -> None:
    plt.style.use(style)
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.family"]   = font_family
    plt.rcParams['grid.color'] = grid_color
    plt.rcParams['grid.alpha'] = 0.5


# ------------------------------------------------------------------
#  1. Histogram of bona-fide vs. spoof scores
# ------------------------------------------------------------------
def plot_score_histogram(df: pd.DataFrame,
                         threshold: float | None = None,
                         *,
                         pos_label: Literal["spoof","bonafide"] = "spoof",
                         score_col: str = "score",
                         label_col: str = "df_class",
                         title: str | None = None,
                         save: str | pathlib.Path | None = None,
                         bins: str | int = "auto",
                         show_kde: bool = False,
                         fill_hist: bool = True,
                         alpha: float = 0.6,
                         figsize: tuple[int, int] | None = None,
                         only_kde: bool = False,
                         kde_alpha: float = 0.5) -> None:
    """One-liner wrapper around the long code block we kept pasting."""
    if df.empty:
        raise ValueError("DataFrame is empty – nothing to plot")

    if figsize:
        plt.figure(figsize=figsize) # Apply figsize if provided

    spoof = df[df[label_col] == pos_label][score_col]
    bona  = df[df[label_col] != pos_label][score_col]

    hist_element = "bars" if fill_hist else "step"
    hist_linewidth = 0 if fill_hist else 1.5
    hist_alpha = alpha if fill_hist else 1.0

    if only_kde:
        sns.kdeplot(spoof, color="orangered", label=f"Spoof (n={len(spoof)})", 
                    fill=True, alpha=kde_alpha, linewidth=1.5, bw_adjust=0.75)
        sns.kdeplot(bona,  color="dodgerblue", label=f"Bona-fide (n={len(bona)})", 
                    fill=True, alpha=kde_alpha, linewidth=1.5, bw_adjust=0.75)
    else:
        sns.histplot(spoof, color="orangered", label=f"Spoof (n={len(spoof)})",
                     kde=show_kde, stat="density", element=hist_element, linewidth=hist_linewidth,
                     bins=bins, alpha=hist_alpha, edgecolor='darkred' if fill_hist else 'orangered')
        sns.histplot(bona,  color="dodgerblue", label=f"Bona-fide (n={len(bona)})",
                     kde=show_kde, stat="density", element=hist_element, linewidth=hist_linewidth,
                     bins=bins, alpha=hist_alpha, edgecolor='darkblue' if fill_hist else 'dodgerblue')

    if threshold is not None:
        plt.axvline(threshold, color="k", ls="--", lw=1.5,
                    label=f"Threshold = {threshold:.4f}")

    plt.xlabel("Score");  plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()

    if title:
        plt.title(title, fontsize=14)

    if save:
        plt.savefig(save, dpi=300)
        if figsize: # Close the figure if a custom one was created for this plot
            plt.close()
    else:
        plt.show()
        if figsize: # Close the figure if a custom one was created for this plot
            plt.close()


# ------------------------------------------------------------------
#  2. KDE per-TTS model (BR-Speech use-case) - Enhanced
# ------------------------------------------------------------------
def plot_kde_by_tts(df: pd.DataFrame,
                    *,
                    group_col: str = "tts",
                    score_col: str = "score",
                    threshold: float | None = None,
                    bonafide_label: str = "Bonafide",
                    title: str | None = None,
                    save: str | pathlib.Path | None = None,
                    ax: plt.Axes | None = None,
                    show: bool = True,
                    legend: bool = True,
                    legend_title: str | None = None,
                    bonafide_color: str = 'darkgrey', # Specific color for bonafide
                    spoof_palette = "tab10", # Palette for other groups
                    bonafide_linewidth: float = 2.0,
                    spoof_linewidth: float = 1.0,
                    fill_alpha: float = 0.2,
                    bw_adjust: float = 0.5,
                    show_classification_regions: bool = False) -> plt.Axes:
    """
    Plots Kernel Density Estimates (KDE) for different groups, with enhanced styling.

    Distinguishes a 'bonafide' group with specific styling and plots it last.
    Optionally shows classification regions based on a threshold.
    """
    if df.empty:
        raise ValueError("Empty dataframe")
    if group_col not in df.columns:
         raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")
    if score_col not in df.columns:
         raise ValueError(f"Score column '{score_col}' not found in DataFrame.")


    # Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7)) # Adjust figsize as needed

    # Separate bonafide and spoof data
    bonafide_df = df[df[group_col] == bonafide_label]
    spoof_df = df[df[group_col] != bonafide_label]

    # Prepare spoof palette
    unique_spoof_groups = spoof_df[group_col].dropna().unique().tolist()
    if isinstance(spoof_palette, str):
        _colors = sns.color_palette(spoof_palette, len(unique_spoof_groups))
        spoof_color_map = {grp: _colors[i % len(_colors)] for i, grp in enumerate(unique_spoof_groups)}
    elif isinstance(spoof_palette, (list, tuple)):
        spoof_color_map = {grp: spoof_palette[i % len(spoof_palette)] for i, grp in enumerate(unique_spoof_groups)}
    elif isinstance(spoof_palette, dict):
        spoof_color_map = spoof_palette
    else:
        # Default fallback
        _colors = sns.color_palette("tab10", len(unique_spoof_groups))
        spoof_color_map = {grp: _colors[i % len(_colors)] for i, grp in enumerate(unique_spoof_groups)}


    # --- Plot Spoof KDEs ---
    zorder_counter = 2 # Start zorder for spoof groups
    plotted_spoof_labels = set()
    for grp, grp_df in spoof_df.groupby(group_col, observed=True):
        if grp_df.empty: continue
        color = spoof_color_map.get(grp)
        label = f"{grp} (n={len(grp_df)})"
        sns.kdeplot(data=grp_df, x=score_col, fill=True, common_norm=False,
                    alpha=fill_alpha, color=color, linewidth=spoof_linewidth,
                    bw_adjust=bw_adjust, label=label, ax=ax, zorder=zorder_counter)
        plotted_spoof_labels.add(label)
        zorder_counter += 1

    # --- Plot Bonafide KDE ---
    if not bonafide_df.empty:
        label = f"{bonafide_label} (n={len(bonafide_df)})"
        sns.kdeplot(data=bonafide_df, x=score_col, fill=True, common_norm=False,
                    alpha=fill_alpha, color=bonafide_color, linewidth=bonafide_linewidth,
                    bw_adjust=bw_adjust, label=label, ax=ax, zorder=zorder_counter)
        zorder_counter += 1
    else:
         print(f"Warning: No data found for bonafide_label='{bonafide_label}'")


    # --- Add Threshold Line and Regions ---
    final_zorder = zorder_counter
    if threshold is not None:
        # Add threshold line first
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5, # Use black for better contrast
                   label=f'Threshold ({threshold:.4f})', zorder=final_zorder)
        final_zorder += 1

        if show_classification_regions:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(0, ymax) # Ensure y starts at 0

            # Region: Classified as FAKE (Spoof)
            ax.fill_between([xmin, threshold], 0, ymax, color='dimgray', alpha=0.08, label='_nolegend_', zorder=1)
            ax.text(xmin + (threshold - xmin) * 0.5, ymax * 0.95, "Classified as FAKE",
                    ha='center', va='top', fontsize=10, color='dimgray', weight='bold', zorder=final_zorder)

            # Region: Classified as BONAFIDE
            ax.fill_between([threshold, xmax], 0, ymax, color='darkseagreen', alpha=0.1, label='_nolegend_', zorder=1)
            ax.text(threshold + (xmax - threshold) * 0.5, ymax * 0.95, "Classified as BONAFIDE",
                    ha='center', va='top', fontsize=10, color='darkgreen', weight='bold', zorder=final_zorder)


    # --- Final Touches ---
    ax.set_xlabel("Score", fontsize=12); ax.set_ylabel("Density", fontsize=12)

    if legend:
        leg_title = legend_title if legend_title is not None else group_col.replace('_', ' ').capitalize()
        ax.legend(title=leg_title, loc='best', fontsize=9, title_fontsize=10)

    if title:
        ax.set_title(title, fontsize=14, pad=15) # Add padding

    if save:
        # Use tight_layout before saving if not showing
        if not show: plt.tight_layout(rect=[0, 0.03, 1, 0.95] if show_classification_regions else None)
        plt.savefig(save, dpi=300)
        if show: plt.close(ax.figure) # Close figure if saved and also showing requested (to avoid double display)
    
    if show and not save: # Only show if not saved
         plt.tight_layout(rect=[0, 0.03, 1, 0.95] if show_classification_regions else None) # Adjust layout before showing
         plt.show()
    elif show and save: # If saved and show requested, maybe user wants to see it interactively too
         plt.tight_layout(rect=[0, 0.03, 1, 0.95] if show_classification_regions else None)
         plt.show()


    return ax


# ------------------------------------------------------------------
#  3. Simple class-count barplot  (train / dev / test)
# ------------------------------------------------------------------
def plot_class_distribution(df: pd.DataFrame,
                            *,
                            subset_col: str = "subset",
                            label_col: str = "df_class",
                            palette=("dodgerblue", "orangered"),
                            save: str | pathlib.Path | None = None) -> None:
    counts = (df.groupby([subset_col, label_col])
                .size().unstack(fill_value=0).reset_index())

    counts.set_index(subset_col).plot(kind="bar", stacked=True,
                                      color=palette)

    plt.ylabel("Count");  plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    else:    plt.show()


# ------------------------------------------------------------------
#  4. Bar plot of rates by group (e.g., TTS detection rate)
# ------------------------------------------------------------------
def plot_rate_by_group(df: pd.DataFrame,
                       *,
                       rate_col: str = "rate", # Column with the rate (0-1)
                       group_col: str = "group", # Column with group names
                       count_col: str | None = "total_samples", # Optional: add counts to labels
                       sort_by: Literal['rate', 'group', 'none'] = 'rate',
                       ascending: bool = True,
                       bonafide_label: str = "Bonafide",
                       bonafide_color: str = 'darkgrey',
                       palette = None,
                       title: str | None = None,
                       xlabel: str | None = None,
                       ylabel: str | None = None,
                       save: str | pathlib.Path | None = None) -> None:
    """Plots horizontal bar chart showing a rate for different groups."""
    if df.empty:
        raise ValueError("Empty dataframe")

    plot_df = df.copy()

    # Prepare labels with counts if requested
    if count_col and count_col in plot_df.columns:
        plot_df['label_with_n'] = plot_df.apply(
            lambda row: f"{row[group_col]} (n={int(row[count_col])})", axis=1
        )
        y_col = 'label_with_n'
    else:
        plot_df['label_with_n'] = plot_df[group_col]
        y_col = group_col

    # Determine colors
    if palette is None:
        palette = sns.color_palette("Set2", len(plot_df))

    # Assign distinct color for Bonafide if present
    if bonafide_label in plot_df[group_col].values:
        num_spoof = len(plot_df) - 1
        spoof_colors = sns.color_palette(palette, num_spoof) if isinstance(palette, str) else list(palette)[:num_spoof]
        bar_colors = [bonafide_color if cat == bonafide_label else spoof_colors[i % len(spoof_colors)]
                      for i, cat in enumerate(plot_df[group_col])]
    else:
        bar_colors = palette

    # Determine sorting order
    if sort_by == 'rate':
        # Special sort: Bonafide first/last, then by rate
        is_bonafide = (plot_df[group_col] == bonafide_label)
        plot_df['sort_key'] = plot_df[rate_col]
        # Assign a value to bonafide to place it first/last based on ascending
        bonafide_sort_val = -np.inf if ascending else np.inf
        plot_df.loc[is_bonafide, 'sort_key'] = bonafide_sort_val
        plot_df = plot_df.sort_values('sort_key', ascending=ascending)
        order = plot_df[y_col].tolist()
    elif sort_by == 'group':
        order = sorted(plot_df[y_col])
    else: # 'none'
        order = plot_df[y_col].tolist()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(plot_df))))
    sns.barplot(x=rate_col, y=y_col, data=plot_df, 
                palette=bar_colors if bonafide_label not in plot_df[group_col].values else None, # Use direct colors if bonafide exists
                hue=y_col if bonafide_label in plot_df[group_col].values else None, # Use hue to assign colors correctly
                order=order, ax=ax, legend=False)
    
    # Manually set colors if hue was used (for bonafide differentiation)
    if bonafide_label in plot_df[group_col].values:
        color_map = {label: color for label, color in zip(plot_df[y_col], bar_colors)}
        for i, bar in enumerate(ax.patches):
             bar.set_facecolor(color_map[ax.get_yticklabels()[i].get_text()])

    # --- Final Touches ---
    if title: ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel(xlabel if xlabel else f'{rate_col.capitalize()} Rate', fontsize=12)
    ax.set_ylabel(ylabel if ylabel else group_col.capitalize(), fontsize=12)
    ax.set_xlim(0, 1.05)

    # Add rate labels to bars
    for i, bar in enumerate(ax.patches):
        label_with_n = ax.get_yticklabels()[i].get_text()
        rate_value = plot_df.set_index(y_col).loc[label_with_n, rate_col]
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{rate_value:.1%}', va='center', ha='left', color='black', fontsize=10)

    sns.despine(ax=ax)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    else: plt.show()


# ------------------------------------------------------------------
#  5. Confusion Matrix Heatmap
# ------------------------------------------------------------------
def plot_confusion_matrix(df: pd.DataFrame,
                          threshold: float | None = None, # If None, assumes 'predicted_label' exists
                          *,
                          score_col: str = "score",
                          label_col: str = "df_class",
                          predicted_label_col: str | None = None, # Optional direct prediction column
                          pos_label: str = "spoof", # The value considered positive (fake)
                          neg_label: str = "bonafide", # The value considered negative (real)
                          cmap: str = 'Blues',
                          title: str | None = None,
                          save: str | pathlib.Path | None = None) -> None:
    """Plots a confusion matrix heatmap using seaborn."""
    if df.empty:
        raise ValueError("Empty dataframe")

    # Determine true labels (0 for pos_label/spoof, 1 for neg_label/bonafide)
    y_true = (df[label_col] == neg_label).astype(int)

    # Determine predicted labels
    if predicted_label_col and predicted_label_col in df.columns:
         # Use existing prediction column (assuming 0 for pos, 1 for neg)
         y_pred = (df[predicted_label_col] == neg_label).astype(int) 
    elif threshold is not None:
        # Classify based on threshold (score >= threshold -> predict bonafide/neg)
        y_pred = (df[score_col] >= threshold).astype(int)
    else:
         raise ValueError("Must provide either 'threshold' or 'predicted_label_col'")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() # TN=pred spoof/true spoof, FP=pred bonafide/true spoof, etc.
    
    # Reorder CM for standard display: True Neg (TN), FP | FN, True Pos (TP)
    # Our y_true/y_pred: 0=spoof, 1=bonafide -> CM is [[TN, FP], [FN, TP]]
    # This is the standard layout for sklearn, so no reordering needed if labels match.
    
    labels = [f'True {pos_label.capitalize()}', f'True {neg_label.capitalize()}']
    predicted_labels = [f'Pred {pos_label.capitalize()}', f'Pred {neg_label.capitalize()}']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=predicted_labels,
                yticklabels=labels,
                annot_kws={"size": 14})
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    if title:
        plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=300)
    else: plt.show() 