#code based on module priciples of data science from my MSc
import pandas as pd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt


def ds_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is designed to create view of a dataframe with respect to types, unique values, missing values and % of missing values.
    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe with all features the user wants to scan
    Returns
    -------
    pd.DataFrame
        dataframe with view of:
                        1) Data Types
                        2) Cardinality
                        3) Volume of NaN values and
                        4) NaN value count as % of total rows
    """

    df = df.copy()

    # Cardinality df
    unvaldf = pd.DataFrame(df.nunique(), columns=["Cardinality"])
    unvaldf.reset_index(drop=False, inplace=True)
    unvaldf.rename(columns={"index": "Feature"}, inplace=True)

    # Data Type DF
    d_Typedf = pd.DataFrame(df.dtypes, columns=["DataType"])
    d_Typedf.reset_index(drop=False, inplace=True)
    d_Typedf.rename(columns={"index": "Feature"}, inplace=True)
    d_Typedf["DataType"] = d_Typedf["DataType"].astype(str)

    # NaN Values df
    nansumdf = pd.DataFrame(df.isna().sum(), columns=["NanCount"])
    nansumdf.reset_index(drop=False, inplace=True)
    nansumdf.rename(columns={"index": "Feature"}, inplace=True)
    nansumdf["NaN_%_of_Rows"] = nansumdf["NanCount"].apply(lambda x: round(x / len(df), 5))

    # Take each of the dataframes created above, re-index on "Feature"
    dfs = [df.set_index("Feature") for df in [unvaldf, d_Typedf, nansumdf]]

    # Join list of dataframes on index
    featureoverview_df = dfs[0].join(dfs[1:])
    featureoverview_df.reset_index(drop=False, inplace=True)

    return featureoverview_df


def add_median_labels(ax: plt.Axes) -> None:
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == "PathPatch"]
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4 : len(lines) : lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontweight="bold", color="white")
        # create median-colored border around white text for contrast
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3, foreground=median.get_color()),
                path_effects.Normal(),
            ]
        )