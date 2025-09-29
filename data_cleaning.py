import pandas as pd


def ffill_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Given a dataframe where there are empty values in region or municipality names, 
    fill the column forwards.

    For example, the "Region" column can be forward filled below.
    Region | Municipality 
    ---|---
    01 | 018
    | 049
    | 078
    02 | 019
    | 202
    ...|...
    """
    df[col] = df[col].ffill()
    return df


def pad_code(df: pd.DataFrame, col: str, code_length: int) -> pd.DataFrame:
    """
    Pad a column of a dataframe with zeros to code_length.
    For example, "1" becomes "01" with code_length=2.
    """
    return df[col].str.zfill(code_length)


def split_code_from_name(df: pd.DataFrame, col: str, code_length: int) -> pd.DataFrame:
    """
    Given a dataframe where area codes are listed with the area name, 
    split the code from the name and keep only the code of given length.
    For example, "MK01 Uusimaa" will become "01".
    """
    df[col] = (
        df[col]
        .astype(str)
        .str.split(" ").str[0]  # take only region code
        .str[-code_length:]     # take only length of the code
        .str.zfill(code_length)      # pad with zeros if needed
    )
    return df


def merge_with_final(df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a cleaned dataframe, merge it into a finalized dataframe on
    Region, Year, and Municipality. Ensure that columns Region and Municipality are
    of type str and Year is of type int.
    """
    merged_df = pd.merge(
        left=final_df,
        right=df,
        how="outer",
        on=["Region", "Year", "Municipality"]
    )
    return merged_df