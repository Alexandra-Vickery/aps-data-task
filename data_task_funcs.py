import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch



def process_file(datafile):
    """
    Goes from the raw file to something workable in python environment
    """
    # loading in the file
    df = pd.read_excel(datafile, sheet_name=0)
    id_cols = ['Age', 'Gender', 'Region', 'Time']
    qu_cols = [c for c in df.columns if c not in id_cols]
    long_df = df.melt(id_vars=id_cols, value_vars=qu_cols, var_name='question_response', value_name='count')
    long_df[['question', 'response_code']] = (long_df['question_response'].str.extract(r'(\d+)v(\d+)'))

    # bespoke replacements
    long_df['question'] = long_df['question'].astype(int)

    long_df["count"] = (long_df["count"].replace({"<10": 5})) # just set <10 to 5 for now, can modify if we want
    long_df['count'] = (long_df['count'].replace({'.' : np.nan}).astype(float))

    long_df['response_code'] = long_df['response_code'].astype(int)
    long_df["response_code"] = (long_df["response_code"].replace({12: '(5) strong disagree'}))
    long_df["response_code"] = (long_df["response_code"].replace({3: '(4) disagree'}))
    long_df["response_code"] = (long_df["response_code"].replace({4: '(3) neither'}))
    long_df["response_code"] = (long_df["response_code"].replace({5: '(2) agree'}))
    long_df["response_code"] = (long_df["response_code"].replace({67: '(1) strong agree'}))

    long_df["Time"] = (long_df["Time"].replace({'March 2019': '2019-03'}))
    long_df["Time"] = (long_df["Time"].replace({'June 2019': '2019-06'}))
    long_df["Time"] = (long_df["Time"].replace({'November 2019': '2019-11'}))
    long_df["Time"] = (long_df["Time"].replace({'February 2020': '2020-02'}))
    long_df["Time"] = (long_df["Time"].replace({'June 2020': '2020-06'}))
    long_df["Time"] = (long_df["Time"].replace({'November 2020': '2020-11'}))
    long_df["Time"] = (long_df["Time"].replace({'February 2021': '2021-02'}))
    long_df["Time"] = (long_df["Time"].replace({'June 2021': '2021-06'}))
    long_df["Time"] = (long_df["Time"].replace({'Jul-Sep 2021': '2021-08'})) # just picking the middle month
    long_df["Time"] = (long_df["Time"].replace({'Oct-Dec 2021': '2021-10'}))
    long_df["Time"] = (long_df["Time"].replace({'Jan-Mar 2022': '2022-02'}))
    long_df["Time"] = (long_df["Time"].replace({'Apr-Jun 2022': '2022-05'}))

    return long_df


def plot_responses_collapsible(
    df_long,
    filters=None,
    question=None,
    date_col="Time",
    normalise=False,
    collapse=None
):
    """
    Use to look at individual samples of the data
    """
    data = df_long.copy()

    # Apply demographic filters
    if filters:
        for col, value in filters.items():
            data = data[data[col] == value]

    # Optional question filter
    if question is not None:
        data = data[data["question"] == int(question)]

    if collapse == 'conviction':
        conviction_map = {
            '(1) strong agree': 'Strong opinion',
            '(2) agree': 'Moderate opinion',
            '(3) neither': 'Neutral',
            '(4) disagree': 'Moderate opinion',
            '(5) strong disagree': 'Strong opinion'
        }
        data = data.copy()
        data["response_code"] = (data["response_code"].map(conviction_map).fillna(data["response_code"]))
        response_colors = {
            'Strong opinion': 'tab:red',
            'Moderate opinion': 'tab:orange',
            'Neutral': 'tab:purple'
        }
    elif collapse == 'sentiment':
        conviction_map = {
            '(1) strong agree': 'Positive',
            '(2) agree': 'Positive',
            '(3) neither': 'Neutral',
            '(4) disagree': 'Negative',
            '(5) strong disagree': 'Negative'
        }
        data = data.copy()
        data["response_code"] = (data["response_code"].map(conviction_map).fillna(data["response_code"]))
        response_colors = {
            'Positive': 'tab:green',
            'Neutral': 'tab:purple',
            'Negative': 'tab:orange'
        }
    else:
        response_colors = {
            '(1) strong agree': "tab:red",
            '(2) agree': "tab:orange",
            '(3) neither': "tab:purple",
            '(4) disagree': "tab:green",
            '(5) strong disagree': "tab:blue",
        }

    # Aggregate
    plot_df = (data.groupby([date_col, "response_code"])["count"].sum().unstack(fill_value=0).sort_index())
    ordered_cols = [col for col in response_colors.keys() if col in plot_df.columns]
    ordered_cols += [col for col in plot_df.columns if col not in ordered_cols]
    plot_df = plot_df[ordered_cols]

    # Normalise so every bar sums to 100%
    if normalise:
        row_totals = plot_df.sum(axis=1)
        plot_df = plot_df.div(row_totals, axis=0).fillna(0)

    colours = [response_colors.get(col, "lightgrey") for col in plot_df.columns]

    ax = plot_df.plot(kind="bar", stacked=True, figsize=(12, 6), color=colours)

    # Build title
    title_parts = []
    if filters:
        title_parts.extend([f"{k}={v}" for k, v in filters.items()])
    if question is not None:
        title_parts.append(f"Question {question}")
    ax.set_title(
        " | ".join(title_parts)
        if title_parts
        else "All Responses"
    )

    ax.set_xlabel("Survey Date")

    if normalise:
        ax.set_ylabel("Percentage of Responses")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.set_ylabel("Number of Responses")

    ax.legend(title="Response", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def compare_groups_collapsible(
    df_long,
    group_cols,
    selected_groups,
    question=None,
    normalise=False,
    collapse=None
):
    """
    Use to compare samples of the data
    """
    data = df_long.copy()

    if question is not None:
        data = data[data["question"] == int(question)]

    if collapse == 'conviction':
        conviction_map = {
            '(1) strong agree': 'Strong opinion',
            '(2) agree': 'Moderate opinion',
            '(3) neither': 'Neutral',
            '(4) disagree': 'Moderate opinion',
            '(5) strong disagree': 'Strong opinion'
        }
        data = data.copy()
        data["response_code"] = (data["response_code"].map(conviction_map).fillna(data["response_code"]))
        response_colors = {
            'Strong opinion': 'tab:red',
            'Moderate opinion': 'tab:orange',
            'Neutral': 'tab:purple'
        }

    elif collapse == 'sentiment':
        sentiment_map = {
            '(1) strong agree': 'Positive',
            '(2) agree': 'Positive',
            '(3) neither': 'Neutral',
            '(4) disagree': 'Negative',
            '(5) strong disagree': 'Negative'
        }
        data = data.copy()
        data["response_code"] = (data["response_code"].map(sentiment_map).fillna(data["response_code"]))
        response_colors = {
            'Positive': 'tab:green',
            'Neutral': 'tab:purple',
            'Negative': 'tab:orange'
        }

    else:
        response_colors = {
            '(1) strong agree': "tab:red",
            '(2) agree': "tab:orange",
            '(3) neither': "tab:purple",
            '(4) disagree': "tab:green",
            '(5) strong disagree': "tab:blue",
        }
    data["group"] = (data[group_cols].astype(str).agg(" | ".join, axis=1))
    data = data[data["group"].isin(selected_groups)]
    summary = (data.groupby(["Time", "group", "response_code"])["count"].sum().reset_index())
    pivot = summary.pivot_table(index=['Time', 'group'], columns='response_code', values='count', fill_value=0)
    totals = (summary.groupby(['Time', 'group'])['count'].sum().to_dict())

    if normalise:
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100

    groups = selected_groups

    if len(groups) == 3:
        group_hatches = {
            groups[0]: "",
            groups[1]: "//",
            groups[2]: "oo"
        }
    else:
        group_hatches = {
            groups[0]: "",
            groups[1]: "//",
        }
    
    dates = sorted(summary["Time"].unique())

    fig, ax = plt.subplots(figsize=(14, 7))

    width = 0.8 / len(groups)
    x = np.arange(len(dates))

    response_order = [r for r in response_colors.keys() if r in summary["response_code"].unique()]
    response_handles = [Patch(facecolor=response_colors[r], label=r) for r in response_order]

    for i, group in enumerate(groups):
        bottoms = np.zeros(len(dates))
        for response in response_order:
            vals = []

            for date in dates:
                mask = (
                    (summary["group"] == group) &
                    (summary["Time"] == date) &
                    (summary["response_code"] == response)
                )

                count = summary.loc[mask, "count"].sum()

                if normalise:
                    total = totals.get((date, group), 0)
                    count = count / total if total > 0 else 0

                vals.append(count)

            vals = np.array(vals)

            ax.bar(
                x + i * width,
                vals,
                width,
                bottom=bottoms,
                color=response_colors[response],
                hatch=group_hatches[group],
                label=f"Response {response}"
                if i == 0 else None
            )

            bottoms += vals

    ax.set_xticks(x + width*(len(groups)-1)/2)
    ax.set_xticklabels(dates, rotation=45)
    ax.set_ylabel('Percentage of Responses' if normalise else 'Responses')
    legend1 = ax.legend(handles=response_handles, title='Response Code', loc='upper right')
    group_handles = [Patch(facecolor='lightgrey', hatch=group_hatches[g], label=g) for g in groups]
    legend2 = ax.legend(handles=group_handles, title='Group', loc='upper left')
    ax.add_artist(legend1)
    plt.tight_layout()
    plt.show()