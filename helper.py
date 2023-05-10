
import numpy as np
import pandas as pd

def fetch_medal_tally(df, year, country):
    # remove duplicate medal entries
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    elif year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    elif year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    else:
        temp_df = medal_df[(medal_df['Year'] == int(year)) & (medal_df['region'] == country)]

    # Convert medals into counts
    medal_counts = pd.pivot_table(
        temp_df,
        index=('Year' if flag == 1 else 'region'),
        columns='Medal',
        values='Event',   # any column, since we just want counts
        aggfunc='count',
        fill_value=0
    ).reset_index()

    # Ensure Gold, Silver, Bronze columns exist
    for col in ['Gold', 'Silver', 'Bronze']:
        if col not in medal_counts.columns:
            medal_counts[col] = 0

    # Add total column
    medal_counts['total'] = medal_counts['Gold'] + medal_counts['Silver'] + medal_counts['Bronze']

    # Sort properly
    if flag == 1:
        medal_counts = medal_counts.sort_values('Year').reset_index(drop=True)
    else:
        medal_counts = medal_counts.sort_values('Gold', ascending=False).reset_index(drop=True)

    return medal_counts


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country


def data_over_time(df, col):
    # Drop duplicates for each year–col combo
    temp_df = df.drop_duplicates(['Year', col])

    # Count how many unique `col` entries per Year
    nations_over_time = temp_df.groupby('Year')[col].count().reset_index()

    # Rename columns for clarity
    nations_over_time.rename(columns={'Year': 'Edition', col: col}, inplace=True)

    return nations_over_time


def most_successful(df, sport):
    temp_df = df.dropna(subset=['Medal'])

    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # Get top 15 athletes
    top_athletes = (
        temp_df['Name']
        .value_counts()
        .reset_index()
        .head(15)
    )
    top_athletes.columns = ['Name', 'Medal_Count']  # Rename for clarity

    # Merge with original DF to get sport & region info
    x = top_athletes.merge(df, on='Name', how='left')[['Name', 'Medal_Count', 'Sport', 'region']].drop_duplicates(
        'Name')

    return x
def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]

    # Get top 10 athletes by medal count
    top_athletes = (
        temp_df['Name']
        .value_counts()
        .reset_index()
        .head(10)
    )
    top_athletes.columns = ['Name', 'Medal_Count']  # rename properly

    # Merge with original df to get sport and region info
    x = (
        top_athletes
        .merge(df, on='Name', how='left')
        [['Name', 'Medal_Count', 'Sport']]
        .drop_duplicates('Name')
    )

    return x

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final
