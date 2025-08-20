import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import plotly as plt
    import plotly.express as px
    import pandas as pd
    import plotly.figure_factory as ff
    return alt, mo, pl


@app.cell
def _(mo):
    valgt_fil = mo.ui.file_browser()
    valgt_fil
    return (valgt_fil,)


@app.cell
def _(valgt_fil):
    file_info = valgt_fil.value[0]
    filepath = file_info.path
    str(filepath)
    return (filepath,)


@app.cell
def _(filepath, mo):
    artsdat_df = mo.sql(
        f"""
         SELECT * FROM read_csv('{str(filepath)}');
        """,
        output=False
    )
    return


@app.cell
def _(artsdata_df, mo):
    # Husk at du må velge data for at kartene skal fungere!

    artsdata_tid_df = mo.ui.table(artsdata_df)
    artsdata_tid_df
    return (artsdata_tid_df,)


@app.cell
def _(artsdata_tid_df):
    artsdata_tid = artsdata_tid_df.value
    return (artsdata_tid,)


@app.cell(hide_code=True)
def _(mo):
    toggle = mo.ui.switch(label="Individer", value=False)
    window_size = mo.ui.slider(start=1, stop=30, step=1, show_value=True, label="Antall dager i rullende gj.snitt")
    window_size
    return toggle, window_size


@app.cell(hide_code=True)
def _(alt, artsdata_tid, mo, pl, toggle, window_size):
    daily_stats = (
        artsdata_tid
        .with_columns([
            pl.col('Observert dato').dt.date().alias('date'),
            pl.col('Observert dato').dt.year().alias('year'),
            pl.col('Antall').cast(pl.Int64, strict=False).fill_null(1).alias('ind_count')
        ])
        .group_by('date')
        .agg([
            pl.len().alias('daily_obs_count'),
            pl.col('ind_count').sum().alias('daily_ind_count'),
            pl.col('year').first().alias('year')
        ])
        # Create a synthetic date using year 2024 for all data to show pattern
        .with_columns([
            pl.date(2024, pl.col('date').dt.month(), pl.col('date').dt.day()).alias('common_date')
        ])
        .group_by('common_date')
        .agg([
            # For observations
            pl.col('daily_obs_count').mean().alias('avg_daily_obs'),
            pl.col('daily_obs_count').std().alias('std_daily_obs'),
            # For individuals
            pl.col('daily_ind_count').mean().alias('avg_daily_ind'),
            pl.col('daily_ind_count').std().alias('std_daily_ind'),
            # Count years
            pl.col('daily_obs_count').count().alias('n_years')
        ])
        .sort('common_date')
        .with_columns([
            # Rolling averages
            pl.col('avg_daily_obs').rolling_mean(window_size.value, center=True).alias('rolling_avg_obs'),
            pl.col('avg_daily_ind').rolling_mean(window_size.value, center=True).alias('rolling_avg_ind'),
            # Standard errors
            (pl.col('std_daily_obs') / pl.col('n_years').sqrt()).alias('se_obs'),
            (pl.col('std_daily_ind') / pl.col('n_years').sqrt()).alias('se_ind')
        ])
        .with_columns([
            # Confidence bands
            (pl.col('rolling_avg_obs') - pl.col('se_obs')).alias('lower_obs'),
            (pl.col('rolling_avg_obs') + pl.col('se_obs')).alias('upper_obs'),
            (pl.col('rolling_avg_ind') - pl.col('se_ind')).alias('lower_ind'),
            (pl.col('rolling_avg_ind') + pl.col('se_ind')).alias('upper_ind')
        ])
    )

    # Create observations chart
    obs_chart = (
        alt.Chart(daily_stats).mark_area(opacity=0.3, color='lightblue').encode(
            x=alt.X('common_date:T', 
                    title='Dato',
                    axis=alt.Axis(format='%d %b')),
            y=alt.Y('lower_obs:Q', title='Rullerende gjennomsnitt (observasjoner)'),
            y2='upper_obs:Q'
        ) +
        alt.Chart(daily_stats).mark_line(point=True, size=2, color='steelblue').encode(
            x='common_date:T',
            y='rolling_avg_obs:Q',
            tooltip=[
                alt.Tooltip('common_date:T', title='Dato', format='%d %B'),
                alt.Tooltip('rolling_avg_obs:Q', title='Rullerende gjennomsnitt', format='.1f'),
                alt.Tooltip('se_obs:Q', title='Standardfeil', format='.2f')
            ]
        )
    ).properties(
        width=900,
        height=400,
        title='Observasjoner'
    ).interactive()

    # Create individuals chart
    ind_chart = (
        alt.Chart(daily_stats).mark_area(opacity=0.3, color='peachpuff').encode(
            x=alt.X('common_date:T', 
                    title='Dato',
                    axis=alt.Axis(format='%d %b')),
            y=alt.Y('lower_ind:Q', title='Rullerende gjennomsnitt (individer)'),
            y2='upper_ind:Q'
        ) +
        alt.Chart(daily_stats).mark_line(
            point={'filled': True, 'fill': 'darkorange', 'size': 20}, 
            size=2, 
            color='darkorange'
        ).encode(
            x='common_date:T',
            y='rolling_avg_ind:Q',
            tooltip=[
                alt.Tooltip('common_date:T', title='Dato', format='%d %B'),
                alt.Tooltip('rolling_avg_ind:Q', title='Rullerende gjennomsnitt', format='.1f'),
                alt.Tooltip('se_ind:Q', title='Standardfeil', format='.2f')
            ]
        )
    ).properties(
        width=900,
        height=800,
        title='Individer'
    ).interactive()

    # Display toggle and appropriate chart
    mo.vstack([
        toggle,
        ind_chart if toggle.value else obs_chart
    ])
    return


if __name__ == "__main__":
    app.run()
