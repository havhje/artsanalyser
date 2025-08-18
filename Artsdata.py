import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full", layout_file="layouts/Artsdata.grid.json")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    import plotly as plt
    import plotly.express as px
    import pandas as pd
    import plotly.figure_factory as ff
    return alt, ff, mo, pl, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Organiserer og transformerer data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Organiserer""")
    return


@app.cell
def _(mo):
    original_df = mo.sql(
        f"""
        SELECT * FROM '/Users/havardhjermstad-sollerud/Documents/Kodeprosjekter/marimo/Artsdatabanken/inndata/*.csv';
        """
    )
    return (original_df,)


@app.cell
def _(original_df):
    dropper_kolonner_df = original_df
    dropper_kolonner_df = dropper_kolonner_df.select(["category", "validScientificNameId", "validScientificName", "preferredPopularName", "taxonGroupName", "collector", "dateTimeCollected", "locality", "coordinateUncertaintyInMeters", "municipality", "county", "individualCount", "latitude", "longitude", "geometry", "scientificNameRank", "behavior"])
    return (dropper_kolonner_df,)


@app.cell
def _(dropper_kolonner_df):
    endrer_navn_df = dropper_kolonner_df
    endrer_navn_df = endrer_navn_df.rename({"category": "Kategori"})
    endrer_navn_df = endrer_navn_df.rename({"validScientificName": "Art"})
    endrer_navn_df = endrer_navn_df.rename({"preferredPopularName": "Navn"})
    endrer_navn_df = endrer_navn_df.rename({"taxonGroupName": "Artsgruppe"})
    endrer_navn_df = endrer_navn_df.rename({"validScientificNameId": "Artens ID"})
    endrer_navn_df = endrer_navn_df.rename({"collector": "Observatør"})
    endrer_navn_df = endrer_navn_df.rename({"dateTimeCollected": "Observert dato"})
    endrer_navn_df = endrer_navn_df.rename({"locality": "Lokalitet"})
    endrer_navn_df = endrer_navn_df.rename({"coordinateUncertaintyInMeters": "Usikkerhet meter"})
    endrer_navn_df = endrer_navn_df.rename({"municipality": "Kommune"})
    endrer_navn_df = endrer_navn_df.rename({"county": "Fylke"})
    endrer_navn_df = endrer_navn_df.rename({"individualCount": "Antall"})
    endrer_navn_df = endrer_navn_df.rename({"scientificNameRank": "Taksonomisk nivå"})
    endrer_navn_df = endrer_navn_df.rename({"behavior": "Atferd"})
    return (endrer_navn_df,)


@app.cell
def _(endrer_navn_df):
    organisert_df = endrer_navn_df
    return (organisert_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Tranformerer""")
    return


@app.cell
def _(organisert_df, pl):
    # Setter alle null verdier i observasjoner lik 1

    artsdata_df1 = organisert_df.with_columns(
        pl.col("Antall").fill_null(0))
    return (artsdata_df1,)


@app.cell
def _(artsdata_df1, pl):
    # Endrer fra komma til punktum seperasjon i lengdegrad og breddegrad

    artsdata_df = artsdata_df1.with_columns(
        pl.col(["latitude", "longitude"]).str.replace_all(",", ".").cast(pl.Float64)
    )
    return (artsdata_df,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Utforsker artsdata (innebygde marimo utforskere)""")
    return


@app.cell
def _(artsdata_df):
    artsdata_df
    return


@app.cell
def _(artsdata_df, mo):
    mo.ui.data_explorer(artsdata_df)
    return


@app.cell
def _(artsdata_df, mo):
    mo.ui.dataframe(artsdata_df, page_size=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Kart""")
    return


@app.cell
def _(artsdata_df, mo):
    # Husk at du må velge data for at kartene skal fungere!

    artsdata_kart_df = mo.ui.table(artsdata_df)
    artsdata_kart_df
    return (artsdata_kart_df,)


@app.cell(hide_code=True)
def _(artsdata_kart_df):
    artsdata_kart = artsdata_kart_df.value
    return (artsdata_kart,)


@app.cell(hide_code=True)
def _(mo):
    # Lager UI elementer for å velge kart
    map_style_dropdown = mo.ui.dropdown(
        options=[
            "carto-positron", 
            "carto-darkmatter", 
            "open-street-map", 
        ],
        value="carto-positron",
        label="Select a base map style:"
    )

    satellite_toggle = mo.ui.checkbox(value=True, label="Show Satellite Imagery")
    return map_style_dropdown, satellite_toggle


@app.cell(hide_code=True)
def _(map_style_dropdown, mo, satellite_toggle):
    controls = mo.vstack([map_style_dropdown, satellite_toggle])
    controls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Punktkart""")
    return


@app.cell(hide_code=True)
def _(artsdata_kart, map_style_dropdown, mo, px, satellite_toggle):
    fig = px.scatter_map(artsdata_kart,
                         lat="latitude",
                         lon="longitude",
                         color="Kategori",       
                         size="Antall", 
                         size_max=100,
                         zoom=10,
                         hover_name="Navn", 
                        )

    fig.update_layout(map_style=map_style_dropdown.value,
                        height=1000 )



    # Conditionally add the satellite layer based on the checkbox's value
    if satellite_toggle.value:
        fig.update_layout(
            map_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        )
    else:
        # An empty list removes any existing raster layers
        fig.update_layout(map_layers=[])



    satelitt_kart = mo.ui.plotly(fig)

    # Display the marimo element
    satelitt_kart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Hex-kart""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a dropdown to switch between count and sum
    aggregation_mode = mo.ui.dropdown(
        options=["Antall observasjoner", "Sum individer"],
        value="Antall observasjoner",
        label="Aggregation mode:"
    )
    aggregation_mode
    return (aggregation_mode,)


@app.cell(hide_code=True)
def _(
    aggregation_mode,
    artsdata_kart,
    ff,
    map_style_dropdown,
    mo,
    satellite_toggle,
):
    import numpy as np

    # Set parameters based on aggregation mode
    if aggregation_mode.value == "Antall observasjoner":
        color_param = None
        agg_func_param = None
        label_text = "Antall observasjoner"
    else:
        color_param = "Antall"
        agg_func_param = np.sum
        label_text = "Sum individer"

    # Create the hexbin map with conditional parameters
    fig_hex = ff.create_hexbin_mapbox(
        data_frame=artsdata_kart, 
        lat="latitude", 
        lon="longitude",
        color=color_param,  # None for count, "Antall" for sum
        nx_hexagon=10, 
        opacity=0.5, 
        labels={"color": label_text},
        min_count=1, 
        color_continuous_scale="Viridis",
        show_original_data=True,
        original_data_marker=dict(size=4, opacity=0.6, color="deeppink"),
        agg_func=agg_func_param  # None for count, np.sum for sum
    )

    # Apply map style settings
    fig_hex.update_layout(mapbox_style=map_style_dropdown.value, height=1000)

    # Conditionally add the satellite layer based on the checkbox's value
    if satellite_toggle.value:
        fig_hex.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        )
    else:
        fig_hex.update_layout(mapbox_layers=[])

    hekskart = mo.ui.plotly(fig_hex)
    hekskart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Tid""")
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Figurer""")
    return


@app.cell(hide_code=True)
def _(artsdata_df, mo):
    # Husk at du må velge 
    artsdata_figurer_df = mo.ui.table(artsdata_df)
    artsdata_figurer_df
    return (artsdata_figurer_df,)


@app.cell(hide_code=True)
def _(artsdata_figurer_df):
    artsdata_fg = artsdata_figurer_df.value
    return (artsdata_fg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Antall

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown_figurer = mo.ui.dropdown(options=["Antall individer", "Antall observasjoner", "Gjennomsnittelig antall individer pr. observasjon"], label="Select metric")
    dropdown_figurer
    return (dropdown_figurer,)


@app.cell(hide_code=True)
def _(artsdata_fg, dropdown_figurer, pl):
    # Calculate based on dropdown selection
    if dropdown_figurer.value == "Antall individer":
        # Sum of individuals per species
        result = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.col('Antall').sum().alias('Total'))
        )
        y_label = "Antall individer"
    elif dropdown_figurer.value == "Antall observasjoner":
        # Count of observations per species
        result = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.len().alias('Total'))
        )
        y_label = "Antall observasjoner"
    else:  # "Gjennomsnittelig antall individer pr. observasjon"
        # Average individuals per observation
        result = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.col('Antall').mean().alias('Total'))
        )
        y_label = "Gjennomsnitt individer per observasjon"

    return result, y_label


@app.cell
def _(alt, artsdata_fg, mo, pl, result, y_label):
    #Når du aggregerer dataene i cellen over, så tar du bare ut art og antall. Ikke noe av de andre kolonnene, må hente dette inn igjen. 

    # Henter kategori for de ulike artene 
    species_categories = (
        artsdata_fg
        .select(['Navn', 'Kategori'])
        .unique()
    )

    # Join category information to your result
    result_with_category = result.join(species_categories, on='Navn')

    # Define custom category order
    category_order = ['CR', 'EN', 'VU', 'NT', 'LC', 'NR', 'LO', 'Unknown']

    # Create a sorted dataframe with custom category order
    sorted_data = (
        result_with_category
        # Add a sort key based on category order
        .with_columns(
            pl.col('Kategori')
            .map_elements(lambda x: category_order.index(x) if x in category_order else len(category_order))
            .alias('category_sort_key')
        )
        # Sort by category order, then by Total descending
        .sort(['category_sort_key', 'Total'], descending=[False, True])
        .drop('category_sort_key')
    )

    # Create a list of species names in the desired order
    species_order = sorted_data['Navn'].to_list()

    # Create the chart with custom sort order
    figur_antall = mo.ui.altair_chart(
        alt.Chart(result_with_category)
        .mark_bar()
        .encode(
            x=alt.X('Navn', 
                    title='Art', 
                    sort=species_order,  # Use custom sort order
                    axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Total', title=y_label),
            color=alt.Color('Kategori', 
                           title='Kategori',
                           scale=alt.Scale(domain=category_order)),  # Also order legend
            tooltip=['Navn', 'Total', 'Kategori']
        )
        .properties(width=1500, height=400)
    )

    figur_antall
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Atferd""")
    return


@app.cell(hide_code=True)
def _(alt, artsdata_fg, mo):
    figur_atferd = mo.ui.altair_chart(alt.Chart(artsdata_fg).mark_bar().encode(
        x="Navn",
        y="Antall",
        color="Atferd",
        tooltip=["Navn", "Antall", "Atferd"]
    ).properties(width=1500, height=400))
    return (figur_atferd,)


@app.cell(hide_code=True)
def _(figur_atferd, mo):
    mo.vstack([figur_atferd, mo.ui.table(figur_atferd.value)])
    return


if __name__ == "__main__":
    app.run()
