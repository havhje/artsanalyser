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


@app.cell
def _(mo):
    artsdata_df = mo.sql(
        f"""
        SELECT * FROM 'C:/Users/havh/OneDrive - Multiconsult/Dokumenter/Kodeprosjekter/Artsdatabanken/Artsdatabanken/behandlede_data/**/*.csv';
        """
    )
    return (artsdata_df,)


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
    mo.md(r"""#### Antall""")
    return


@app.cell(hide_code=True)
def _(mo):
    # Cell 1: Create dropdowns (unchanged)
    metric_dropdown = mo.ui.dropdown(
        options=["Antall individer", "Antall observasjoner", "Gjennomsnittelig antall individer pr. observasjon"], 
        value="Antall individer",
        label="Velg metrikk"
    )

    grouping_dropdown = mo.ui.dropdown(
        options=["Art (kategori)", "Familie", "Orden"], 
        value="Art (kategori)",
        label="Sorter etter"
    )

    mo.vstack([metric_dropdown, grouping_dropdown])
    return grouping_dropdown, metric_dropdown


@app.cell(hide_code=True)
def _(artsdata_fg, metric_dropdown, pl):
    if metric_dropdown.value == "Antall individer":
        aggregated_data = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.col('Antall').sum().alias('Total'))
        )
        y_label = "Antall individer"
    elif metric_dropdown.value == "Antall observasjoner":
        aggregated_data = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.len().alias('Total'))
        )
        y_label = "Antall observasjoner"
    else:
        aggregated_data = (
            artsdata_fg
            .group_by('Navn')
            .agg(pl.col('Antall').mean().alias('Total'))
        )
        y_label = "Gjennomsnitt individer per observasjon"

    # Join with species information - INCLUDING THE SPECIAL CATEGORIES
    species_info = (
        artsdata_fg
        .select([
            'Navn', 'Kategori', 'Familie', 'Orden',
            'Ansvarsarter', 'Andre spesielt hensynskrevende arter', 'Prioriterte arter'
        ])
        .unique()
    )

    data_with_info = aggregated_data.join(species_info, on='Navn')
    return data_with_info, y_label


@app.cell(hide_code=True)
def _(data_with_info, grouping_dropdown, pl):
    # Cell 3: Sort data and calculate group statistics
    # Define sorting field based on dropdown
    if grouping_dropdown.value == "Art (kategori)":
        sort_field = 'Kategori'
        color_field = 'Kategori'
        color_title = 'Rødlistekategori'

        # Define explicit sort order for all possible categories
        # Norwegian Red List categories (IUCN)
        redlist_order = ['CR', 'EN', 'VU', 'NT', 'LC', 'DD', 'NR']

        # Alien species risk categories (Fremmede arter)
        alien_order = ['SE', 'HI', 'PH', 'LO', 'NK']

        # Other categories
        other_order = ['NA', 'Unknown']

        # Combined order: Red list first (most to least threatened), then alien species (highest to lowest risk), then others
        kategori_order = redlist_order + alien_order + other_order

        # Create a mapping for sort priority
        kategori_priority = {cat: i for i, cat in enumerate(kategori_order)}

        # Add sort priority column
        data_with_priority = data_with_info.with_columns(
            pl.col('Kategori').map_elements(
                lambda x: kategori_priority.get(x, 999), 
                return_dtype=pl.Int32
            ).alias('kategori_priority')
        )

        # Sort by category priority first, then by Total within each group
        sorted_data = data_with_priority.sort(['kategori_priority', 'Total'], descending=[False, True])

        # Remove the temporary priority column
        sorted_data = sorted_data.drop('kategori_priority')

    elif grouping_dropdown.value == "Familie":
        sort_field = 'Familie'
        color_field = 'Familie'
        color_title = 'Familie'
        # Sort alphabetically by Familie, then by Total within each group
        sorted_data = data_with_info.sort([sort_field, 'Total'], descending=[False, True])

    else:
        sort_field = 'Orden'
        color_field = 'Orden'
        color_title = 'Orden'
        # Sort alphabetically by Orden, then by Total within each group
        sorted_data = data_with_info.sort([sort_field, 'Total'], descending=[False, True])

    # Calculate group totals for annotations
    group_totals = (
        sorted_data
        .group_by(sort_field)
        .agg([
            pl.col('Total').sum().alias('GroupTotal'),
            pl.col('Navn').count().alias('SpeciesCount'),
            pl.col('Navn').first().alias('FirstSpecies'),  # To position the annotation
            pl.col('Navn').last().alias('LastSpecies')
        ])
    )

    # Add x-position for each species (for separator lines)
    sorted_data_with_pos = sorted_data.with_columns(
        pl.arange(0, sorted_data.height).alias('x_position')
    )

    # Find group boundaries for separator lines
    group_boundaries = (
        sorted_data_with_pos
        .group_by(sort_field)
        .agg(pl.col('x_position').max().alias('last_position'))
        .filter(pl.col('last_position') < sorted_data_with_pos.height - 1)  # Exclude last group
        .with_columns((pl.col('last_position') + 0.5).alias('separator_position'))
    )

    # Create species order for x-axis
    species_order = sorted_data['Navn'].to_list()

    # Get unique values for consistent color ordering
    if grouping_dropdown.value == "Art (kategori)":
        # Use the explicit order for categories
        unique_groups = [cat for cat in kategori_order if cat in sorted_data[sort_field].unique()]
    else:
        # Use alphabetical order for other groupings
        unique_groups = sorted_data[sort_field].unique().sort().to_list()
    return (
        color_field,
        color_title,
        kategori_order,
        sort_field,
        sorted_data,
        species_order,
        unique_groups,
    )


@app.cell(hide_code=True)
def _(
    alt,
    color_field,
    color_title,
    grouping_dropdown,
    kategori_order,
    sorted_data,
    unique_groups,
):
    # Cell 4: Define color schemes
    # Define color schemes based on grouping type
    if grouping_dropdown.value == "Art (kategori)":
        # Combined color scheme for both Red List and alien species
        kategori_colors = {
            # Red List categories (threat-based colors)
            'CR': '#d62728',  # Critically Endangered - Dark Red
            'EN': '#ff7f0e',  # Endangered - Orange
            'VU': '#ffbb78',  # Vulnerable - Light Orange
            'NT': '#aec7e8',  # Near Threatened - Light Blue
            'LC': '#2ca02c',  # Least Concern - Green
            'DD': '#c7c7c7',  # Data Deficient - Gray
            'NR': '#f7f7f7',  # Not Evaluated - Light Gray

            # Alien species categories (risk-based colors)
            'SE': '#8b0000',  # Severe impact - Dark Red
            'HI': '#ff1493',  # High impact - Deep Pink
            'PH': '#ff69b4',  # Potentially high impact - Hot Pink
            'LO': '#dda0dd',  # Low impact - Plum
            'NK': '#e6e6fa',  # No known impact - Lavender

            # Other categories
            'NA': '#b0b0b0',  # Not Applicable - Medium Gray
            'Unknown': '#888888'  # Unknown - Dark Gray
        }

        # Use kategori_order from Cell 3 to ensure consistent ordering
        # kategori_order is already defined in Cell 3

        # Get actual categories in the data, maintaining the defined order
        actual_categories = sorted_data['Kategori'].unique().to_list()
        color_domain = [cat for cat in kategori_order if cat in actual_categories]
        color_range = [kategori_colors[cat] for cat in color_domain]

        color_scale = alt.Scale(domain=color_domain, range=color_range)
        legend_sort = color_domain  # Explicit sort order for legend
    else:
        # Use default color scheme for Familie and Orden
        color_scale = alt.Scale(scheme='category20' if len(unique_groups) > 10 else 'category10')
        legend_sort = unique_groups  # Alphabetical order from Cell 3

    # Create color encoding with explicit sort order
    color_encoding = alt.Color(
        color_field, 
        title=color_title,
        scale=color_scale,
        sort=legend_sort,  # Use explicit sort order for legend
        legend=alt.Legend(orient='right', titleLimit=200)
    )
    return (color_encoding,)


@app.cell(hide_code=True)
def _(
    alt,
    color_encoding,
    metric_dropdown,
    mo,
    pl,
    sort_field,
    sorted_data,
    species_order,
    y_label,
):
    # --- 1. Initial Setup (similar to your original code) ---

    # Calculate dynamic bar width based on number of species
    num_species = sorted_data.height
    bar_width = max(0.5, min(0.9, 30 / num_species))

    # Calculate a base marker offset (e.g., 5% of the max value)
    max_value = sorted_data['Total'].max()
    marker_offset = max_value * 0.05 if max_value > 0 else 1

    # Base chart with bars (no changes here)
    bars = (
        alt.Chart(sorted_data)
        .mark_bar(width=alt.RelativeBandSize(bar_width))
        .encode(
            x=alt.X('Navn', 
                    title='Art', 
                    sort=species_order,
                    axis=alt.Axis(labelAngle=-45, labelLimit=200, labelOverlap=False)),
            y=alt.Y('Total', title=y_label, scale=alt.Scale(domain=[0, max_value * 1.2])), # Extend domain to make space
            color=color_encoding,
            tooltip=[
                alt.Tooltip('Navn', title='Art'),
                alt.Tooltip('Total', title=y_label, format='.2f' if 'Gjennomsnitt' in y_label else '.0f'),
                alt.Tooltip('Kategori', title='Rødlistestatus'),
                alt.Tooltip('Familie', title='Familie'),
                alt.Tooltip('Orden', title='Orden'),
                alt.Tooltip('Ansvarsarter', title='Ansvarsart'),
                alt.Tooltip('Andre spesielt hensynskrevende arter', title='Hensynskrevende'),
                alt.Tooltip('Prioriterte arter', title='Prioritert')
            ]
        )
    )

    # --- 2. Data Transformation for Markers (Corrected) ---

    # Define the columns that represent marker categories
    marker_cols = ['Ansvarsarter', 'Andre spesielt hensynskrevende arter', 'Prioriterte arter']

    # Reshape the data from wide to long format for markers
    # This is the key step for creating a legend automatically
    # NOTE: .melt() is replaced with .unpivot() and arguments are updated
    marker_data = (
        sorted_data
        .filter(pl.any_horizontal(pl.col(c) for c in marker_cols)) # Keep only rows with at least one special status
        .unpivot( # <-- Changed from .melt()
            index=['Navn', 'Total'],         # <-- Changed from id_vars
            on=marker_cols,                  # <-- Changed from value_vars
            variable_name='Status', 
            value_name='Is_True'
        )
        .filter(pl.col('Is_True')) # Keep only the True values
    )


    # --- 3. Create the Improved Marker Layer (Single Black & Hollow Legend) ---

    # Check if there is any marker data to plot
    if marker_data.height > 0:
        markers = (
            alt.Chart(marker_data)
            .mark_point(
                size=50,         # A good size for visibility
                filled=False,     # Makes the markers hollow
                stroke='black',   # Statically sets the outline color to black for all markers
                strokeWidth=0.5   # A thicker outline is easier to see
            )
            .encode(
                x=alt.X('Navn:N', sort=species_order),
                y=alt.Y('y_pos:Q', axis=None), 
            
                # --- SHAPE ENCODING ---
                # This is now the ONLY encoding that will generate a legend for the markers.
                shape=alt.Shape('Status:N', 
                                scale=alt.Scale(
                                    domain=marker_cols, 
                                    range=['circle', 'square', 'triangle-up']
                                ),
                                # Configure the legend title
                                legend=alt.Legend(title="Forvaltningsinteresse")),
            
                # The color encoding has been removed!
            
                tooltip=[
                    alt.Tooltip('Navn', title='Art'),
                    alt.Tooltip('Status', title='Status')
                ]
            )
            .transform_window(
                # For each species ('Navn'), assign a rank to its statuses for stacking.
                marker_rank='rank()',
                groupby=['Navn']
            )
            .transform_calculate(
                # Calculate the y-position to stack markers above the bars.
                y_pos=f'datum.Total + {marker_offset} * datum.marker_rank'
            )
        )
    
        # Layer the bars and the markers together
        chart = bars + markers
    else:
        # If no marker data exists, just show the bars
        chart = bars

    # --- 4. Final Chart Configuration ---
    # Your existing code for this section is fine, but you might want to ensure
    # the legend symbols are styled correctly. You can add a .configure_legend()
    # call to be explicit.

    final_chart = (
        chart
        .properties(
            width=1200, 
            height=500,
            title=f"{metric_dropdown.value} sortert etter {sort_field.lower()}"
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12
        )
        .configure_title(
            fontSize=16,
            anchor='start'
        )
        .configure_legend(
            titleFontSize=12,
            labelFontSize=11,
            orient='right', # Place legend on the right side
            # Explicitly make legend symbols hollow with a black outline
            symbolFillColor='transparent',
            symbolStrokeColor='black'
        )
    )

    interactive_chart = mo.ui.altair_chart(final_chart)
    interactive_chart


    # --- 4. Final Chart Configuration ---

    final_chart = (
        chart
        .properties(
            width=1200, 
            height=500,
            title=f"{metric_dropdown.value} sortert etter {sort_field.lower()}"
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=12
        )
        .configure_title(
            fontSize=16,
            anchor='start'
        )
        .configure_legend(
            titleFontSize=12,
            labelFontSize=11,
            orient='right' # Place legend on the right side
        )
    )


    interactive_chart = mo.ui.altair_chart(final_chart)
    interactive_chart
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
