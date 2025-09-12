import marimo

__generated_with = "0.14.17"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.figure_factory as ff
    import polars as pl
    import requests
    import json
    import time
    import tempfile
    import os
    import duckdb
    return alt, duckdb, ff, json, mo, os, pd, pl, px, requests, tempfile, time


@app.cell(hide_code=True)
def _(mo):
    valgt_fil = mo.ui.file_browser(initial_path="/Users/havardhjermstad-sollerud/Downloads")
    valgt_fil
    return (valgt_fil,)


@app.cell(hide_code=True)
def _(valgt_fil):
    file_info = valgt_fil.value[0]
    filepath = file_info.path
    str(filepath)
    return (filepath,)


@app.cell
def _(filepath, mo):
    arter_df = mo.sql(
        f"""
        select * from read_csv('{filepath}')
        """
    )
    return (arter_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Kart""")
    return


@app.cell(hide_code=True)
def _(artsdata_df):
    artsdata_kart = artsdata_df.value
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
        label="Select a base map style:",
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
    fig = px.scatter_map(
        artsdata_kart,
        lat="latitude",
        lon="longitude",
        color="Kategori",
        size="Antall",
        size_max=100,
        zoom=10,
        hover_name="Navn",
    )

    fig.update_layout(map_style=map_style_dropdown.value, height=1000)

    # Conditionally add the satellite layer based on the checkbox's value
    if satellite_toggle.value:
        fig.update_layout(
            map_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
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
        label="Aggregation mode:",
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
        agg_func=agg_func_param,  # None for count, np.sum for sum
    )

    # Apply map style settings
    fig_hex.update_layout(mapbox_style=map_style_dropdown.value, height=1000)

    # Conditionally add the satellite layer based on the checkbox's value
    if satellite_toggle.value:
        fig_hex.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
        )
    else:
        fig_hex.update_layout(mapbox_layers=[])

    hekskart = mo.ui.plotly(fig_hex)
    hekskart
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""### Utforsker artsdata (innebygde marimo utforskere)""")
    return


@app.cell
def _(arter_df, mo):
    artsdata_df = mo.ui.table(arter_df, page_size=50)
    artsdata_df
    return (artsdata_df,)


@app.cell
def _(artsdata_df, mo):
    mo.ui.data_explorer(artsdata_df.value)
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""### Tid""")
    return


@app.cell(hide_code=True)
def _(artsdata_df):
    artsdata_tid = artsdata_df.value
    return (artsdata_tid,)


@app.cell(hide_code=True)
def _(alt, artsdata_tid, mo, pl):
    # Group by date and sum the number of individuals
    individuals_by_date = (
        artsdata_tid.group_by(pl.col("Observert dato").dt.date().alias("date"))
        .agg(
            [
                pl.len().alias("observation_count"),  # Using pl.len() as requested
                pl.col("Antall").sum().alias("individual_count"),  # Sum of individuals
            ]
        )
        .sort("date")
    )

    # Create the Altair chart for individuals
    chart_tid = (
        alt.Chart(individuals_by_date)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Dato", axis=alt.Axis(format="%Y")),
            y=alt.Y("individual_count:Q", title="Antall individer"),
            tooltip=[
                alt.Tooltip("date:T", title="Dato", format="%d %B %Y"),
                alt.Tooltip("individual_count:Q", title="Antall individer"),
                alt.Tooltip("observation_count:Q", title="Antall observasjoner"),
            ],
        )
        .properties(width=900, height=400, title="Antall individer observert over tid")
        .interactive()
    )

    mo.ui.altair_chart(chart_tid)
    return


@app.cell(hide_code=True)
def _(mo):
    toggle = mo.ui.switch(label="Individer", value=False)
    window_size = mo.ui.slider(start=1, stop=30, step=1, show_value=True, label="Antall dager i rullende gj.snitt")
    window_size
    return toggle, window_size


@app.cell(hide_code=True)
def _(alt, artsdata_tid, mo, pl, toggle, window_size):
    daily_stats = (
        artsdata_tid.with_columns(
            [
                pl.col("Observert dato").dt.date().alias("date"),
                pl.col("Observert dato").dt.year().alias("year"),
                pl.col("Antall").cast(pl.Int64, strict=False).fill_null(1).alias("ind_count"),
            ]
        )
        .group_by("date")
        .agg(
            [
                pl.len().alias("daily_obs_count"),
                pl.col("ind_count").sum().alias("daily_ind_count"),
                pl.col("year").first().alias("year"),
            ]
        )
        # Create a synthetic date using year 2024 for all data to show pattern
        .with_columns([pl.date(2024, pl.col("date").dt.month(), pl.col("date").dt.day()).alias("common_date")])
        .group_by("common_date")
        .agg(
            [
                # For observations
                pl.col("daily_obs_count").mean().alias("avg_daily_obs"),
                pl.col("daily_obs_count").std().alias("std_daily_obs"),
                # For individuals
                pl.col("daily_ind_count").mean().alias("avg_daily_ind"),
                pl.col("daily_ind_count").std().alias("std_daily_ind"),
                # Count years
                pl.col("daily_obs_count").count().alias("n_years"),
            ]
        )
        .sort("common_date")
        .with_columns(
            [
                # Rolling averages
                pl.col("avg_daily_obs").rolling_mean(window_size.value, center=True).alias("rolling_avg_obs"),
                pl.col("avg_daily_ind").rolling_mean(window_size.value, center=True).alias("rolling_avg_ind"),
                # Standard errors
                (pl.col("std_daily_obs") / pl.col("n_years").sqrt()).alias("se_obs"),
                (pl.col("std_daily_ind") / pl.col("n_years").sqrt()).alias("se_ind"),
            ]
        )
        .with_columns(
            [
                # Confidence bands
                (pl.col("rolling_avg_obs") - pl.col("se_obs")).alias("lower_obs"),
                (pl.col("rolling_avg_obs") + pl.col("se_obs")).alias("upper_obs"),
                (pl.col("rolling_avg_ind") - pl.col("se_ind")).alias("lower_ind"),
                (pl.col("rolling_avg_ind") + pl.col("se_ind")).alias("upper_ind"),
            ]
        )
    )

    # Create observations chart
    obs_chart = (
        (
            alt.Chart(daily_stats)
            .mark_area(opacity=0.3, color="lightblue")
            .encode(
                x=alt.X("common_date:T", title="Dato", axis=alt.Axis(format="%d %b")),
                y=alt.Y("lower_obs:Q", title="Rullerende gjennomsnitt (observasjoner)"),
                y2="upper_obs:Q",
            )
            + alt.Chart(daily_stats)
            .mark_line(point=True, size=2, color="steelblue")
            .encode(
                x="common_date:T",
                y="rolling_avg_obs:Q",
                tooltip=[
                    alt.Tooltip("common_date:T", title="Dato", format="%d %B"),
                    alt.Tooltip("rolling_avg_obs:Q", title="Rullerende gjennomsnitt", format=".1f"),
                    alt.Tooltip("se_obs:Q", title="Standardfeil", format=".2f"),
                ],
            )
        )
        .properties(width=900, height=400, title="Observasjoner")
        .interactive()
    )

    # Create individuals chart
    ind_chart = (
        (
            alt.Chart(daily_stats)
            .mark_area(opacity=0.3, color="peachpuff")
            .encode(
                x=alt.X("common_date:T", title="Dato", axis=alt.Axis(format="%d %b")),
                y=alt.Y("lower_ind:Q", title="Rullerende gjennomsnitt (individer)"),
                y2="upper_ind:Q",
            )
            + alt.Chart(daily_stats)
            .mark_line(point={"filled": True, "fill": "darkorange", "size": 20}, size=2, color="darkorange")
            .encode(
                x="common_date:T",
                y="rolling_avg_ind:Q",
                tooltip=[
                    alt.Tooltip("common_date:T", title="Dato", format="%d %B"),
                    alt.Tooltip("rolling_avg_ind:Q", title="Rullerende gjennomsnitt", format=".1f"),
                    alt.Tooltip("se_ind:Q", title="Standardfeil", format=".2f"),
                ],
            )
        )
        .properties(width=900, height=800, title="Individer")
        .interactive()
    )

    # Display toggle and appropriate chart
    mo.vstack([toggle, ind_chart if toggle.value else obs_chart])
    return


@app.cell(column=3, hide_code=True)
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
        label="Velg metrikk",
    )

    grouping_dropdown = mo.ui.dropdown(
        options=["Art (kategori)", "Familie", "Orden"], value="Art (kategori)", label="Sorter etter"
    )

    mo.vstack([metric_dropdown, grouping_dropdown])
    return grouping_dropdown, metric_dropdown


@app.cell(hide_code=True)
def _(artsdata_fg, metric_dropdown, pl):
    if metric_dropdown.value == "Antall individer":
        aggregated_data = artsdata_fg.group_by("Navn").agg(pl.col("Antall").sum().alias("Total"))
        y_label = "Antall individer"
    elif metric_dropdown.value == "Antall observasjoner":
        aggregated_data = artsdata_fg.group_by("Navn").agg(pl.len().alias("Total"))
        y_label = "Antall observasjoner"
    else:
        aggregated_data = artsdata_fg.group_by("Navn").agg(pl.col("Antall").mean().alias("Total"))
        y_label = "Gjennomsnitt individer per observasjon"

    # Join with species information - INCLUDING THE SPECIAL CATEGORIES
    species_info = artsdata_fg.select(
        [
            "Navn",
            "Kategori",
            "Familie",
            "Orden",
            "Ansvarsarter",
            "Andre spesielt hensynskrevende arter",
            "Prioriterte arter",
        ]
    ).unique()

    data_with_info = aggregated_data.join(species_info, on="Navn")
    return data_with_info, y_label


@app.cell(hide_code=True)
def _(data_with_info, grouping_dropdown, pl):
    # Cell 3: Sort data and calculate group statistics
    # Define sorting field based on dropdown
    if grouping_dropdown.value == "Art (kategori)":
        sort_field = "Kategori"
        color_field = "Kategori"
        color_title = "Rødlistekategori"

        # Define explicit sort order for all possible categories
        # Norwegian Red List categories (IUCN)
        redlist_order = ["CR", "EN", "VU", "NT", "LC", "DD", "NR"]

        # Alien species risk categories (Fremmede arter)
        alien_order = ["SE", "HI", "PH", "LO", "NK"]

        # Other categories
        other_order = ["NA", "Unknown"]

        # Combined order: Red list first (most to least threatened), then alien species (highest to lowest risk), then others
        kategori_order = redlist_order + alien_order + other_order

        # Create a mapping for sort priority
        kategori_priority = {cat: i for i, cat in enumerate(kategori_order)}

        # Add sort priority column
        data_with_priority = data_with_info.with_columns(
            pl.col("Kategori")
            .map_elements(lambda x: kategori_priority.get(x, 999), return_dtype=pl.Int32)
            .alias("kategori_priority")
        )

        # Sort by category priority first, then by Total within each group
        sorted_data = data_with_priority.sort(["kategori_priority", "Total"], descending=[False, True])

        # Remove the temporary priority column
        sorted_data = sorted_data.drop("kategori_priority")

    elif grouping_dropdown.value == "Familie":
        sort_field = "Familie"
        color_field = "Familie"
        color_title = "Familie"
        # Sort alphabetically by Familie, then by Total within each group
        sorted_data = data_with_info.sort([sort_field, "Total"], descending=[False, True])

    else:
        sort_field = "Orden"
        color_field = "Orden"
        color_title = "Orden"
        # Sort alphabetically by Orden, then by Total within each group
        sorted_data = data_with_info.sort([sort_field, "Total"], descending=[False, True])

    # Calculate group totals for annotations
    group_totals = sorted_data.group_by(sort_field).agg(
        [
            pl.col("Total").sum().alias("GroupTotal"),
            pl.col("Navn").count().alias("SpeciesCount"),
            pl.col("Navn").first().alias("FirstSpecies"),  # To position the annotation
            pl.col("Navn").last().alias("LastSpecies"),
        ]
    )

    # Add x-position for each species (for separator lines)
    sorted_data_with_pos = sorted_data.with_columns(pl.arange(0, sorted_data.height).alias("x_position"))

    # Find group boundaries for separator lines
    group_boundaries = (
        sorted_data_with_pos.group_by(sort_field)
        .agg(pl.col("x_position").max().alias("last_position"))
        .filter(pl.col("last_position") < sorted_data_with_pos.height - 1)  # Exclude last group
        .with_columns((pl.col("last_position") + 0.5).alias("separator_position"))
    )

    # Create species order for x-axis
    species_order = sorted_data["Navn"].to_list()

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
            "CR": "#d62728",  # Critically Endangered - Dark Red
            "EN": "#ff7f0e",  # Endangered - Orange
            "VU": "#ffbb78",  # Vulnerable - Light Orange
            "NT": "#aec7e8",  # Near Threatened - Light Blue
            "LC": "#2ca02c",  # Least Concern - Green
            "DD": "#c7c7c7",  # Data Deficient - Gray
            "NR": "#f7f7f7",  # Not Evaluated - Light Gray
            # Alien species categories (risk-based colors)
            "SE": "#8b0000",  # Severe impact - Dark Red
            "HI": "#ff1493",  # High impact - Deep Pink
            "PH": "#ff69b4",  # Potentially high impact - Hot Pink
            "LO": "#dda0dd",  # Low impact - Plum
            "NK": "#e6e6fa",  # No known impact - Lavender
            # Other categories
            "NA": "#b0b0b0",  # Not Applicable - Medium Gray
            "Unknown": "#888888",  # Unknown - Dark Gray
        }

        # Use kategori_order from Cell 3 to ensure consistent ordering
        # kategori_order is already defined in Cell 3

        # Get actual categories in the data, maintaining the defined order
        actual_categories = sorted_data["Kategori"].unique().to_list()
        color_domain = [cat for cat in kategori_order if cat in actual_categories]
        color_range = [kategori_colors[cat] for cat in color_domain]

        color_scale = alt.Scale(domain=color_domain, range=color_range)
        legend_sort = color_domain  # Explicit sort order for legend
    else:
        # Use default color scheme for Familie and Orden
        color_scale = alt.Scale(scheme="category20" if len(unique_groups) > 10 else "category10")
        legend_sort = unique_groups  # Alphabetical order from Cell 3

    # Create color encoding with explicit sort order
    color_encoding = alt.Color(
        color_field,
        title=color_title,
        scale=color_scale,
        sort=legend_sort,  # Use explicit sort order for legend
        legend=alt.Legend(orient="right", titleLimit=200),
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
    max_value = sorted_data["Total"].max()
    marker_offset = max_value * 0.05 if max_value > 0 else 1

    # Base chart with bars (no changes here)
    bars = (
        alt.Chart(sorted_data)
        .mark_bar(width=alt.RelativeBandSize(bar_width))
        .encode(
            x=alt.X(
                "Navn",
                title="Art",
                sort=species_order,
                axis=alt.Axis(labelAngle=-45, labelLimit=200, labelOverlap=False),
            ),
            y=alt.Y(
                "Total", title=y_label, scale=alt.Scale(domain=[0, max_value * 1.2])
            ),  # Extend domain to make space
            color=color_encoding,
            tooltip=[
                alt.Tooltip("Navn", title="Art"),
                alt.Tooltip("Total", title=y_label, format=".2f" if "Gjennomsnitt" in y_label else ".0f"),
                alt.Tooltip("Kategori", title="Rødlistestatus"),
                alt.Tooltip("Familie", title="Familie"),
                alt.Tooltip("Orden", title="Orden"),
                alt.Tooltip("Ansvarsarter", title="Ansvarsart"),
                alt.Tooltip("Andre spesielt hensynskrevende arter", title="Hensynskrevende"),
                alt.Tooltip("Prioriterte arter", title="Prioritert"),
            ],
        )
    )

    # --- 2. Data Transformation for Markers (Corrected) ---

    # Define the columns that represent marker categories
    marker_cols = ["Ansvarsarter", "Andre spesielt hensynskrevende arter", "Prioriterte arter"]

    # Reshape the data from wide to long format for markers
    # This is the key step for creating a legend automatically
    # NOTE: .melt() is replaced with .unpivot() and arguments are updated
    marker_data = (
        sorted_data.filter(
            pl.any_horizontal(pl.col(c) for c in marker_cols)
        )  # Keep only rows with at least one special status
        .unpivot(  # <-- Changed from .melt()
            index=["Navn", "Total"],  # <-- Changed from id_vars
            on=marker_cols,  # <-- Changed from value_vars
            variable_name="Status",
            value_name="Is_True",
        )
        .filter(pl.col("Is_True"))  # Keep only the True values
    )

    # --- 3. Create the Improved Marker Layer (Single Black & Hollow Legend) ---

    # Check if there is any marker data to plot
    if marker_data.height > 0:
        markers = (
            alt.Chart(marker_data)
            .mark_point(
                size=50,  # A good size for visibility
                filled=False,  # Makes the markers hollow
                stroke="black",  # Statically sets the outline color to black for all markers
                strokeWidth=0.5,  # A thicker outline is easier to see
            )
            .encode(
                x=alt.X("Navn:N", sort=species_order),
                y=alt.Y("y_pos:Q", axis=None),
                # --- SHAPE ENCODING ---
                # This is now the ONLY encoding that will generate a legend for the markers.
                shape=alt.Shape(
                    "Status:N",
                    scale=alt.Scale(domain=marker_cols, range=["circle", "square", "triangle-up"]),
                    # Configure the legend title
                    legend=alt.Legend(title="Forvaltningsinteresse"),
                ),
                # The color encoding has been removed!
                tooltip=[alt.Tooltip("Navn", title="Art"), alt.Tooltip("Status", title="Status")],
            )
            .transform_window(
                # For each species ('Navn'), assign a rank to its statuses for stacking.
                marker_rank="rank()",
                groupby=["Navn"],
            )
            .transform_calculate(
                # Calculate the y-position to stack markers above the bars.
                y_pos=f"datum.Total + {marker_offset} * datum.marker_rank"
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
        chart.properties(width=1200, height=500, title=f"{metric_dropdown.value} sortert etter {sort_field.lower()}")
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=16, anchor="start")
        .configure_legend(
            titleFontSize=12,
            labelFontSize=11,
            orient="right",  # Place legend on the right side
            # Explicitly make legend symbols hollow with a black outline
            symbolFillColor="transparent",
            symbolStrokeColor="black",
        )
    )

    interactive_chart = mo.ui.altair_chart(final_chart)
    interactive_chart

    # --- 4. Final Chart Configuration ---

    final_chart = (
        chart.properties(width=1200, height=500, title=f"{metric_dropdown.value} sortert etter {sort_field.lower()}")
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=16, anchor="start")
        .configure_legend(
            titleFontSize=12,
            labelFontSize=11,
            orient="right",  # Place legend on the right side
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
    figur_atferd = mo.ui.altair_chart(
        alt.Chart(artsdata_fg)
        .mark_bar()
        .encode(x="Navn", y="Antall", color="Atferd", tooltip=["Navn", "Antall", "Atferd"])
        .properties(width=1500, height=400)
    )
    return (figur_atferd,)


@app.cell(hide_code=True)
def _(figur_atferd, mo):
    mo.vstack([figur_atferd, mo.ui.table(figur_atferd.value)])
    return


@app.cell(column=4, hide_code=True)
def _(mo):
    mo.md(r"""## Overlagsanalyse mot hovedøkosystemkartet""")
    return


@app.cell(hide_code=True)
def _(arter_df, mo, pd):
    # Extract UTM coordinates from geometry column
    coords_utm = mo.sql("""
        SELECT 
            TRY_CAST(regexp_extract(geometry, 'POINT \\(([0-9.]+)', 1) AS DOUBLE) as x,
            TRY_CAST(regexp_extract(geometry, 'POINT \\([0-9.]+ ([0-9.]+)', 1) AS DOUBLE) as y
        FROM arter_df
        WHERE geometry IS NOT NULL 
            AND geometry != ''
            AND geometry LIKE 'POINT%'
    """)

    # Calculate bounding box with 10% buffer
    bbox_utm = mo.sql("""
        SELECT 
            MIN(x) - (MAX(x) - MIN(x)) * 0.1 as xmin,
            MAX(x) + (MAX(x) - MIN(x)) * 0.1 as xmax,
            MIN(y) - (MAX(y) - MIN(y)) * 0.1 as ymin,
            MAX(y) + (MAX(y) - MIN(y)) * 0.1 as ymax,
            (MAX(x) - MIN(x)) * (MAX(y) - MIN(y)) / 1000000 as area_km2
        FROM coords_utm
        WHERE x IS NOT NULL AND y IS NOT NULL
    """)

    # Convert to pandas DataFrame and extract values
    if hasattr(bbox_utm, "to_pandas"):
        bbox_df = bbox_utm.to_pandas()
    else:
        bbox_df = pd.DataFrame(bbox_utm)

    # Extract values from the DataFrame
    xmin = float(bbox_df["xmin"].values[0])
    xmax = float(bbox_df["xmax"].values[0])
    ymin = float(bbox_df["ymin"].values[0])
    ymax = float(bbox_df["ymax"].values[0])
    area_km2 = float(bbox_df["area_km2"].values[0])

    mo.md(f"""
    ### Bounding Box UTM Zone 33N (EPSG:25833)
    - **X Range:** {xmin:.0f} - {xmax:.0f}
    - **Y Range:** {ymin:.0f} - {ymax:.0f}
    - **Area:** {area_km2:.1f} km²
    """)
    return area_km2, coords_utm, xmax, xmin, ymax, ymin


@app.cell(hide_code=True)
def _(
    area_km2,
    map_style_dropdown,
    mo,
    satellite_toggle,
    xmax,
    xmin,
    ymax,
    ymin,
):
    # Convert UTM bounding box corners to lat/lon for map display
    import pyproj
    import plotly.graph_objects as go


    # Create transformer from UTM Zone 33N to WGS84
    transformer = pyproj.Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

    # Convert bounding box corners to lat/lon
    bbox_corners_utm = [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin)  # Close the polygon
    ]

    bbox_lons = []
    bbox_lats = []
    for x, y in bbox_corners_utm:
        lon, lat = transformer.transform(x, y)
        bbox_lons.append(lon)
        bbox_lats.append(lat)

    # Create plotly map figure
    fig_map_bbox = go.Figure()

    # Add the bounding box as a polygon on the map
    fig_map_bbox.add_trace(go.Scattermap(
        mode='lines',
        lon=bbox_lons,
        lat=bbox_lats,
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(width=3, color='red'),
        name='Bounding Box',
        text=f'Area: {area_km2:.1f} km²'
    ))

    # Calculate center of bounding box for map centering
    center_lon = sum(bbox_lons[:-1]) / 4
    center_lat = sum(bbox_lats[:-1]) / 4

    # Update layout with map settings
    fig_map_bbox.update_layout(
        map=dict(
            style=map_style_dropdown.value,
            center=dict(lat=center_lat, lon=center_lon),
            zoom=9
        ),
        height=700,
        title='UTM Zone 33N Bounding Box on Map',
        showlegend=True
    )

    # Add satellite imagery if toggle is on
    if satellite_toggle.value:
        fig_map_bbox.update_layout(
            map_style="white-bg",
            map_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "source": [
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
        )

    mo.ui.plotly(fig_map_bbox)
    return


@app.cell(hide_code=True)
def _():
    service_url = "https://kart2.miljodirektoratet.no/arcgis/rest/services/hovedokosystem/hovedokosystem/MapServer"
    return (service_url,)


@app.cell(hide_code=True)
def _(mo, requests, service_url, time, xmax, xmin, ymax, ymin):
    # If envelope works better, here's an updated download function
    def download_arcgis_utm33_envelope(service_url, layer_id, xmin, ymin, xmax, ymax, max_records=2000):
        """
        Download GeoJSON data using envelope geometry
        """
        base_url = f"{service_url}/{layer_id}/query"

        # Use envelope format for the geometry
        base_params = {
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "25833",
            "outSR": "25833",
            "where": "1=1",
            "f": "json"
        }

        # Get total count first
        count_params = {**base_params, "returnCountOnly": "true"}

        try:
            response = requests.get(base_url, params=count_params)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                mo.md(f"**Service error:** {result['error']}")
                return {"type": "FeatureCollection", "features": []}

            total_count = result.get("count", 0)
            mo.md(f"**Found {total_count} features in the bounding box**")

            if total_count == 0:
                return {"type": "FeatureCollection", "features": []}

        except requests.exceptions.RequestException as e:
            mo.md(f"**Error querying service:** {str(e)}")
            return {"type": "FeatureCollection", "features": []}

        # Download features with pagination
        all_features = []
        offset = 0

        while offset < total_count:
            query_params = {
                **base_params,
                "outFields": "*",
                "returnGeometry": "true",
                "resultOffset": offset,
                "resultRecordCount": min(max_records, total_count - offset),
                "f": "geojson"
            }

            try:
                response = requests.get(base_url, params=query_params)
                response.raise_for_status()

                geojson_data = response.json()
                features = geojson_data.get("features", [])
                all_features.extend(features)

                downloaded = len(features)
                offset += downloaded

                mo.md(f"Progress: {offset}/{total_count} features downloaded")

                if downloaded == 0:
                    break

                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                mo.md(f"**Error downloading batch at offset {offset}:** {str(e)}")
                break

        return {"type": "FeatureCollection", "features": all_features}

    # Try the envelope-based download
    ecosystem_geojson_envelope = download_arcgis_utm33_envelope(
        service_url, 0, xmin, ymin, xmax, ymax
    )

    mo.md(f"""### Ecosystem Data Download (Envelope Method)
    Downloaded **{len(ecosystem_geojson_envelope["features"])}** ecosystem polygons
    """)
    return (ecosystem_geojson_envelope,)


@app.cell
def _(ecosystem_geojson_envelope, json, tempfile):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(ecosystem_geojson_envelope, f)
        temp_geojson_path = f.name
    return (temp_geojson_path,)


@app.cell
def _(os):
    os.environ['OGR_GEOJSON_MAX_OBJ_SIZE'] = '0'
    return


@app.cell
def _(arter_df, mo, temp_geojson_path):
    _df = mo.sql(
        f"""
        -- Install the spatial extension
        INSTALL spatial;

        -- Load the spatial extension
        LOAD spatial;

        -- Load ecosystem data from the temp GeoJSON file
        CREATE OR REPLACE TABLE ecosystem_types AS
        SELECT * FROM ST_Read('{temp_geojson_path}');

        -- Extract point geometries from arter_df using lat/long columns
        CREATE OR REPLACE TABLE species_points AS
        SELECT 
            *,
            ST_Point(longitude, latitude) as geom_point
        FROM arter_df
        WHERE latitude IS NOT NULL 
            AND longitude IS NOT NULL;

        -- Perform spatial join to count species per ecosystem type
        CREATE OR REPLACE TABLE species_per_ecosystem AS
        SELECT 
            e.OBJECTID,
            e.ecotype as ecosystem_type,
            COUNT(DISTINCT s.Navn) as species_count,
            COUNT(*) as observation_count,
            SUM(CAST(s.Antall AS INTEGER)) as total_individuals,
            ST_Area(e.geom) / 1000000 as area_km2
        FROM ecosystem_types e
        LEFT JOIN species_points s 
            ON ST_Within(s.geom_point, e.geom)
        GROUP BY e.OBJECTID, e.ecotype, e.geom
        ORDER BY species_count DESC;

        -- Summary by ecosystem type
        SELECT 
            ecosystem_type,
            COUNT(DISTINCT OBJECTID) as polygon_count,
            SUM(species_count) as total_species,
            SUM(observation_count) as total_observations,
            SUM(total_individuals) as total_individuals,
            SUM(area_km2) as total_area_km2,
            ROUND(SUM(species_count) / SUM(area_km2), 2) as species_per_km2
        FROM species_per_ecosystem
        WHERE ecosystem_type IS NOT NULL
        GROUP BY ecosystem_type
        ORDER BY total_species DESC;
        """
    )
    return ecosystem_types, species_per_ecosystem, species_points


@app.cell
def _(columns, information_schema, mo):
    # First, let's see what columns are actually in the ecosystem data
    ecosystem_columns = mo.sql("""
        -- Check the column names in the ecosystem_types table
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'ecosystem_types'
    """)
    ecosystem_columns
    return


@app.cell
def _(duckdb, temp_geojson_path):
    # Create a connection and install/load the spatial extension
    conn = duckdb.connect(':memory:')
    conn.execute("INSTALL spatial")
    conn.execute("LOAD spatial")

    # Now read the GeoJSON file using DuckDB's spatial capabilities
    ecosystem_df = conn.execute(f"""
        SELECT * FROM ST_Read('{temp_geojson_path}')
    """).df()

    # Register the ecosystem dataframe with the existing DuckDB connection
    conn.register('ecosystem_data', ecosystem_df)

    return


if __name__ == "__main__":
    app.run()
