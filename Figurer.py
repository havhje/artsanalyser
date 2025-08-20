import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import plotly.express as px
    import plotly.figure_factory as ff
    import polars as pl

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
    artsdata_df = mo.sql(
        f"""
        SELECT * FROM read_csv('{str(filepath)}');
        """,
        output=False
    )
    return (artsdata_df,)


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
    mo.md(r"""### Antall""")
    return


@app.cell
def _(mo):
    mo.md(r"""#### Artsgrupper""")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""####Arter""")
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


@app.cell
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

    # Base chart with bars
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
                "Total", 
                title=y_label, 
                scale=alt.Scale(domain=[0, max_value * 1.2])
            ),
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

    # --- 2. Data Transformation for Markers ---
    marker_cols = ["Ansvarsarter", "Andre spesielt hensynskrevende arter", "Prioriterte arter"]

    marker_data = (
        sorted_data.filter(
            pl.any_horizontal(pl.col(c) for c in marker_cols)
        )
        .unpivot(
            index=["Navn", "Total"],
            on=marker_cols,
            variable_name="Status",
            value_name="Is_True",
        )
        .filter(pl.col("Is_True"))
    )

    # --- 3. Create the Improved Marker Layer ---
    if marker_data.height > 0:
        markers = (
            alt.Chart(marker_data)
            .mark_point(
                size=50,
                filled=False,
                stroke="black",
                strokeWidth=0.5,
            )
            .encode(
                x=alt.X("Navn:N", sort=species_order),
                y=alt.Y("y_pos:Q"),  # REMOVED axis=None here
                shape=alt.Shape(
                    "Status:N",
                    scale=alt.Scale(domain=marker_cols, range=["circle", "square", "triangle-up"]),
                    legend=alt.Legend(title="Forvaltningsinteresse"),
                ),
                tooltip=[alt.Tooltip("Navn", title="Art"), alt.Tooltip("Status", title="Status")],
            )
            .transform_window(
                marker_rank="rank()",
                groupby=["Navn"],
            )
            .transform_calculate(
                y_pos=f"datum.Total + {marker_offset} * datum.marker_rank"
            )
        )

        # Layer the charts and resolve the Y scale independently
        chart = alt.layer(bars, markers).resolve_scale(
            y='shared'  # This ensures the Y-axis from the bars is used
        )
    else:
        chart = bars

    # --- 4. Final Chart Configuration ---
    final_chart = (
        chart.properties(
            width=1200, 
            height=500, 
            title=f"{metric_dropdown.value} sortert etter {sort_field.lower()}"
        )
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_title(fontSize=16, anchor="start")
        .configure_legend(
            titleFontSize=12,
            labelFontSize=11,
            orient="right",
            symbolFillColor="transparent",
            symbolStrokeColor="black",
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


if __name__ == "__main__":
    app.run()
