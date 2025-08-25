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

    return ff, mo, px


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


@app.cell
def _(mo):
    mo.md(
        r"""
    Ting å huske på

    1. Alle polygoner fra artskart hvor det er knyttet flere arts.obs til blir kollapset til ett enkelt punkt. Slik at et punkt kan ha veldig mange obs.
    2. Husk at punktene er registrert ofte med stor usikkerhet for fugl/bevegelige dyr - slik at disse blir nesten mer en guideline. Du må se mer til habitat/den strukturelle konnektiviteten
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create a dropdown to toggle between Kategori and Navn
    legend_toggle = mo.ui.dropdown(
        options=["Kategori", "Navn"],
        value="Kategori",
        label="Tegnforklaring:"
    )
    legend_toggle
    return (legend_toggle,)


@app.cell(hide_code=True)
def _(
    artsdata_kart,
    legend_toggle,
    map_style_dropdown,
    mo,
    px,
    satellite_toggle,
):
    fig = px.scatter_map(
        artsdata_kart,
        lat="latitude",
        lon="longitude",
        color=legend_toggle.value,
        zoom=10,
        size="Antall",
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
        options=["Antall observasjoner", "Sum individer"], value="Antall observasjoner", label="Aggregation mode:"
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


if __name__ == "__main__":
    app.run()
