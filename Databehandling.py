import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import time
    from functools import lru_cache
    import marimo as mo
    import polars as pl
    import requests

    # Constants
    NORTAXA_API_BASE_URL = "https://nortaxa.artsdatabanken.no/api/v1/TaxonName"
    DESIRED_RANKS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus"]
    RATE_LIMIT_DELAY = 0.1  # seconds between API calls (adjust as needed)
    return (
        DESIRED_RANKS,
        NORTAXA_API_BASE_URL,
        RATE_LIMIT_DELAY,
        lru_cache,
        mo,
        pl,
        requests,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    valgt_fil = mo.ui.file_browser()
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
    orginal_df = mo.sql(
        f"""
        SELECT * FROM read_csv('{str(filepath)}');
        """
    )
    return (orginal_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## API (legger til data fra Artsdatabanken)""")
    return


@app.cell
def _(DESIRED_RANKS, NORTAXA_API_BASE_URL, lru_cache, requests):
    # Cell 3: API functions with caching
    @lru_cache(maxsize=10000)
    def fetch_taxon_data(scientific_name_id):
        """Fetch taxon data with caching to avoid duplicate API calls."""
        try:
            response = requests.get(
                f"{NORTAXA_API_BASE_URL}/ByScientificNameId/{scientific_name_id}",
                timeout=10
            )
            if response.ok:
                return response.json()
        except Exception as e:
            print(f"Error fetching ID {scientific_name_id}: {e}")
        return None

    def extract_hierarchy_and_ids(api_data):
        """Extract taxonomic hierarchy and rank IDs from API data."""
        hierarchy = {}
        family_id = order_id = None

        if api_data and "higherClassification" in api_data:
            for level in api_data["higherClassification"]:
                rank = level.get("taxonRank")
                if rank in DESIRED_RANKS:
                    hierarchy[rank] = level.get("scientificName")
                if rank == "Family":
                    family_id = level.get("scientificNameId")
                elif rank == "Order":
                    order_id = level.get("scientificNameId")

        return hierarchy, family_id, order_id

    def get_norwegian_name(api_data):
        """Extract Norwegian vernacular name (prioritize Bokmål over Nynorsk)."""
        if not api_data or "vernacularNames" not in api_data:
            return None

        names = api_data["vernacularNames"]
        # First try Bokmål
        for name in names:
            if name.get("languageIsoCode") == "nb":
                return name.get("vernacularName")
        # Fallback to Nynorsk
        for name in names:
            if name.get("languageIsoCode") == "nn":
                return name.get("vernacularName")
        return None
    return extract_hierarchy_and_ids, fetch_taxon_data, get_norwegian_name


@app.cell(hide_code=True)
def _(
    DESIRED_RANKS,
    RATE_LIMIT_DELAY,
    extract_hierarchy_and_ids,
    fetch_taxon_data,
    get_norwegian_name,
    mo,
    pl,
    time,
):
    def process_and_enrich_data(source_df):
        """Process the dataframe and enrich with taxonomy data."""
        # Convert to Polars for better performance
        df_work = pl.from_pandas(source_df.to_pandas() if hasattr(source_df, 'to_pandas') else source_df)

        # Check if required column exists
        if "validScientificNameId" not in df_work.columns:
            mo.md("❌ Error: 'validScientificNameId' column not found in input data.")
            return None, None

        # Get unique IDs
        unique_ids = df_work.select("validScientificNameId").drop_nulls().unique().to_series().to_list()
        total_ids = len(unique_ids)

        # Storage for results
        taxonomy_data = {}
        family_names = {}
        order_names = {}

        # Process with progress bar
        with mo.status.progress_bar(total=total_ids) as bar:
            bar.update(0, title="Fetching taxonomy data from NorTaxa API...")

            for i, species_id in enumerate(unique_ids):
                try:
                    species_id = int(species_id)
                except (ValueError, TypeError):
                    bar.update(i + 1)
                    continue

                # Fetch species data
                species_data = fetch_taxon_data(species_id)
                if species_data:
                    hierarchy, family_id, order_id = extract_hierarchy_and_ids(species_data)
                    taxonomy_data[species_id] = hierarchy

                    # Fetch family name if available
                    if family_id:
                        family_data = fetch_taxon_data(family_id)
                        if family_data:
                            family_names[species_id] = get_norwegian_name(family_data)

                    # Fetch order name if available
                    if order_id:
                        order_data = fetch_taxon_data(order_id)
                        if order_data:
                            order_names[species_id] = get_norwegian_name(order_data)

                # Rate limiting
                if RATE_LIMIT_DELAY > 0:
                    time.sleep(RATE_LIMIT_DELAY)

                # Update progress
                bar.update(i + 1, title=f"Processing ID {species_id} ({i+1}/{total_ids})")

        # Add taxonomy columns with proper return_dtype
        for rank in DESIRED_RANKS:
            df_work = df_work.with_columns(
                pl.col("validScientificNameId")
                .map_elements(
                    lambda x: taxonomy_data.get(int(x), {}).get(rank) if x and x is not None else None,
                    return_dtype=pl.Utf8  # Fixed: Added return_dtype
                )
                .alias(rank)
            )

        # Add Norwegian names with proper return_dtype
        df_work = df_work.with_columns([
            pl.col("validScientificNameId")
            .map_elements(
                lambda x: family_names.get(int(x)) if x and x is not None else None,
                return_dtype=pl.Utf8  # Fixed: Added return_dtype
            )
            .alias("FamilieNavn"),

            pl.col("validScientificNameId")
            .map_elements(
                lambda x: order_names.get(int(x)) if x and x is not None else None,
                return_dtype=pl.Utf8  # Fixed: Added return_dtype
            )
            .alias("OrdenNavn")
        ])

        return df_work, len(taxonomy_data)
    return (process_and_enrich_data,)


@app.cell
def _(orginal_df, process_and_enrich_data):
    # Cell 5: Execute the enrichment
    df_enriched, num_species_processed = process_and_enrich_data(orginal_df)
    return (df_enriched,)


@app.cell
def _(df_enriched):
    df_enriched
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Legger til arter av nasjonal forvaltningsinteresse""")
    return


@app.cell(hide_code=True)
def _(mo, pl):
    # Cell 1: Load Excel with criteria
    excel_path = r"C:\Users\havh\OneDrive - Multiconsult\Dokumenter\Kodeprosjekter\Artsdatabanken\Artsdatabanken\Arter av nasjonal forvaltningsinteresse\ArtslisteArtnasjonal_2023_01-31 (1).xlsx"
    df_excel = pl.read_excel(excel_path)

    # Get criteria columns (those starting with "Kriterium")
    criteria_cols = [col for col in df_excel.columns[4:] if col.startswith("Kriterium")]
    mo.md(f"Found {len(criteria_cols)} criteria columns")
    return criteria_cols, df_excel


@app.cell(hide_code=True)
def _(criteria_cols, df_excel, pl):
    # Cell 2: Process criteria data
    # Convert X marks to Yes/No for each criterion
    criteria_data = df_excel.select(
        ["ValidScientificNameId"] + 
        [pl.when(pl.col(col).str.to_uppercase().str.strip_chars() == "X")
         .then(pl.lit("Yes"))
         .otherwise(pl.lit("No"))
         .alias(col.replace("Kriterium_", "").replace("_", " "))
         for col in criteria_cols]
    )
    criteria_data
    return (criteria_data,)


@app.cell
def _(criteria_cols, criteria_data, df_enriched, pl):
    # Cell 3: Merge with enriched data
    # Left join to keep all rows from df_enriched
    df_with_criteria = df_enriched.join(
        criteria_data,
        left_on="validScientificNameId",
        right_on="ValidScientificNameId",
        how="left"
    )

    # Fill nulls with "No" for non-matched rows
    criteria_renamed = [col.replace("Kriterium_", "").replace("_", " ") for col in criteria_cols]
    df_with_criteria = df_with_criteria.with_columns(
        [pl.col(col).fill_null("No") for col in criteria_renamed]
    )
    return (df_with_criteria,)


@app.cell
def _(df_with_criteria):
    # Cell 4: Display results
    df_with_criteria
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Rydder opp i kolonner og navn""")
    return


@app.cell
def _(df_with_criteria):
    dropper_kolonner_df = df_with_criteria
    dropper_kolonner_df = dropper_kolonner_df.select(["category", "validScientificNameId", "validScientificName", "preferredPopularName", "taxonGroupName", "collector", "dateTimeCollected", "locality", "coordinateUncertaintyInMeters", "municipality", "county", "individualCount", "latitude", "longitude", "geometry", "scientificNameRank", "behavior", "FamilieNavn", "OrdenNavn", "Ansvarsarter", "Andre spesielt hensynskrevende arter", "Spesielle okologiske former", "Prioriterte arter", "Fredete arter", "Fremmede arter"])
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
    endrer_navn_df = endrer_navn_df.rename({"FamilieNavn": "Familie"})
    endrer_navn_df = endrer_navn_df.rename({"OrdenNavn": "Orden"})
    endrer_navn_df = endrer_navn_df.rename({"Spesielle okologiske former": "Spesielle økologiske former"})
    return (endrer_navn_df,)


@app.cell
def _(endrer_navn_df):
    oppryddet_df=endrer_navn_df

    oppryddet_df
    return (oppryddet_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Utforsker og fikser mangler/utfordringer i datasettet""")
    return


@app.cell
def _(mo, oppryddet_df, pl):
    # Cell 1: Missing Value Overview
    null_summary = pl.DataFrame({
        "Column": oppryddet_df.columns,
        "Null Count": [oppryddet_df[col].null_count() for col in oppryddet_df.columns],
        "Total Rows": len(oppryddet_df)
    }).with_columns(
        (pl.col("Null Count") / pl.col("Total Rows") * 100).round(2).alias("Null %")
    ).select(["Column", "Null Count", "Null %"]).filter(
        pl.col("Null Count") > 0
    ).sort("Null Count", descending=True)

    # Define expected vs unexpected nulls
    expected_null_columns = [
        "Usikkerhet meter", 
        "Atferd", 
        "Observatør",
    ]

    # Add column to indicate if nulls are expected
    null_summary_with_status = null_summary.with_columns(
        pl.when(pl.col("Column").is_in(expected_null_columns))
        .then(pl.lit("Expected"))
        .otherwise(pl.lit("UNEXPECTED"))
        .alias("Null Status")
    )

    # Filter unexpected nulls
    unexpected_nulls = null_summary_with_status.filter(pl.col("Null Status") == "UNEXPECTED")

    mo.md(f"""
    #### Missing Value Overview

    Found **{null_summary.height}** columns with missing values out of **{len(oppryddet_df.columns)}** total columns.

    ### Unexpected Missing Values
    **{unexpected_nulls.height}** columns have unexpected null values:

    {mo.ui.table(unexpected_nulls.select(["Column", "Null Count", "Null %"]), selection=None) if unexpected_nulls.height > 0 else "No unexpected null values found!"}

    ### All Missing Values
    {mo.ui.table(null_summary_with_status, selection=None)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Fikser evt. feil/mangler""")
    return


@app.cell
def _(oppryddet_df, pl):
    # Setter alle null verdier i observasjoner lik 1

    null_verdier_df = oppryddet_df.with_columns(
        pl.col("Antall").fill_null(0))
    return (null_verdier_df,)


@app.cell
def _(null_verdier_df, pl):
    # Endrer fra komma til punktum seperasjon i lengdegrad og breddegrad

    lat_lon_ok_df = null_verdier_df.with_columns(
        pl.col(["latitude", "longitude"]).str.replace_all(",", ".").cast(pl.Float64)
    )
    return (lat_lon_ok_df,)


@app.cell
def _(lat_lon_ok_df):
    endelig_datasett = lat_lon_ok_df
    return (endelig_datasett,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Eksporter fikset og databehandlet datasett""")
    return


@app.cell
def _(endelig_datasett):
    endelig_datasett
    return


if __name__ == "__main__":
    app.run()
