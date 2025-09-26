import marimo

__generated_with = "0.16.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
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
    valgt_fil = mo.ui.file_browser(initial_path=r"C:\Users\havh\OneDrive - Multiconsult\Dokumenter\Oppdrag")
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
    mo.md(r"""### Oppdatere og rydder i datasettet""")
    return


@app.cell
def _(add_national_interest_criteria, orginal_df, process_and_enrich_data):
    #Legger til artsdatabanken info
    df_enriched, num_species_processed = process_and_enrich_data(orginal_df)

    #Legger til arter av nasjonal forvaltningsinteresse
    df_with_criteria = add_national_interest_criteria(df_enriched)

    #Dropper kolonner som ikke er relevante og endrer til norske navn
    oppryddet_df = clean_and_rename_columns(df_with_criteria)
    return (oppryddet_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Utforsker og fikser mangler/utfordringer i datasettet""")
    return


@app.cell(hide_code=True)
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
        "Lokalitet"
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

    # Get columns with unexpected nulls
    unexpected_null_columns = unexpected_nulls.get_column("Column").to_list()

    # Create filter expression for rows with any unexpected null
    if unexpected_null_columns:
        filter_expr = pl.any_horizontal([pl.col(col).is_null() for col in unexpected_null_columns])
        rows_with_unexpected_nulls = oppryddet_df.filter(filter_expr)

        # Select relevant columns for display - avoid duplicates
        base_columns = ["Art", "Navn"]
        # Remove any base columns that are already in unexpected_null_columns to avoid duplicates
        base_columns = [col for col in base_columns if col not in unexpected_null_columns]
        display_columns = base_columns + unexpected_null_columns

        # Ensure all columns exist in the dataframe
        display_columns = [col for col in display_columns if col in oppryddet_df.columns]

        unexpected_null_sample = rows_with_unexpected_nulls.select(display_columns)
    else:
        unexpected_null_sample = pl.DataFrame()

    mo.md(f"""
    #### Missing Value Overview
    ### All Missing Values Summary
    {mo.ui.table(null_summary_with_status, selection=None)}
    Found **{null_summary.height}** columns with missing values out of **{len(oppryddet_df.columns)}** total columns.

    ### Unexpected Missing Values
    **{unexpected_nulls.height}** columns have unexpected null values:

    {mo.ui.table(unexpected_nulls.select(["Column", "Null Count", "Null %"]), selection=None) if unexpected_nulls.height > 0 else "No unexpected null values found!"}

    ### Sample Rows with Unexpected Nulls
    {f"Showing all {rows_with_unexpected_nulls.height} rows with unexpected null values:" if unexpected_null_columns else ""}

    {mo.ui.table(unexpected_null_sample, selection=None) if not unexpected_null_sample.is_empty() else ""}


    """)
    return


@app.cell(hide_code=True)
def _(mo, oppryddet_df, pl):
    # Fikser navn

    # Get columns that have at least one null value
    columns_with_nulls = oppryddet_df.filter(pl.col("Navn").is_null())

    # Always include "Art" column even if it has no nulls
    if "Art" not in columns_with_nulls:
        columns_with_nulls = ["Art"] + columns_with_nulls

    # Filter to only rows that have at least one null value
    rows_with_nulls = oppryddet_df.filter(
        pl.any_horizontal([pl.col(col).is_null() for col in oppryddet_df.columns])
    )

    # Select only the relevant columns
    filtered_df = rows_with_nulls.select(columns_with_nulls)

    # Create the editor with the filtered data
    editor = mo.ui.data_editor(
        data=filtered_df, 
        label=f"Edit Data - {filtered_df.height} rows with nulls, {len(columns_with_nulls)} columns"
    )
    mo.vstack([
    mo.md(f""" ####Endrer navn manuelt"""),
    editor

    ])


    return (editor,)


@app.cell
def _(editor):
    # This will show the dataframe with your edits
    edited_df1 = editor.value
    edited_df1
    return


@app.cell(hide_code=True)
def _(editor, oppryddet_df, pl):
    # Get the edited data
    edited_df = editor.value

    # Create a mapping of Artens ID to edited Navn values
    navn_updates = edited_df.select(["Artens ID", "Navn"]).filter(pl.col("Navn").is_not_null())

    # Update the original dataframe
    updated_df = oppryddet_df.join(
        navn_updates.rename({"Navn": "Navn_new"}),
        on="Artens ID",
        how="left"
    ).with_columns(
        pl.when(pl.col("Navn_new").is_not_null())
        .then(pl.col("Navn_new"))
        .otherwise(pl.col("Navn"))
        .alias("Navn")
    ).drop("Navn_new")
    return (updated_df,)


@app.cell
def _(pl, updated_df):
    # Setter alle null verdier i observasjoner lik 1

    null_verdier_df = updated_df.with_columns(
        pl.col("Antall").fill_null(1))
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


@app.cell
def _(mo):
    mo.md(r"""## Legger til en ny kolonne spm oppsumerer hvilken av forvaltningsinteressekategorien arter en""")
    return


@app.cell
def _(endelig_datasett, pl):
    # Add column summarizing national management interest categories
    category_columns = [
        "Ansvarsarter",
        "Andre spesielt hensynskrevende arter",
        "Spesielle økologiske former", 
        "Prioriterte arter",
        "Fredete arter",
        "Fremmede arter"
    ]

    # Create expressions to check each category and return its name if TRUE
    category_expressions = []
    for col in category_columns:
        category_expressions.append(
            pl.when(pl.col(col) == "Yes")
            .then(pl.lit(col))
            .otherwise(pl.lit(None))
        )

    # Combine all categories into a single column
    endelig_datasett_med_kategori = endelig_datasett.with_columns(
        pl.when(
            pl.concat_list(category_expressions)
            .list.drop_nulls()
            .list.join(", ") != ""
        )
        .then(
            pl.concat_list(category_expressions)
            .list.drop_nulls()
            .list.join(", ")
        )
        .otherwise(pl.lit("Nei"))
        .alias("Art av nasjonal forvaltningsinteresse")
    )

    endelig_datasett_med_kategori
    return (endelig_datasett_med_kategori,)


@app.cell
def _(mo):
    mo.md(r"""#Endrer kolonnerekkefølge""")
    return


@app.cell
def _(endelig_datasett_med_kategori):
    # Define desired column order
    first_columns = [
        "Kategori",
        "Art av nasjonal forvaltningsinteresse",
        "Navn", 
        "Art",
        "Antall",
        "Observert dato",
        "Usikkerhet meter",
        "Atferd",
        "Familie",
        "Orden"
    ]

    # Get remaining columns not in first_columns
    remaining_columns = [col for col in endelig_datasett_med_kategori.columns if col not in first_columns]

    # Combine to create final column order
    final_column_order = first_columns + remaining_columns

    # Reorder the dataframe
    endelig_datasett_reordered = endelig_datasett_med_kategori.select(final_column_order)

    endelig_datasett_reordered
    return (endelig_datasett_reordered,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Eksporter fikset og databehandlet datasett""")
    return


@app.cell
def _(endelig_datasett_reordered):
    endelig_datasett_reordered
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""# Utility functions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### API fra artsdatabanken""")
    return


@app.cell
def _(DESIRED_RANKS, NORTAXA_API_BASE_URL, lru_cache, requests):
    # Henter artsdata via API

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


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Arter av nasjonal forvaltningsinteresse""")
    return


@app.cell
def _(pl):
    def add_national_interest_criteria(df_enriched, excel_path=None):
        """
        Add national interest criteria from Excel file to enriched dataframe.

        Parameters:
        -----------
        df_enriched : pl.DataFrame
            The enriched dataframe with species data
        excel_path : str, optional
            Path to the Excel file with criteria. If None, uses default path.

        Returns:
        --------
        pl.DataFrame
            DataFrame with added criteria columns
        """
        # Use default path if not provided
        if excel_path is None:
            excel_path = r"Arter av nasjonal forvaltningsinteresse\ArtslisteArtnasjonal_2023_01-31 (1).xlsx"

        # Load Excel with criteria
        df_excel = pl.read_excel(excel_path)

        # Get criteria columns (those starting with "Kriterium")
        criteria_cols = [col for col in df_excel.columns[4:] if col.startswith("Kriterium")]

        # Process criteria data - convert X marks to Yes/No
        criteria_data = df_excel.select(
            ["ValidScientificNameId"] + 
            [pl.when(pl.col(col).str.to_uppercase().str.strip_chars() == "X")
             .then(pl.lit("Yes"))
             .otherwise(pl.lit("No"))
             .alias(col.replace("Kriterium_", "").replace("_", " "))
             for col in criteria_cols]
        )

        # Merge with enriched data
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

        return df_with_criteria
    return (add_national_interest_criteria,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Rydder og fikser navn""")
    return


@app.function
def clean_and_rename_columns(df):
    """
    Select specific columns and rename them to Norwegian names.

    Parameters:
    -----------
    df : pl.DataFrame
        Input dataframe with raw column names

    Returns:
    --------
    pl.DataFrame
        DataFrame with selected and renamed columns
    """
    # Define columns to keep
    columns_to_keep = [
        "category", "validScientificNameId", "validScientificName", 
        "preferredPopularName", "taxonGroupName", "collector", 
        "dateTimeCollected", "locality", "coordinateUncertaintyInMeters", 
        "municipality", "county", "individualCount", "latitude", 
        "longitude", "geometry", "scientificNameRank", "behavior", 
        "FamilieNavn", "OrdenNavn", "Ansvarsarter", 
        "Andre spesielt hensynskrevende arter", "Spesielle okologiske former", 
        "Prioriterte arter", "Fredete arter", "Fremmede arter"
    ]

    # Define column renaming mapping
    rename_mapping = {
        "category": "Kategori",
        "validScientificName": "Art",
        "preferredPopularName": "Navn",
        "taxonGroupName": "Artsgruppe",
        "validScientificNameId": "Artens ID",
        "collector": "Observatør",
        "dateTimeCollected": "Observert dato",
        "locality": "Lokalitet",
        "coordinateUncertaintyInMeters": "Usikkerhet meter",
        "municipality": "Kommune",
        "county": "Fylke",
        "individualCount": "Antall",
        "scientificNameRank": "Taksonomisk nivå",
        "behavior": "Atferd",
        "FamilieNavn": "Familie",
        "OrdenNavn": "Orden",
        "Spesielle okologiske former": "Spesielle økologiske former"
    }

    # Select columns and rename in one operation
    cleaned_df = (df
        .select(columns_to_keep)
        .rename(rename_mapping)
    )

    return cleaned_df


if __name__ == "__main__":
    app.run()
