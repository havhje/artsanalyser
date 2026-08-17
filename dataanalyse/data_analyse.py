import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns", layout_file="layouts/data_analyse.grid.json")

with app.setup(hide_code=True):
    import marimo as mo
    import altair as alt
    import polars as pl
    import plotly.express as px
    import leafmap.foliumap as leafmap
    import colorcet as cc
    import holoviews.operation.datashader as h
    import hvplot.polars
    import holoviews as hv
    import datashader as ds
    import geopandas as gpd
    from holoviews.element.tiles import EsriImagery

    import great_tables as gt
    from datetime import date
    import inspect
    import textwrap


@app.cell(hide_code=True)
def _():
    valgt_fil = mo.ui.file_browser(
        filetypes=[".parquet"],
        multiple=False,
        label="Velg ferdig behandlet Parquet-fil",
    )
    valgt_fil
    return (valgt_fil,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Funksjons og tester GT-artstabell
    """)
    return


@app.cell(hide_code=True)
def _(arter_df, lag_artsstatistikk):
    artsstatistikk_df = lag_artsstatistikk(arter_df)
    return (artsstatistikk_df,)


@app.cell(hide_code=True)
def _():
    ARTSSTATISTIKK_INPUTKOLONNER = {
        "Artens ID",
        "Art",
        "Navn",
        "Kategori",
        "Verdi M1941",
        "Art av nasjonal forvaltningsinteresse (eks. rødlista)",
        "Antall",
        "Atferd",
        "Observert dato",
        "Familie",
        "Orden",
    }

    ARTSSTATISTIKK_OUTPUTKOLONNER = [
        "Artens ID",
        "Kategori",
        "Verdi M1941",
        "Forvaltningsinteresse",
        "Navn",
        "Art",
        "Observasjoner",
        "Individer",
        "Gj.snitt individer",
        "År-periode",
        "Måneder",
        "Månedsprofil",
        "Familie",
        "Orden",
        "Reproduksjon",
        "Mulig reproduksjon",
    ]

    ARTSSTATISTIKK_METADATAKOLONNER = [
        "Navn",
        "Kategori",
        "Verdi M1941",
        "Art av nasjonal forvaltningsinteresse (eks. rødlista)",
        "Familie",
        "Orden",
    ]

    ARTSSTATISTIKK_TEKSTKOLONNER = {
        "Art",
        "Navn",
        "Kategori",
        "Verdi M1941",
        "Art av nasjonal forvaltningsinteresse (eks. rødlista)",
        "Atferd",
        "Familie",
        "Orden",
    }

    ARTSSTATISTIKK_MAANEDSNAVN = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "Mai",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Okt",
        11: "Nov",
        12: "Des",
    }

    ARTSSTATISTIKK_KATEGORI_REKKEFOELGE = {
        kategori: indeks
        for indeks, kategori in enumerate(
            [
                "RE",
                "CR",
                "EN",
                "VU",
                "NT",
                "LC",
                "DD",
                "SE",
                "HI",
                "PH",
                "LO",
                "NK",
                "NA",
                "NE",
                "Unknown",
            ]
        )
    }

    ARTSSTATISTIKK_KATEGORIFARGER = {
        "RE": "#000000",
        "CR": "#D81E05",
        "EN": "#FC7F3F",
        "VU": "#F9E814",
        "NT": "#CCE226",
        "LC": "#60C659",
        "DD": "#D1D1C6",
        "SE": "#A50026",
        "HI": "#D73027",
        "PH": "#F46D43",
        "LO": "#FEE08B",
        "NK": "#D9EF8B",
        "NA": "#C1B5A5",
        "NE": "#FFFFFF",
        "Unknown": "#D9D9D9",
    }

    ARTSSTATISTIKK_M1941_FARGER = {
        "Svært stor verdi": "#AF0F0F",
        "Stor verdi": "#FD7032",
        "Middels verdi": "#FEC02D",
        "Noe verdi": "#FFFF66",
        "Ingen": "#D9D9D9",
    }
    return (
        ARTSSTATISTIKK_INPUTKOLONNER,
        ARTSSTATISTIKK_KATEGORIFARGER,
        ARTSSTATISTIKK_KATEGORI_REKKEFOELGE,
        ARTSSTATISTIKK_M1941_FARGER,
        ARTSSTATISTIKK_MAANEDSNAVN,
        ARTSSTATISTIKK_METADATAKOLONNER,
        ARTSSTATISTIKK_OUTPUTKOLONNER,
        ARTSSTATISTIKK_TEKSTKOLONNER,
    )


@app.cell(hide_code=True)
def _(ARTSSTATISTIKK_INPUTKOLONNER):
    def hent_påkrevde_artsstatistikk_kolonner() -> set[str]:
        """Returner kolonnene som kreves for å lage artsstatistikken."""
        return set(ARTSSTATISTIKK_INPUTKOLONNER)

    return (hent_påkrevde_artsstatistikk_kolonner,)


@app.cell(hide_code=True)
def _(
    ARTSSTATISTIKK_KATEGORI_REKKEFOELGE,
    ARTSSTATISTIKK_METADATAKOLONNER,
    ARTSSTATISTIKK_TEKSTKOLONNER,
    hent_påkrevde_artsstatistikk_kolonner,
):
    def valider_artsstatistikk_input(df: pl.DataFrame) -> None:
        """Valider inputkontrakten for artsstatistikk.

        Args:
            df: Ferdig behandlet observasjonsdata.

        Raises:
            TypeError: Når input eller en obligatorisk kolonne har feil datatype.
            ValueError: Når kolonner mangler, kategorier er ukjente eller samme
                takson har motstridende metadata.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Artsstatistikk krever en Polars DataFrame")

        manglende_kolonner = sorted(hent_påkrevde_artsstatistikk_kolonner() - set(df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler obligatoriske kolonner for artsstatistikk: " + ", ".join(manglende_kolonner))

        if not df.schema["Artens ID"].is_integer():
            raise TypeError("Kolonnen `Artens ID` må ha heltallstype")
        if not df.schema["Antall"].is_integer():
            raise TypeError("Kolonnen `Antall` må ha heltallstype")
        if df.schema["Observert dato"].base_type() not in {pl.Date, pl.Datetime}:
            raise TypeError("Kolonnen `Observert dato` må ha typen Date eller Datetime")

        feil_teksttyper = sorted(kolonne for kolonne in ARTSSTATISTIKK_TEKSTKOLONNER if df.schema[kolonne] != pl.String)
        if feil_teksttyper:
            raise TypeError("Følgende artsstatistikk-kolonner må ha teksttype: " + ", ".join(feil_teksttyper))

        if df.get_column("Antall").null_count() > 0:
            raise ValueError("Kolonnen `Antall` kan ikke inneholde nullverdier")

        tillatte_kategorier = set(ARTSSTATISTIKK_KATEGORI_REKKEFOELGE)
        ukjente_kategorier = (
            df.filter(pl.col("Kategori").is_null() | ~pl.col("Kategori").is_in(tillatte_kategorier))
            .get_column("Kategori")
            .unique()
            .to_list()
        )
        if ukjente_kategorier:
            kategoritekst = ", ".join(sorted("<null>" if verdi is None else str(verdi) for verdi in ukjente_kategorier))
            raise ValueError(f"Ukjente Kategori-verdier: {kategoritekst}")

        metadata_konflikter = (
            df.group_by(["Artens ID", "Art"])
            .agg(
                [pl.col(kolonne).drop_nulls().n_unique().alias(kolonne) for kolonne in ARTSSTATISTIKK_METADATAKOLONNER]
            )
            .filter(pl.any_horizontal([pl.col(kolonne) > 1 for kolonne in ARTSSTATISTIKK_METADATAKOLONNER]))
        )
        if metadata_konflikter.height > 0:
            konfliktkolonner = [
                kolonne
                for kolonne in ARTSSTATISTIKK_METADATAKOLONNER
                if metadata_konflikter.filter(pl.col(kolonne) > 1).height > 0
            ]
            raise ValueError("Motstridende artsmetadata for samme Artens ID/Art: " + ", ".join(konfliktkolonner))

    return (valider_artsstatistikk_input,)


@app.cell(hide_code=True)
def _(
    ARTSSTATISTIKK_KATEGORI_REKKEFOELGE,
    ARTSSTATISTIKK_MAANEDSNAVN,
    ARTSSTATISTIKK_OUTPUTKOLONNER,
    valider_artsstatistikk_input,
):
    def lag_artsstatistikk(df: pl.DataFrame) -> pl.DataFrame:
        """Aggreger observasjoner til én rad per takson.

        Månedsprofilen inneholder alltid tolv heltall i rekkefølgen januar–desember.
        Taksa identifiseres med kombinasjonen `Artens ID` og `Art`, ikke norsk navn.
        """
        valider_artsstatistikk_input(df)

        maanedsaggregeringer = [
            (pl.col("Observert dato").dt.month() == maaned).sum().cast(pl.Int64).alias(f"__maaned_{maaned}")
            for maaned in range(1, 13)
        ]

        return (
            df.group_by(["Artens ID", "Art"], maintain_order=True)
            .agg(
                [
                    pl.col("Navn").drop_nulls().first().alias("Navn"),
                    pl.col("Kategori").drop_nulls().first().alias("Kategori"),
                    pl.col("Verdi M1941").drop_nulls().first().alias("Verdi M1941"),
                    pl.col("Art av nasjonal forvaltningsinteresse (eks. rødlista)")
                    .drop_nulls()
                    .first()
                    .alias("Forvaltningsinteresse"),
                    pl.len().cast(pl.Int64).alias("Observasjoner"),
                    pl.col("Antall").sum().cast(pl.Int64).alias("Individer"),
                    pl.col("Antall").mean().cast(pl.Float64).alias("Gj.snitt individer"),
                    pl.col("Observert dato").dt.year().min().alias("__aar_fra"),
                    pl.col("Observert dato").dt.year().max().alias("__aar_til"),
                    pl.col("Observert dato").dt.month().drop_nulls().unique().sort().alias("__maaneder"),
                    pl.col("Familie").drop_nulls().first().alias("Familie"),
                    pl.col("Orden").drop_nulls().first().alias("Orden"),
                    (pl.col("Atferd") == "reproductive").sum().cast(pl.Int64).alias("Reproduksjon"),
                    (pl.col("Atferd") == "possiblereproductive").sum().cast(pl.Int64).alias("Mulig reproduksjon"),
                    *maanedsaggregeringer,
                ]
            )
            .with_columns(
                [
                    pl.coalesce(
                        [
                            pl.col("Navn").str.strip_chars().replace("", None),
                            pl.col("Art"),
                            pl.lit("Ukjent art"),
                        ]
                    ).alias("Navn"),
                    pl.when(pl.col("__aar_fra").is_null())
                    .then(pl.lit(None, dtype=pl.String))
                    .when(pl.col("__aar_fra") == pl.col("__aar_til"))
                    .then(pl.col("__aar_fra").cast(pl.String))
                    .otherwise(pl.concat_str([pl.col("__aar_fra"), pl.lit("–"), pl.col("__aar_til")]))
                    .alias("År-periode"),
                    pl.col("__maaneder")
                    .list.eval(
                        pl.element().replace_strict(
                            ARTSSTATISTIKK_MAANEDSNAVN,
                            return_dtype=pl.String,
                        )
                    )
                    .list.join(", ")
                    .alias("Måneder"),
                    pl.concat_list([pl.col(f"__maaned_{maaned}") for maaned in range(1, 13)]).alias("Månedsprofil"),
                ]
            )
            .with_columns(
                pl.col("Kategori")
                .replace_strict(
                    ARTSSTATISTIKK_KATEGORI_REKKEFOELGE,
                    default=999,
                )
                .alias("__kategori_sortering")
            )
            .sort(
                ["__kategori_sortering", "Observasjoner"],
                descending=[False, True],
                maintain_order=True,
            )
            .select(ARTSSTATISTIKK_OUTPUTKOLONNER)
        )

    return (lag_artsstatistikk,)


@app.cell(hide_code=True)
def _(
    ARTSSTATISTIKK_KATEGORIFARGER,
    ARTSSTATISTIKK_KATEGORI_REKKEFOELGE,
    ARTSSTATISTIKK_M1941_FARGER,
    ARTSSTATISTIKK_OUTPUTKOLONNER,
):
    def lag_artsstatistikk_tabell(artsstatistikk_df: pl.DataFrame) -> gt.GT:
        """Bygg en formatert Great Tables-tabell fra aggregert artsstatistikk."""
        manglende_kolonner = sorted(set(ARTSSTATISTIKK_OUTPUTKOLONNER) - set(artsstatistikk_df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler kolonner for Great Tables-rendering: " + ", ".join(manglende_kolonner))

        antall_arter = artsstatistikk_df.height
        antall_observasjoner = int(artsstatistikk_df["Observasjoner"].sum() or 0)
        antall_individer = int(artsstatistikk_df["Individer"].sum() or 0)

        kategori_domene = list(ARTSSTATISTIKK_KATEGORI_REKKEFOELGE)
        kategori_palett = [ARTSSTATISTIKK_KATEGORIFARGER[kategori] for kategori in kategori_domene]
        m1941_domene = list(ARTSSTATISTIKK_M1941_FARGER)
        m1941_palett = [ARTSSTATISTIKK_M1941_FARGER[verdi] for verdi in m1941_domene]

        tabell = (
            gt.GT(artsstatistikk_df, id="artsstatistikk", locale="nb")
            .tab_header(
                title="Artsstatistikk for valgte observasjoner",
                subtitle=(
                    f"{antall_arter} arter · {antall_observasjoner} observasjoner · {antall_individer} individer"
                ),
            )
            .cols_merge(
                columns=["Navn", "Art"],
                hide_columns=["Art"],
                pattern="<strong>{0}</strong><br><em>{1}</em>",
            )
            .cols_hide(columns="Artens ID")
            .cols_label(
                cases={
                    "Navn": "Art",
                    "Gj.snitt individer": "Gj.snitt",
                    "År-periode": "År",
                }
            )
            .tab_style(
                style=gt.style.text(align="center"),
                locations=gt.loc.column_labels(),
            )
            .cols_align(
                align="center",
                columns=[
                    "Observasjoner",
                    "Individer",
                    "Gj.snitt individer",
                    "Reproduksjon",
                    "Mulig reproduksjon",
                ],
            )
            .fmt_integer(
                columns=[
                    "Observasjoner",
                    "Individer",
                    "Reproduksjon",
                    "Mulig reproduksjon",
                ],
                locale="nb",
            )
            .fmt_number(
                columns="Gj.snitt individer",
                decimals=1,
                locale="nb",
            )
            .fmt_nanoplot(
                columns="Månedsprofil",
                plot_type="bar",
                plot_height="2.2em",
                autoscale=False,
                options=gt.nanoplot_options(
                    data_bar_fill_color="#3B82F6",
                    data_bar_stroke_color="#1D4ED8",
                    data_bar_stroke_width=1,
                    interactive_data_values=True,
                ),
            )
            .sub_missing(missing_text="–")
            .data_color(
                columns="Kategori",
                domain=kategori_domene,
                palette=kategori_palett,
                autocolor_text=True,
            )
            .data_color(
                columns="Verdi M1941",
                domain=m1941_domene,
                palette=m1941_palett,
                autocolor_text=True,
            )
            .tab_spanner(
                label="Art og forvaltning",
                columns=["Kategori", "Verdi M1941", "Forvaltningsinteresse", "Navn"],
            )
            .tab_spanner(
                label="Omfang",
                columns=["Observasjoner", "Individer", "Gj.snitt individer"],
            )
            .tab_spanner(
                label="Tidsrom",
                columns=["År-periode", "Måneder", "Månedsprofil"],
            )
            .tab_spanner(label="Taksonomi", columns=["Familie", "Orden"])
            .tab_spanner(
                label="Aktivitet",
                columns=["Reproduksjon", "Mulig reproduksjon"],
            )
            .tab_footnote(
                footnote=(
                    "Tolv søyler viser antall observasjoner fra januar til desember. "
                    "Skalaen tilpasses hver art for å fremheve sesongmønsteret."
                ),
                locations=gt.loc.column_labels(columns="Månedsprofil"),
            )
            .tab_footnote(
                footnote="Antall observasjoner registrert med atferden «reproductive».",
                locations=gt.loc.column_labels(columns="Reproduksjon"),
            )
            .tab_footnote(
                footnote="Antall observasjoner registrert med atferden «possiblereproductive».",
                locations=gt.loc.column_labels(columns="Mulig reproduksjon"),
            )
            .tab_footnote(
                footnote=(
                    "Oppsummering av nasjonale forvaltningskriterier; «Nei» betyr "
                    "ingen treff utover eventuell rødlistestatus."
                ),
                locations=gt.loc.column_labels(columns="Forvaltningsinteresse"),
            )
            .tab_source_note(source_note="Datagrunnlag: valgte rader i observasjonstabellen.")
            .opt_row_striping()
            .cols_width(
                cases={
                    "Kategori": "70px",
                    "Verdi M1941": "105px",
                    "Forvaltningsinteresse": "190px",
                    "Navn": "180px",
                    "Observasjoner": "85px",
                    "Individer": "80px",
                    "Gj.snitt individer": "75px",
                    "År-periode": "90px",
                    "Måneder": "150px",
                    "Månedsprofil": "170px",
                    "Familie": "130px",
                    "Orden": "150px",
                    "Reproduksjon": "95px",
                    "Mulig reproduksjon": "115px",
                }
            )
            .tab_options(
                container_width="1800px",
                container_height="1200px",
                container_overflow_x="auto",
                container_overflow_y="auto",
                table_width="1800px",
                table_layout="fixed",
                table_font_size="12px",
                heading_title_font_size="18px",
                heading_subtitle_font_size="12px",
                column_labels_font_size="12px",
                data_row_padding="5px",
                row_striping_background_color="#F7F9FC",
                grand_summary_row_background_color="#EAF2F8",
                footnotes_marks="letters",
            )
        )

        if artsstatistikk_df.height > 0:
            tabell = tabell.grand_summary_rows(
                fns={
                    "Totalt": pl.col(
                        "Observasjoner",
                        "Individer",
                        "Reproduksjon",
                        "Mulig reproduksjon",
                    ).sum()
                },
                fmt=lambda verdier: gt.vals.fmt_integer(verdier, locale="nb"),
            )

        return tabell

    return (lag_artsstatistikk_tabell,)


@app.cell(hide_code=True)
def _(ARTSSTATISTIKK_OUTPUTKOLONNER):
    def lag_artsstatistikk_csv(artsstatistikk_df: pl.DataFrame) -> bytes:
        """Eksporter artsstatistikk som Excel-vennlig semikolonseparert UTF-8."""
        manglende_kolonner = sorted(set(ARTSSTATISTIKK_OUTPUTKOLONNER) - set(artsstatistikk_df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler kolonner for artsstatistikk-eksport: " + ", ".join(manglende_kolonner))

        eksport_df = artsstatistikk_df.with_columns(
            pl.col("Månedsprofil")
            .list.eval(pl.element().cast(pl.String))
            .list.join(",")
            .alias("Månedsprofil (Jan–Des)")
        ).drop("Månedsprofil")
        return ("\ufeff" + eksport_df.write_csv(separator=";")).encode("utf-8")

    return (lag_artsstatistikk_csv,)


@app.cell(hide_code=True)
def _(
    hent_påkrevde_artsstatistikk_kolonner,
    lag_artsstatistikk,
    lag_artsstatistikk_csv,
    lag_artsstatistikk_tabell,
    valider_artsstatistikk_input,
):
    def _vis_kildekode(_funksjon):
        _kildekode = textwrap.dedent(inspect.getsource(_funksjon)).strip()
        return mo.md(f"#### `{_funksjon.__name__}`\n\n```python\n{_kildekode}\n```")

    _funksjonsobjekter = [
        hent_påkrevde_artsstatistikk_kolonner,
        valider_artsstatistikk_input,
        lag_artsstatistikk,
        lag_artsstatistikk_tabell,
        lag_artsstatistikk_csv,
    ]
    _funksjonsinnhold = mo.vstack(
        [
            mo.md(
                "Funksjonene under utgjør den validerte kjeden fra observasjoner "
                "til Great Tables-visning og CSV-eksport."
            ),
            *[_vis_kildekode(_funksjon) for _funksjon in _funksjonsobjekter],
        ]
    )
    funksjonsseksjon = mo.accordion({"Funksjoner for artsstatistikken": _funksjonsinnhold})
    funksjonsseksjon
    return


@app.cell(hide_code=True)
def _(ARTSSTATISTIKK_OUTPUTKOLONNER, ARTSSTATISTIKK_TEKSTKOLONNER):
    def lag_artsstatistikk_testinput(
        rad_overrides: list[dict[str, object]] | None = None,
    ) -> pl.DataFrame:
        """Lag en liten, komplett fixture for artsstatistikk-testene."""
        if rad_overrides is None:
            rad_overrides = [{}]

        grunnrad = {
            "Artens ID": 1001,
            "Art": "Species standardus",
            "Navn": "standardart",
            "Kategori": "LC",
            "Verdi M1941": "Noe verdi",
            "Art av nasjonal forvaltningsinteresse (eks. rødlista)": "Nei",
            "Antall": 1,
            "Atferd": None,
            "Observert dato": date(2020, 1, 1),
            "Familie": "standardfamilien",
            "Orden": "standardorden",
        }
        return pl.DataFrame(
            [{**grunnrad, **overrides} for overrides in rad_overrides],
            schema_overrides={kolonne: pl.String for kolonne in ARTSSTATISTIKK_TEKSTKOLONNER},
        )

    def lag_tom_artsstatistikk_testinput() -> pl.DataFrame:
        """Lag tom artsstatistikk-input med riktig schema."""
        tekstkolonner = ARTSSTATISTIKK_TEKSTKOLONNER
        kolonner = {kolonne: pl.Series(kolonne, [], dtype=pl.String) for kolonne in tekstkolonner}
        kolonner["Artens ID"] = pl.Series("Artens ID", [], dtype=pl.Int64)
        kolonner["Antall"] = pl.Series("Antall", [], dtype=pl.Int64)
        kolonner["Observert dato"] = pl.Series("Observert dato", [], dtype=pl.Date)
        return pl.DataFrame(kolonner)

    def artsstatistikk_forventede_kolonner() -> list[str]:
        """Returner godkjent kolonnerekkefølge for artsstatistikken."""
        return list(ARTSSTATISTIKK_OUTPUTKOLONNER)

    return (
        artsstatistikk_forventede_kolonner,
        lag_artsstatistikk_testinput,
        lag_tom_artsstatistikk_testinput,
    )


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_001():
        test_df = lag_artsstatistikk_testinput(
            [
                {
                    "Artens ID": 1,
                    "Art": "Avis exemplaris",
                    "Navn": "eksempelfugl",
                    "Antall": 2,
                    "Atferd": "reproductive",
                    "Observert dato": date(2021, 1, 2),
                },
                {
                    "Artens ID": 1,
                    "Art": "Avis exemplaris",
                    "Navn": "eksempelfugl",
                    "Antall": 4,
                    "Atferd": "possiblereproductive",
                    "Observert dato": date(2023, 3, 4),
                },
            ]
        )
        result = lag_artsstatistikk(test_df)

        assert result.height == 1
        assert result["Observasjoner"].to_list() == [2]
        assert result["Individer"].to_list() == [6]
        assert result["Gj.snitt individer"].to_list() == [3.0]
        assert result["År-periode"].to_list() == ["2021–2023"]
        assert result["Måneder"].to_list() == ["Jan, Mar"]
        assert result["Reproduksjon"].to_list() == [1]
        assert result["Mulig reproduksjon"].to_list() == [1]
        assert result["Månedsprofil"].to_list() == [[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    test_artsstatistikk_mtm_001()
    return (test_artsstatistikk_mtm_001,)


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_002():
        test_df = lag_artsstatistikk_testinput(
            [
                {"Artens ID": 1, "Art": "Species alpha", "Navn": "samme navn"},
                {"Artens ID": 2, "Art": "Species beta", "Navn": "samme navn"},
            ]
        )
        result = lag_artsstatistikk(test_df)

        assert result.height == 2, "Taksa med samme norske navn skal ikke slås sammen"
        assert result["Artens ID"].to_list() == [1, 2]
        assert result["Art"].to_list() == ["Species alpha", "Species beta"]

    test_artsstatistikk_mtm_002()
    return (test_artsstatistikk_mtm_002,)


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_003():
        test_df = lag_artsstatistikk_testinput(
            [
                {"Artens ID": 1, "Art": "lc-liten", "Kategori": "LC"},
                {"Artens ID": 2, "Art": "cr-liten", "Kategori": "CR"},
                {"Artens ID": 3, "Art": "hi-liten", "Kategori": "HI"},
                {"Artens ID": 4, "Art": "ukjent-liten", "Kategori": "Unknown"},
                {"Artens ID": 5, "Art": "lc-stor", "Kategori": "LC"},
                {"Artens ID": 5, "Art": "lc-stor", "Kategori": "LC"},
            ]
        )
        result = lag_artsstatistikk(test_df)

        assert result["Art"].to_list() == [
            "cr-liten",
            "lc-stor",
            "lc-liten",
            "hi-liten",
            "ukjent-liten",
        ]

    test_artsstatistikk_mtm_003()
    return (test_artsstatistikk_mtm_003,)


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_004():
        test_df = lag_artsstatistikk_testinput(
            [
                {"Artens ID": 1, "Art": "Species alpha", "Kategori": "LC"},
                {"Artens ID": 1, "Art": "Species alpha", "Kategori": "NT"},
            ]
        )
        try:
            lag_artsstatistikk(test_df)
        except ValueError as exc:
            assert "Kategori" in str(exc)
            assert "Motstridende artsmetadata" in str(exc)
        else:
            raise AssertionError("Motstridende kategori skulle gitt ValueError")

    test_artsstatistikk_mtm_004()
    return (test_artsstatistikk_mtm_004,)


@app.cell(hide_code=True)
def _(
    artsstatistikk_forventede_kolonner,
    lag_artsstatistikk,
    lag_tom_artsstatistikk_testinput,
):
    def test_artsstatistikk_mtm_005():
        result = lag_artsstatistikk(lag_tom_artsstatistikk_testinput())

        assert result.height == 0
        assert result.columns == artsstatistikk_forventede_kolonner()
        assert result.schema["Observasjoner"] == pl.Int64
        assert result.schema["Individer"] == pl.Int64
        assert result.schema["Gj.snitt individer"] == pl.Float64
        assert result.schema["Månedsprofil"] == pl.List(pl.Int64)

    test_artsstatistikk_mtm_005()
    return (test_artsstatistikk_mtm_005,)


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_006():
        test_df = lag_artsstatistikk_testinput().drop("Atferd")
        try:
            lag_artsstatistikk(test_df)
        except ValueError as exc:
            assert "Atferd" in str(exc)
        else:
            raise AssertionError("Manglende Atferd skulle gitt ValueError")

    test_artsstatistikk_mtm_006()
    return (test_artsstatistikk_mtm_006,)


@app.cell(hide_code=True)
def _(lag_artsstatistikk, lag_artsstatistikk_testinput):
    def test_artsstatistikk_mtm_007():
        feil_antall = lag_artsstatistikk_testinput([{"Antall": "to"}])
        try:
            lag_artsstatistikk(feil_antall)
        except TypeError as exc:
            assert "Antall" in str(exc)
        else:
            raise AssertionError("Antall med teksttype skulle gitt TypeError")

        ukjent_kategori = lag_artsstatistikk_testinput([{"Kategori": "XX"}])
        try:
            lag_artsstatistikk(ukjent_kategori)
        except ValueError as exc:
            assert "XX" in str(exc)
        else:
            raise AssertionError("Ukjent kategori skulle gitt ValueError")

    test_artsstatistikk_mtm_007()
    return (test_artsstatistikk_mtm_007,)


@app.cell(hide_code=True)
def _(
    lag_artsstatistikk,
    lag_artsstatistikk_tabell,
    lag_artsstatistikk_testinput,
):
    def test_artsstatistikk_mtm_008():
        statistikk = lag_artsstatistikk(lag_artsstatistikk_testinput())
        tabell = lag_artsstatistikk_tabell(statistikk)
        html = tabell.as_raw_html()

        assert isinstance(tabell, gt.GT)
        assert "Artsstatistikk for valgte observasjoner" in html
        assert "Månedsprofil" in html
        assert "<svg" in html, "Månedsprofilen skal renderes som nanoplot"
        assert "reproductive" in html

    test_artsstatistikk_mtm_008()
    return (test_artsstatistikk_mtm_008,)


@app.cell(hide_code=True)
def _(
    lag_artsstatistikk,
    lag_artsstatistikk_csv,
    lag_artsstatistikk_testinput,
):
    def test_artsstatistikk_mtm_009():
        statistikk = lag_artsstatistikk(lag_artsstatistikk_testinput())
        eksport = lag_artsstatistikk_csv(statistikk)
        tekst = eksport.decode("utf-8")

        assert eksport.startswith(b"\xef\xbb\xbf")
        assert ";" in tekst.splitlines()[0]
        assert "Månedsprofil (Jan–Des)" in tekst.splitlines()[0]
        assert "1,0,0,0,0,0,0,0,0,0,0,0" in tekst

    test_artsstatistikk_mtm_009()
    return (test_artsstatistikk_mtm_009,)


@app.cell(hide_code=True)
def _(
    artsstatistikk_forventede_kolonner,
    lag_artsstatistikk_testinput,
    lag_tom_artsstatistikk_testinput,
    test_artsstatistikk_mtm_001,
    test_artsstatistikk_mtm_002,
    test_artsstatistikk_mtm_003,
    test_artsstatistikk_mtm_004,
    test_artsstatistikk_mtm_005,
    test_artsstatistikk_mtm_006,
    test_artsstatistikk_mtm_007,
    test_artsstatistikk_mtm_008,
    test_artsstatistikk_mtm_009,
):
    def _vis_testkode(_funksjon):
        _kildekode = textwrap.dedent(inspect.getsource(_funksjon)).strip()
        return mo.md(f"#### `{_funksjon.__name__}`\n\n```python\n{_kildekode}\n```")

    _testmatrise = mo.md(r"""
    ### Testmatrise

    | ID | Scenario | Forventet resultat |
    |---|---|---|
    | ARTSTABELL-MTM-001 | To observasjoner av samme art | Korrekte summer, gjennomsnitt, tidsrom, måneder og aktiviteter |
    | ARTSTABELL-MTM-002 | To taksa med samme norske navn | Taksa holdes atskilt med `Artens ID` og `Art` |
    | ARTSTABELL-MTM-003 | Blandet kategori og observasjonsmengde | Full kategoriorden og flest observasjoner først innen kategori |
    | ARTSTABELL-MTM-004 | Motstridende artsmetadata | Tydelig `ValueError` i stedet for vilkårlig `.first()` |
    | ARTSTABELL-MTM-005 | Tom input med riktig schema | Tom output med fast kolonnerekkefølge og riktige typer |
    | ARTSTABELL-MTM-006 | Manglende obligatorisk kolonne | Tidlig feil som nevner kolonnen |
    | ARTSTABELL-MTM-007 | Feil datatype eller ukjent kategori | Tidlig og forklarende feil |
    | ARTSTABELL-MTM-008 | Great Tables-rendering | Tabell, nanoplot, tittel og fotnoter renderes |
    | ARTSTABELL-MTM-009 | CSV-eksport | UTF-8-BOM, semikolon og tekstlig Jan–Des-profil |
    """)
    _testfunksjoner = [
        lag_artsstatistikk_testinput,
        lag_tom_artsstatistikk_testinput,
        artsstatistikk_forventede_kolonner,
        test_artsstatistikk_mtm_001,
        test_artsstatistikk_mtm_002,
        test_artsstatistikk_mtm_003,
        test_artsstatistikk_mtm_004,
        test_artsstatistikk_mtm_005,
        test_artsstatistikk_mtm_006,
        test_artsstatistikk_mtm_007,
        test_artsstatistikk_mtm_008,
        test_artsstatistikk_mtm_009,
    ]
    _testinnhold = mo.vstack(
        [
            mo.md("Testene kjøres reaktivt og dekker inputkontrakt, aggregering, rendering, nanoplot og eksport."),
            _testmatrise,
            *[_vis_testkode(_funksjon) for _funksjon in _testfunksjoner],
        ]
    )
    testseksjon = mo.accordion({"Tester og testmatrise": _testinnhold})
    testseksjon
    return


@app.cell(hide_code=True)
def aarsprofil_dokumentasjon():
    mo.md(r"""
    ### Funksjoner og tester – observasjoner og individer gjennom året

    Årsprofilen samler kalenderdatoer på tvers av år. Rådatalinjen viser et
    sentrert, rullerende gjennomsnitt, feltet viser ± én standardfeil, og den
    normaliserte linjen viser kartleggingsintensitet på en egen prosentakse.
    """)
    return


@app.cell(hide_code=True)
def aarsprofil_konstanter():
    AARSPROFIL_INPUTKOLONNER = {"Observert dato", "Antall"}

    AARSPROFIL_OUTPUTKOLONNER = [
        "Dato",
        "Gjennomsnitt observasjoner",
        "Gjennomsnitt individer",
        "Standardfeil observasjoner",
        "Standardfeil individer",
        "Antall år",
        "Rullerende observasjoner",
        "Rullerende individer",
        "Nedre observasjoner",
        "Øvre observasjoner",
        "Nedre individer",
        "Øvre individer",
    ]

    KARTLEGGINGSINTENSITET_OUTPUTKOLONNER = [
        "Dato",
        "Valgte observasjoner",
        "Alle observasjoner",
        "Valgte observasjoner i vindu",
        "Alle observasjoner i vindu",
        "Kartleggingsintensitet (%)",
    ]

    AARSPROFIL_MAAL = {
        "Observasjoner": {
            "gjennomsnitt": "Gjennomsnitt observasjoner",
            "rullerende": "Rullerende observasjoner",
            "standardfeil": "Standardfeil observasjoner",
            "nedre": "Nedre observasjoner",
            "øvre": "Øvre observasjoner",
            "farge": "#2563EB",
            "båndfarge": "#93C5FD",
            "tittel": "Observasjoner gjennom året",
            "y-tittel": "Gjennomsnittlig antall observasjoner per dag",
        },
        "Individer": {
            "gjennomsnitt": "Gjennomsnitt individer",
            "rullerende": "Rullerende individer",
            "standardfeil": "Standardfeil individer",
            "nedre": "Nedre individer",
            "øvre": "Øvre individer",
            "farge": "#D97706",
            "båndfarge": "#FED7AA",
            "tittel": "Individer gjennom året",
            "y-tittel": "Gjennomsnittlig antall individer per dag",
        },
    }
    return (
        AARSPROFIL_INPUTKOLONNER,
        AARSPROFIL_MAAL,
        AARSPROFIL_OUTPUTKOLONNER,
        KARTLEGGINGSINTENSITET_OUTPUTKOLONNER,
    )


@app.cell(hide_code=True)
def aarsprofil_inputkolonner(AARSPROFIL_INPUTKOLONNER):
    def hent_påkrevde_aarsprofilkolonner() -> set[str]:
        """Returner kolonnene som kreves for å beregne årsprofilen."""
        return set(AARSPROFIL_INPUTKOLONNER)

    return (hent_påkrevde_aarsprofilkolonner,)


@app.cell(hide_code=True)
def aarsprofil_validering(hent_påkrevde_aarsprofilkolonner):
    def valider_aarsprofil_input(
        df: pl.DataFrame,
        vindusstorrelse: int,
    ) -> None:
        """Valider observasjonsdata og vindusstørrelse for årsprofilen.

        Args:
            df: Ferdig behandlet observasjonsdata.
            vindusstorrelse: Antall datapunkter i det sentrerte rullevinduet.

        Raises:
            TypeError: Når input, dato, antall eller vindusstørrelse har feil type.
            ValueError: Når kolonner mangler eller data bryter inputkontrakten.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("Årsprofil krever en Polars DataFrame")
        if isinstance(vindusstorrelse, bool) or not isinstance(vindusstorrelse, int):
            raise TypeError("Vindusstørrelse må være et heltall")
        if vindusstorrelse < 1:
            raise ValueError("Vindusstørrelse må være minst 1")

        manglende_kolonner = sorted(hent_påkrevde_aarsprofilkolonner() - set(df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler obligatoriske kolonner for årsprofil: " + ", ".join(manglende_kolonner))

        if df.schema["Observert dato"].base_type() not in {pl.Date, pl.Datetime}:
            raise TypeError("Kolonnen `Observert dato` må ha typen Date eller Datetime")
        if not df.schema["Antall"].is_integer():
            raise TypeError("Kolonnen `Antall` må ha heltallstype")
        if df.get_column("Observert dato").null_count() > 0:
            raise ValueError("Kolonnen `Observert dato` kan ikke inneholde nullverdier")
        if df.get_column("Antall").null_count() > 0:
            raise ValueError("Kolonnen `Antall` kan ikke inneholde nullverdier")
        if df.filter(pl.col("Antall") < 0).height > 0:
            raise ValueError("Kolonnen `Antall` kan ikke inneholde negative verdier")

    return (valider_aarsprofil_input,)


@app.cell(hide_code=True)
def aarsprofil_aggregering(
    AARSPROFIL_OUTPUTKOLONNER,
    valider_aarsprofil_input,
):
    def lag_aarsprofil(
        df: pl.DataFrame,
        vindusstorrelse: int = 7,
    ) -> pl.DataFrame:
        """Beregn daglig årsprofil for observasjoner og individer.

        Observasjoner aggregeres først per faktisk dato. Samme måned og dag fra
        ulike år samles deretter på en felles skuddårskalender. Dager uten
        registreringer tolkes ikke som null observasjoner.
        """
        valider_aarsprofil_input(df, vindusstorrelse)

        daglig = (
            df.select(
                pl.col("Observert dato").cast(pl.Date).alias("__dato"),
                pl.col("Antall").alias("__antall"),
            )
            .group_by("__dato")
            .agg(
                pl.len().cast(pl.Int64).alias("__observasjoner"),
                pl.col("__antall").sum().cast(pl.Int64).alias("__individer"),
            )
            .with_columns(
                pl.date(
                    2000,
                    pl.col("__dato").dt.month(),
                    pl.col("__dato").dt.day(),
                ).alias("Dato")
            )
        )

        return (
            daglig.group_by("Dato")
            .agg(
                pl.col("__observasjoner").mean().alias("Gjennomsnitt observasjoner"),
                pl.col("__individer").mean().alias("Gjennomsnitt individer"),
                pl.col("__observasjoner").std().fill_null(0.0).alias("__standardavvik_observasjoner"),
                pl.col("__individer").std().fill_null(0.0).alias("__standardavvik_individer"),
                pl.len().cast(pl.Int64).alias("Antall år"),
            )
            .sort("Dato")
            .with_columns(
                (pl.col("__standardavvik_observasjoner") / pl.col("Antall år").sqrt()).alias(
                    "Standardfeil observasjoner"
                ),
                (pl.col("__standardavvik_individer") / pl.col("Antall år").sqrt()).alias("Standardfeil individer"),
                pl.col("Gjennomsnitt observasjoner")
                .rolling_mean(vindusstorrelse, center=True, min_samples=1)
                .alias("Rullerende observasjoner"),
                pl.col("Gjennomsnitt individer")
                .rolling_mean(vindusstorrelse, center=True, min_samples=1)
                .alias("Rullerende individer"),
            )
            .with_columns(
                (pl.col("Rullerende observasjoner") - pl.col("Standardfeil observasjoner")).alias(
                    "Nedre observasjoner"
                ),
                (pl.col("Rullerende observasjoner") + pl.col("Standardfeil observasjoner")).alias("Øvre observasjoner"),
                (pl.col("Rullerende individer") - pl.col("Standardfeil individer")).alias("Nedre individer"),
                (pl.col("Rullerende individer") + pl.col("Standardfeil individer")).alias("Øvre individer"),
            )
            .select(AARSPROFIL_OUTPUTKOLONNER)
        )

    return (lag_aarsprofil,)


@app.cell(hide_code=True)
def kartleggingsintensitet_aggregering(
    KARTLEGGINGSINTENSITET_OUTPUTKOLONNER,
    valider_aarsprofil_input,
):
    def lag_kartleggingsintensitet(
        valgte_df: pl.DataFrame,
        normaliseringsgrunnlag_df: pl.DataFrame,
        vindusstorrelse: int = 7,
    ) -> pl.DataFrame:
        """Normaliser valgte observasjoner mot all kartleggingsaktivitet.

        Intensiteten er prosentandelen valgte observasjoner av alle observasjoner
        i det samme sentrerte rullevinduet. Normaliseringsgrunnlaget skal være det
        komplette, ufiltrerte datasettet som `valgte_df` er hentet fra.
        """
        valider_aarsprofil_input(valgte_df, vindusstorrelse)
        valider_aarsprofil_input(normaliseringsgrunnlag_df, vindusstorrelse)

        alle_daglig = (
            normaliseringsgrunnlag_df.select(pl.col("Observert dato").cast(pl.Date).alias("__dato"))
            .group_by("__dato")
            .agg(pl.len().cast(pl.Int64).alias("__alle_observasjoner"))
        )
        valgte_daglig = (
            valgte_df.select(pl.col("Observert dato").cast(pl.Date).alias("__dato"))
            .group_by("__dato")
            .agg(pl.len().cast(pl.Int64).alias("__valgte_observasjoner"))
        )

        ukjente_datoer = valgte_daglig.join(
            alle_daglig.select("__dato"),
            on="__dato",
            how="anti",
        )
        if ukjente_datoer.height > 0:
            raise ValueError("Valgte observasjoner må være en del av normaliseringsgrunnlaget")

        daglig = alle_daglig.join(
            valgte_daglig,
            on="__dato",
            how="left",
        ).with_columns(pl.col("__valgte_observasjoner").fill_null(0))
        if daglig.filter(pl.col("__valgte_observasjoner") > pl.col("__alle_observasjoner")).height > 0:
            raise ValueError("Valgte data kan ikke ha flere observasjoner enn normaliseringsgrunnlaget på samme dato")

        return (
            daglig.with_columns(
                pl.date(
                    2000,
                    pl.col("__dato").dt.month(),
                    pl.col("__dato").dt.day(),
                ).alias("Dato")
            )
            .group_by("Dato")
            .agg(
                pl.col("__valgte_observasjoner").sum().cast(pl.Int64).alias("Valgte observasjoner"),
                pl.col("__alle_observasjoner").sum().cast(pl.Int64).alias("Alle observasjoner"),
            )
            .sort("Dato")
            .with_columns(
                pl.col("Valgte observasjoner")
                .rolling_sum(vindusstorrelse, center=True, min_samples=1)
                .alias("Valgte observasjoner i vindu"),
                pl.col("Alle observasjoner")
                .rolling_sum(vindusstorrelse, center=True, min_samples=1)
                .alias("Alle observasjoner i vindu"),
            )
            .with_columns(
                (100 * pl.col("Valgte observasjoner i vindu") / pl.col("Alle observasjoner i vindu")).alias(
                    "Kartleggingsintensitet (%)"
                )
            )
            .select(KARTLEGGINGSINTENSITET_OUTPUTKOLONNER)
        )

    return (lag_kartleggingsintensitet,)


@app.cell(hide_code=True)
def aarsprofil_figurfunksjon(
    AARSPROFIL_MAAL,
    AARSPROFIL_OUTPUTKOLONNER,
    KARTLEGGINGSINTENSITET_OUTPUTKOLONNER,
):
    def lag_aarsprofilfigur(
        aarsprofil_df: pl.DataFrame,
        kartleggingsintensitet_df: pl.DataFrame,
        maal: str,
    ) -> alt.LayerChart:
        """Bygg årsprofilfigur med rådata og normalisert intensitet."""
        if not isinstance(aarsprofil_df, pl.DataFrame):
            raise TypeError("Årsprofilfiguren krever en Polars DataFrame")
        if not isinstance(kartleggingsintensitet_df, pl.DataFrame):
            raise TypeError("Kartleggingsintensiteten må være en Polars DataFrame")

        manglende_kolonner = sorted(set(AARSPROFIL_OUTPUTKOLONNER) - set(aarsprofil_df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler kolonner for årsprofilfiguren: " + ", ".join(manglende_kolonner))
        manglende_intensitetskolonner = sorted(
            set(KARTLEGGINGSINTENSITET_OUTPUTKOLONNER) - set(kartleggingsintensitet_df.columns)
        )
        if manglende_intensitetskolonner:
            raise ValueError("Mangler kolonner for kartleggingsintensitet: " + ", ".join(manglende_intensitetskolonner))
        if maal not in AARSPROFIL_MAAL:
            raise ValueError(f"Ukjent mål for årsprofilfiguren: {maal}. Velg Observasjoner eller Individer.")

        oppsett = AARSPROFIL_MAAL[maal]
        x_akse = alt.X(
            field="Dato",
            type="temporal",
            title="Dato",
            axis=alt.Axis(
                format="%d. %b",
                tickCount={"interval": "month", "step": 1},
            ),
        )
        feilbaand = (
            alt.Chart(aarsprofil_df)
            .mark_area(opacity=0.35, color=oppsett["båndfarge"])
            .encode(
                x=x_akse,
                y=alt.Y(
                    field=oppsett["nedre"],
                    type="quantitative",
                    title=oppsett["y-tittel"],
                ),
                y2=alt.Y2(field=oppsett["øvre"]),
            )
        )
        linje = (
            alt.Chart(aarsprofil_df)
            .mark_line(
                point={"filled": True, "size": 28},
                strokeWidth=2,
                color=oppsett["farge"],
            )
            .encode(
                x=x_akse,
                y=alt.Y(
                    field=oppsett["rullerende"],
                    type="quantitative",
                    title=oppsett["y-tittel"],
                ),
                tooltip=[
                    alt.Tooltip(
                        field="Dato",
                        type="temporal",
                        title="Dato",
                        format="%d. %B",
                    ),
                    alt.Tooltip(
                        field=oppsett["gjennomsnitt"],
                        type="quantitative",
                        title="Daglig gjennomsnitt",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        field=oppsett["rullerende"],
                        type="quantitative",
                        title="Rullerende gjennomsnitt",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        field=oppsett["standardfeil"],
                        type="quantitative",
                        title="Standardfeil",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        field="Antall år",
                        type="quantitative",
                        title="År med data",
                    ),
                ],
            )
        )
        raaserie = alt.layer(feilbaand, linje)
        normalisert_linje = (
            alt.Chart(kartleggingsintensitet_df)
            .mark_line(
                color="#047857",
                strokeWidth=2.5,
                strokeDash=[7, 4],
            )
            .encode(
                x=x_akse,
                y=alt.Y(
                    field="Kartleggingsintensitet (%)",
                    type="quantitative",
                    title="Kartleggingsintensitet (%)",
                    scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(
                        orient="right",
                        format=".0f",
                        grid=False,
                        titleColor="#047857",
                        labelColor="#047857",
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        field="Dato",
                        type="temporal",
                        title="Dato",
                        format="%d. %B",
                    ),
                    alt.Tooltip(
                        field="Kartleggingsintensitet (%)",
                        type="quantitative",
                        title="Kartleggingsintensitet",
                        format=".1f",
                    ),
                    alt.Tooltip(
                        field="Valgte observasjoner i vindu",
                        type="quantitative",
                        title="Valgte observasjoner i vinduet",
                        format="d",
                    ),
                    alt.Tooltip(
                        field="Alle observasjoner i vindu",
                        type="quantitative",
                        title="Alle observasjoner i vinduet",
                        format="d",
                    ),
                ],
            )
        )

        return (
            alt.layer(raaserie, normalisert_linje)
            .resolve_scale(y="independent")
            .properties(
                width=900,
                height=400,
                title=oppsett["tittel"],
            )
            .interactive(bind_y=False)
        )

    return (lag_aarsprofilfigur,)


@app.cell(hide_code=True)
def aarsprofil_funksjonsvisning(
    hent_påkrevde_aarsprofilkolonner,
    lag_aarsprofil,
    lag_aarsprofilfigur,
    lag_kartleggingsintensitet,
    valider_aarsprofil_input,
):
    def _vis_aarsprofil_kildekode(_funksjon):
        _kildekode = textwrap.dedent(inspect.getsource(_funksjon)).strip()
        return mo.md(f"#### `{_funksjon.__name__}`\n\n```python\n{_kildekode}\n```")

    _aarsprofil_funksjoner = [
        hent_påkrevde_aarsprofilkolonner,
        valider_aarsprofil_input,
        lag_aarsprofil,
        lag_kartleggingsintensitet,
        lag_aarsprofilfigur,
    ]
    _aarsprofil_funksjonsinnhold = mo.vstack(
        [
            mo.md(
                "Funksjonene under validerer observasjonsdata, beregner årsprofil "
                "og normalisert kartleggingsintensitet med Polars, og bygger "
                "Altair-figuren."
            ),
            *[_vis_aarsprofil_kildekode(_funksjon) for _funksjon in _aarsprofil_funksjoner],
        ]
    )
    aarsprofil_funksjonsseksjon = mo.accordion({"Funksjoner for årsprofilen": _aarsprofil_funksjonsinnhold})
    aarsprofil_funksjonsseksjon
    return


@app.cell(hide_code=True)
def aarsprofil_testdata(AARSPROFIL_OUTPUTKOLONNER):
    def lag_aarsprofil_testinput(
        rad_overrides: list[dict[str, object]] | None = None,
    ) -> pl.DataFrame:
        """Lag en liten fixture for årsprofiltestene."""
        if rad_overrides is None:
            rad_overrides = [{}]

        grunnrad = {
            "Observert dato": date(2020, 1, 1),
            "Antall": 1,
        }
        return pl.DataFrame(
            [{**grunnrad, **overrides} for overrides in rad_overrides],
            schema_overrides={
                "Observert dato": pl.Date,
                "Antall": pl.Int64,
            },
        )

    def lag_tom_aarsprofil_testinput() -> pl.DataFrame:
        """Lag tom årsprofil-input med riktig schema."""
        return pl.DataFrame(
            {
                "Observert dato": pl.Series([], dtype=pl.Date),
                "Antall": pl.Series([], dtype=pl.Int64),
            }
        )

    def aarsprofil_forventede_kolonner() -> list[str]:
        """Returner godkjent kolonnerekkefølge for årsprofilen."""
        return list(AARSPROFIL_OUTPUTKOLONNER)

    return (
        aarsprofil_forventede_kolonner,
        lag_aarsprofil_testinput,
        lag_tom_aarsprofil_testinput,
    )


@app.cell(hide_code=True)
def test_aarsprofil_001(lag_aarsprofil, lag_aarsprofil_testinput):
    def test_aarsprofil_mtm_001():
        test_df = lag_aarsprofil_testinput(
            [
                {"Observert dato": date(2020, 1, 1), "Antall": 2},
                {"Observert dato": date(2020, 1, 1), "Antall": 4},
                {"Observert dato": date(2021, 1, 1), "Antall": 3},
                {"Observert dato": date(2020, 1, 2), "Antall": 10},
            ]
        )
        result = lag_aarsprofil(test_df, vindusstorrelse=1)
        nyttarsdag = result.filter(pl.col("Dato") == date(2000, 1, 1)).row(
            0,
            named=True,
        )

        assert result["Dato"].to_list() == [date(2000, 1, 1), date(2000, 1, 2)]
        assert nyttarsdag["Gjennomsnitt observasjoner"] == 1.5
        assert nyttarsdag["Gjennomsnitt individer"] == 4.5
        assert nyttarsdag["Standardfeil observasjoner"] == 0.5
        assert abs(nyttarsdag["Standardfeil individer"] - 1.5) < 1e-12
        assert nyttarsdag["Antall år"] == 2
        assert nyttarsdag["Nedre individer"] == 3.0
        assert nyttarsdag["Øvre individer"] == 6.0

    test_aarsprofil_mtm_001()
    return (test_aarsprofil_mtm_001,)


@app.cell(hide_code=True)
def test_aarsprofil_002(lag_aarsprofil, lag_aarsprofil_testinput):
    def test_aarsprofil_mtm_002():
        test_df = lag_aarsprofil_testinput(
            [
                {"Observert dato": date(2019, 1, 1), "Antall": 1},
                {"Observert dato": date(2020, 1, 2), "Antall": 2},
                {"Observert dato": date(2020, 2, 29), "Antall": 3},
            ]
        )
        result = lag_aarsprofil(test_df, vindusstorrelse=3)

        assert result["Dato"].to_list() == [
            date(2000, 1, 1),
            date(2000, 1, 2),
            date(2000, 2, 29),
        ]
        assert result["Rullerende observasjoner"].to_list() == [1.0, 1.0, 1.0]
        assert result["Rullerende individer"].to_list() == [1.5, 2.0, 2.5]
        assert result["Standardfeil individer"].to_list() == [0.0, 0.0, 0.0]

    test_aarsprofil_mtm_002()
    return (test_aarsprofil_mtm_002,)


@app.cell(hide_code=True)
def test_aarsprofil_003(lag_aarsprofil, lag_aarsprofil_testinput):
    def test_aarsprofil_mtm_003():
        mangler_dato = lag_aarsprofil_testinput().drop("Observert dato")
        try:
            lag_aarsprofil(mangler_dato)
        except ValueError as exc:
            assert "Observert dato" in str(exc)
        else:
            raise AssertionError("Manglende Observert dato skulle gitt ValueError")

        feil_antall = pl.DataFrame(
            {
                "Observert dato": [date(2020, 1, 1)],
                "Antall": ["én"],
            }
        )
        try:
            lag_aarsprofil(feil_antall)
        except TypeError as exc:
            assert "Antall" in str(exc)
        else:
            raise AssertionError("Antall med teksttype skulle gitt TypeError")

    test_aarsprofil_mtm_003()
    return (test_aarsprofil_mtm_003,)


@app.cell(hide_code=True)
def test_aarsprofil_004(lag_aarsprofil, lag_aarsprofil_testinput):
    def test_aarsprofil_mtm_004():
        negativt_antall = lag_aarsprofil_testinput([{"Antall": -1}])
        try:
            lag_aarsprofil(negativt_antall)
        except ValueError as exc:
            assert "negative" in str(exc)
        else:
            raise AssertionError("Negativt Antall skulle gitt ValueError")

        try:
            lag_aarsprofil(lag_aarsprofil_testinput(), vindusstorrelse=0)
        except ValueError as exc:
            assert "minst 1" in str(exc)
        else:
            raise AssertionError("Vindusstørrelse 0 skulle gitt ValueError")

    test_aarsprofil_mtm_004()
    return (test_aarsprofil_mtm_004,)


@app.cell(hide_code=True)
def test_aarsprofil_005(
    aarsprofil_forventede_kolonner,
    lag_aarsprofil,
    lag_tom_aarsprofil_testinput,
):
    def test_aarsprofil_mtm_005():
        result = lag_aarsprofil(lag_tom_aarsprofil_testinput())

        assert result.height == 0
        assert result.columns == aarsprofil_forventede_kolonner()
        assert result.schema["Dato"] == pl.Date
        assert result.schema["Antall år"] == pl.Int64
        assert result.schema["Rullerende observasjoner"] == pl.Float64
        assert result.schema["Rullerende individer"] == pl.Float64

    test_aarsprofil_mtm_005()
    return (test_aarsprofil_mtm_005,)


@app.cell(hide_code=True)
def test_aarsprofil_006(
    lag_aarsprofil,
    lag_aarsprofil_testinput,
    lag_aarsprofilfigur,
    lag_kartleggingsintensitet,
):
    def test_aarsprofil_mtm_006():
        test_df = lag_aarsprofil_testinput()
        profil = lag_aarsprofil(test_df, vindusstorrelse=1)
        intensitet = lag_kartleggingsintensitet(
            test_df,
            test_df,
            vindusstorrelse=1,
        )

        for maal, forventet_tittel in [
            ("Observasjoner", "Observasjoner gjennom året"),
            ("Individer", "Individer gjennom året"),
        ]:
            figur = lag_aarsprofilfigur(profil, intensitet, maal)
            spesifikasjon = figur.to_dict()
            raalag = spesifikasjon["layer"][0]["layer"]
            intensitetslag = spesifikasjon["layer"][1]

            assert isinstance(figur, alt.LayerChart)
            assert spesifikasjon["title"] == forventet_tittel
            assert spesifikasjon["resolve"]["scale"]["y"] == "independent"
            assert raalag[0]["mark"]["type"] == "area"
            assert raalag[1]["mark"]["type"] == "line"
            assert intensitetslag["mark"]["type"] == "line"
            assert intensitetslag["encoding"]["y"]["axis"]["orient"] == "right"

    test_aarsprofil_mtm_006()
    return (test_aarsprofil_mtm_006,)


@app.cell(hide_code=True)
def test_kartleggingsintensitet_007(
    KARTLEGGINGSINTENSITET_OUTPUTKOLONNER,
    lag_aarsprofil_testinput,
    lag_kartleggingsintensitet,
    lag_tom_aarsprofil_testinput,
):
    def test_aarsprofil_mtm_007():
        alle_df = lag_aarsprofil_testinput(
            [
                {"Observert dato": date(2020, 1, 1)},
                {"Observert dato": date(2020, 1, 1)},
                {"Observert dato": date(2020, 1, 1)},
                {"Observert dato": date(2020, 1, 1)},
                {"Observert dato": date(2020, 1, 2)},
                {"Observert dato": date(2020, 1, 2)},
            ]
        )
        valgte_df = lag_aarsprofil_testinput(
            [
                {"Observert dato": date(2020, 1, 1)},
                {"Observert dato": date(2020, 1, 2)},
            ]
        )
        result = lag_kartleggingsintensitet(
            valgte_df,
            alle_df,
            vindusstorrelse=1,
        )

        assert result["Dato"].to_list() == [date(2000, 1, 1), date(2000, 1, 2)]
        assert result["Valgte observasjoner"].to_list() == [1, 1]
        assert result["Alle observasjoner"].to_list() == [4, 2]
        assert result["Kartleggingsintensitet (%)"].to_list() == [25.0, 50.0]

        tomt_resultat = lag_kartleggingsintensitet(
            lag_tom_aarsprofil_testinput(),
            lag_tom_aarsprofil_testinput(),
        )
        assert tomt_resultat.columns == KARTLEGGINGSINTENSITET_OUTPUTKOLONNER
        assert tomt_resultat.schema["Kartleggingsintensitet (%)"] == pl.Float64

    test_aarsprofil_mtm_007()
    return (test_aarsprofil_mtm_007,)


@app.cell(hide_code=True)
def aarsprofil_testvisning(
    aarsprofil_forventede_kolonner,
    lag_aarsprofil_testinput,
    lag_tom_aarsprofil_testinput,
    test_aarsprofil_mtm_001,
    test_aarsprofil_mtm_002,
    test_aarsprofil_mtm_003,
    test_aarsprofil_mtm_004,
    test_aarsprofil_mtm_005,
    test_aarsprofil_mtm_006,
    test_aarsprofil_mtm_007,
):
    def _vis_aarsprofil_testkode(_funksjon):
        _kildekode = textwrap.dedent(inspect.getsource(_funksjon)).strip()
        return mo.md(f"#### `{_funksjon.__name__}`\n\n```python\n{_kildekode}\n```")

    _aarsprofil_testmatrise = mo.md(r"""
    ### Testmatrise

    | ID | Scenario | Forventet resultat |
    |---|---|---|
    | ÅRSPROFIL-MTM-001 | Flere registreringer på samme dato i to år | Korrekt dagsaggregering, gjennomsnitt og standardfeil |
    | ÅRSPROFIL-MTM-002 | Tre kalenderdager, inkludert 29. februar | Sortert skuddårskalender og korrekt sentrert rullevindu |
    | ÅRSPROFIL-MTM-003 | Manglende kolonne eller feil datatype | Tidlig og forklarende feil |
    | ÅRSPROFIL-MTM-004 | Negativt antall eller ugyldig vindusstørrelse | Tidlig og forklarende feil |
    | ÅRSPROFIL-MTM-005 | Tom input med riktig schema | Tom output med fast kolonnerekkefølge og riktige typer |
    | ÅRSPROFIL-MTM-006 | Rendering av rådata og normalisert intensitet | Delt venstre/høyre y-akse med feilbånd og to linjer |
    | ÅRSPROFIL-MTM-007 | Valgte observasjoner normalisert mot alle observasjoner | Korrekt prosentandel per rullevindu og stabil tom output |
    """)
    _aarsprofil_testfunksjoner = [
        lag_aarsprofil_testinput,
        lag_tom_aarsprofil_testinput,
        aarsprofil_forventede_kolonner,
        test_aarsprofil_mtm_001,
        test_aarsprofil_mtm_002,
        test_aarsprofil_mtm_003,
        test_aarsprofil_mtm_004,
        test_aarsprofil_mtm_005,
        test_aarsprofil_mtm_006,
        test_aarsprofil_mtm_007,
    ]
    _aarsprofil_testinnhold = mo.vstack(
        [
            mo.md(
                "Testene kjøres reaktivt og dekker inputkontrakt, aggregering, "
                "rullevindu, standardfeil, normalisering, tom input og rendering."
            ),
            _aarsprofil_testmatrise,
            *[_vis_aarsprofil_testkode(_funksjon) for _funksjon in _aarsprofil_testfunksjoner],
        ]
    )
    aarsprofil_testseksjon = mo.accordion({"Tester og testmatrise for årsprofilen": _aarsprofil_testinnhold})
    aarsprofil_testseksjon
    return


@app.cell(hide_code=True)
def _(valgt_fil):
    mo.stop(
        not valgt_fil.value,
        mo.md("Velg en ferdig behandlet Parquet-fil for å starte analysen."),
    )

    file_info = valgt_fil.value[0]
    arter_df_lest_inn = pl.read_parquet(file_info.path)
    artsdata_df = mo.ui.table(arter_df_lest_inn, page_size=20)
    return arter_df_lest_inn, artsdata_df


@app.cell(hide_code=True)
def _(artsdata_df):
    arter_df = artsdata_df.value
    return (arter_df,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def aarsprofil_overskrift():
    mo.md(r"""
    ## Observasjoner og individer gjennom året

    Figuren viser gjennomsnittet for samme kalenderdag på tvers av år. Rådatalinjen
    er et sentrert rullerende gjennomsnitt, og det fargede feltet viser ± én
    standardfeil. Den grønne, stiplede linjen bruker høyre y-akse og viser valgte
    observasjoner som prosent av alle observasjoner i samme rullevindu. Dager uten
    registreringer i normaliseringsgrunnlaget regnes ikke som nullobservasjoner.
    """)
    return


@app.cell(hide_code=True)
def aarsprofil_kontroller():
    aarsprofil_vis_individer = mo.ui.switch(
        label="Vis individer",
        value=False,
    )
    aarsprofil_vindusstorrelse = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=7,
        show_value=True,
        label="Antall datapunkter i rullerende gjennomsnitt",
    )

    mo.hstack(
        [aarsprofil_vis_individer, aarsprofil_vindusstorrelse],
        justify="start",
        gap=2,
    )
    return aarsprofil_vindusstorrelse, aarsprofil_vis_individer


@app.cell(hide_code=True)
def aarsprofil_beregning(
    aarsprofil_vindusstorrelse,
    arter_df,
    arter_df_lest_inn,
    lag_aarsprofil,
    lag_kartleggingsintensitet,
):
    aarsprofil_df = lag_aarsprofil(
        arter_df,
        vindusstorrelse=aarsprofil_vindusstorrelse.value,
    )
    kartleggingsintensitet_df = lag_kartleggingsintensitet(
        arter_df,
        arter_df_lest_inn,
        vindusstorrelse=aarsprofil_vindusstorrelse.value,
    )
    return aarsprofil_df, kartleggingsintensitet_df


@app.cell(hide_code=True)
def aarsprofil_visning(
    aarsprofil_df,
    aarsprofil_vis_individer,
    kartleggingsintensitet_df,
    lag_aarsprofilfigur,
):
    aarsprofil_maal = "Individer" if aarsprofil_vis_individer.value else "Observasjoner"
    aarsprofil_figur = mo.ui.altair_chart(
        lag_aarsprofilfigur(
            aarsprofil_df,
            kartleggingsintensitet_df,
            aarsprofil_maal,
        ),
        chart_selection=False,
        legend_selection=False,
    )
    aarsprofil_figur
    return


@app.cell(column=1, hide_code=True)
def _(artsdata_df):
    artsdata_df
    return


@app.cell(hide_code=True)
def _(artsstatistikk_df, lag_artsstatistikk_tabell):
    artsstatistikk_tabell = lag_artsstatistikk_tabell(artsstatistikk_df)
    artsstatistikk_tabell
    return


@app.cell(hide_code=True)
def _(arter_df, plotly_map, plotly_map_fig):
    def get_selected_row_nrs(points, figure):
        """For every selected map point, use its curveNumber to find the right Plotly trace, use its pointIndex to find the right point inside that trace, look in that point’s hidden customdata, take the first value, convert it to an integer, and return all those integers as a list."""
        return [int(figure.data[point["curveNumber"]].customdata[point["pointIndex"]][0]) for point in points]

    selected_row_nrs = get_selected_row_nrs(plotly_map.points, plotly_map_fig)

    selected_arter_df = (
        arter_df.with_row_index("__row_nr").filter(pl.col("__row_nr").is_in(selected_row_nrs)).drop("__row_nr")
    )

    mo.vstack(
        [
            mo.md(f"**Valgte observasjoner fra heatmap:** {selected_arter_df.height}"),
            mo.ui.table(selected_arter_df, page_size=10),
        ]
    )
    return (selected_arter_df,)


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    #Kart
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Selekteringskart
    """)
    return


@app.cell(hide_code=True)
def _():
    farge_kart_arter = mo.ui.dropdown(
        options=["Navn", "Verdi M1941", "Atferd"],
        value="Verdi M1941",
        label="Farge på punkter (punkter hvor atferd ikke er registrert vises ikke i kartet)",
    )

    mo.vstack(
        [
            farge_kart_arter,
            mo.md(
                "*Merk: Punkter uten registrert atferd vises ikke når kartet fargelegges etter atferd (eller punkter med null values)*"
            ),
        ]
    )
    return (farge_kart_arter,)


@app.cell(hide_code=True)
def plotlymap(arter_df, farge_kart_arter):
    verdi_m1941_color_map = {
        "Svært stor verdi": "#AF0F0F",
        "Stor verdi": "#FD7032",
        "Middels verdi": "#FEC02D",
        "Noe verdi": "#FFFF00",
        "Uten betydning for KU": "#D9D9D9",
        "Ikke definert": "#000000",
    }

    verdi_m1941_draw_order = [
        "Uten betydning for KU",
        "Ikke definert",
        "Noe verdi",
        "Middels verdi",
        "Stor verdi",
        "Svært stor verdi",
    ]

    atferd_priority_order = [
        "reproductive",
        "possiblereproductive",
        "feeding",
        "stationary",
        "moving",
        "dead",
    ]
    atferd_draw_order = list(reversed(atferd_priority_order))

    plotly_color_kwargs = {}
    if farge_kart_arter.value == "Verdi M1941":
        plotly_color_kwargs = {
            "color_discrete_map": verdi_m1941_color_map,
            "category_orders": {"Verdi M1941": verdi_m1941_draw_order},
        }
    elif farge_kart_arter.value == "Atferd":
        plotly_color_kwargs = {
            "category_orders": {"Atferd": atferd_draw_order},
        }

    plotly_arter_df = arter_df.with_row_index("__row_nr")
    plotly_kartflis_lag = [
        {
            "below": "traces",
            "source": ["https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"],
            "sourceattribution": "© OpenStreetMap-bidragsytere © CARTO",
            "sourcetype": "raster",
        }
    ]

    plotly_map_fig = px.scatter_map(
        plotly_arter_df,
        lat="latitude",
        lon="longitude",
        hover_name="Navn",
        hover_data=[
            "Antall",
            "Kategori",
            "Art av nasjonal forvaltningsinteresse (eks. rødlista)",
            "Atferd",
            "Observert dato",
            "Verdi M1941",
            "Art",
        ],
        custom_data=["__row_nr", "Artens ID", "Navn"],
        zoom=8,
        height=650,
        map_style="white-bg",
        color=farge_kart_arter.value,
        **plotly_color_kwargs,
    )
    plotly_map_fig.update_traces(
        marker={"size": 8, "opacity": 0.9},
        selected={"marker": {"size": 11, "opacity": 1.0}},
        unselected={"marker": {"opacity": 0.35}},
    )
    plotly_map_fig.update_layout(
        dragmode="lasso",
        clickmode="event+select",
        map_layers=plotly_kartflis_lag,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )
    plotly_map = mo.ui.plotly(
        plotly_map_fig,
        config={"scrollZoom": True, "displaylogo": False},
    )
    plotly_map
    return plotly_map, plotly_map_fig


@app.cell
def _():
    mo.md(r"""
    #Heatmap
    """)
    return


@app.cell(hide_code=True)
def test(selected_arter_df):
    arter_pdf = selected_arter_df.to_pandas()

    arter_gdf = gpd.GeoDataFrame(
        arter_pdf,
        geometry=gpd.points_from_xy(arter_pdf["longitude"], arter_pdf["latitude"]),
        crs="EPSG:4326",  # lat/lon
    ).to_crs("EPSG:3857")  # Web Mercator for kartfliser

    arter_map_df = pl.from_pandas(
        arter_gdf.assign(
            x_webmercator=arter_gdf.geometry.x,
            y_webmercator=arter_gdf.geometry.y,
        ).drop(columns="geometry")
    )
    return (arter_map_df,)


@app.cell(hide_code=True)
def _():
    max_px_value = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=10,
        show_value=True,
        label="Maks spredning",
    )
    threshold_value = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.95,
        show_value=True,
        label="Terskel",
    )
    return max_px_value, threshold_value


@app.cell(hide_code=True)
def _(max_px_value, threshold_value):
    stack = mo.vstack(
        [
            max_px_value,
            mo.md("*Største antall piksler punktene kan utvides på hver side.*"),
            threshold_value,
            mo.md("*Tettheten som må nås før spredningen stopper. En høyere terskel gir mer spredning.*"),
        ]
    )

    stack
    return


@app.cell(hide_code=True)
def heatmap(arter_map_df, max_px_value, threshold_value):
    species_density = arter_map_df.hvplot.points(
        x="x_webmercator",
        y="y_webmercator",
        rasterize=True,
        dynspread=True,
        max_px=max_px_value.value,
        threshold=threshold_value.value,
        aggregator=ds.count(),
        cnorm="eq_hist",
        cmap=cc.fire[100:],
        width=900,
        height=700,
        xaxis=None,
        yaxis=None,
    )

    EsriImagery().opts(alpha=0.75) * species_density
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
