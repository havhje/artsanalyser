import marimo

__generated_with = "0.23.16"
app = marimo.App(width="columns", layout_file="layouts/data_analyse.grid.json")

with app.setup(hide_code=True):
    import inspect
    import textwrap
    from datetime import date

    import altair as alt
    import colorcet as cc
    import datashader as ds
    import geopandas as gpd
    import great_tables as gt
    import holoviews as hv
    import holoviews.operation.datashader as h
    import hvplot.polars
    import leafmap.foliumap as leafmap
    import marimo as mo
    import plotly.express as px
    import polars as pl
    from holoviews.element.tiles import EsriImagery


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
def _():
    def _valider_sesongprofilgrunnlag(
        tabellgrunnlag: pl.DataFrame,
        innsatsgrunnlag: pl.DataFrame,
        artens_id: int,
        vindusstorrelse: int,
        *,
        krever_antall: bool,
    ) -> None:
        """Valider felles input til sesongprofilberegningene."""
        if not isinstance(tabellgrunnlag, pl.DataFrame) or not isinstance(innsatsgrunnlag, pl.DataFrame):
            raise TypeError("Sesongprofil krever Polars DataFrame som tabell- og innsatsgrunnlag")
        if isinstance(vindusstorrelse, bool) or not isinstance(vindusstorrelse, int):
            raise TypeError("Vindusstørrelsen må være et heltall")
        if not 1 <= vindusstorrelse <= 366:
            raise ValueError("Vindusstørrelsen må være mellom 1 og 366 dager")

        mangler_tabell = sorted({"Artens ID", "Observert dato"} - set(tabellgrunnlag.columns))
        mangler_innsats = sorted({"Observert dato"} - set(innsatsgrunnlag.columns))
        if krever_antall and "Antall" not in tabellgrunnlag.columns:
            mangler_tabell.append("Antall")
        if mangler_tabell:
            raise ValueError("Mangler kolonner i tabellgrunnlaget: " + ", ".join(mangler_tabell))
        if mangler_innsats:
            raise ValueError("Mangler kolonner i innsatsgrunnlaget: " + ", ".join(mangler_innsats))

        for navn, grunnlag in (("tabellgrunnlaget", tabellgrunnlag), ("innsatsgrunnlaget", innsatsgrunnlag)):
            if grunnlag.schema["Observert dato"].base_type() not in {pl.Date, pl.Datetime}:
                raise TypeError(f"`Observert dato` i {navn} må ha typen Date eller Datetime")
        if not tabellgrunnlag.schema["Artens ID"].is_integer():
            raise TypeError("`Artens ID` i tabellgrunnlaget må ha heltallstype")
        if krever_antall:
            if not tabellgrunnlag.schema["Antall"].is_numeric():
                raise TypeError("`Antall` må ha numerisk datatype for individprofilen")
            if tabellgrunnlag.filter(pl.col("Artens ID") == artens_id).get_column("Antall").null_count() > 0:
                raise ValueError("`Antall` kan ikke inneholde null for valgt art")

    def _beregn_sesongprofil(
        tabellgrunnlag: pl.DataFrame,
        innsatsgrunnlag: pl.DataFrame,
        artens_id: int,
        vindusstorrelse: int,
        *,
        metrikk: str,
        verdikolonne: str | None,
    ) -> pl.DataFrame:
        """Fold alle år til ett kalenderår og beregn rå og innsatsjustert profil."""
        _valider_sesongprofilgrunnlag(
            tabellgrunnlag,
            innsatsgrunnlag,
            artens_id,
            vindusstorrelse,
            krever_antall=verdikolonne is not None,
        )

        _tabell = tabellgrunnlag.with_columns(
            pl.col("Observert dato").cast(pl.Date).alias("__dato")
        ).filter(pl.col("__dato").is_not_null())
        if _tabell.is_empty():
            raise ValueError("Tabellgrunnlaget inneholder ingen gyldige observasjonsdatoer")

        _valgt_art = _tabell.filter(pl.col("Artens ID") == artens_id)
        if _valgt_art.is_empty():
            raise ValueError(f"Fant ingen observasjoner for Artens ID {artens_id}")

        _periode_fra = _tabell.get_column("__dato").min()
        _periode_til = _tabell.get_column("__dato").max()
        _innsats = (
            innsatsgrunnlag.with_columns(pl.col("Observert dato").cast(pl.Date).alias("__dato"))
            .filter(pl.col("__dato").is_not_null())
            .filter(pl.col("__dato").is_between(_periode_fra, _periode_til, closed="both"))
        )

        _verdiuttrykk = (
            pl.len().cast(pl.Float64)
            if verdikolonne is None
            else pl.col(verdikolonne).sum().cast(pl.Float64)
        )
        _valgt_per_dato = _valgt_art.group_by("__dato").agg(_verdiuttrykk.alias("__valgt_verdi"))
        _innsats_per_dato = _innsats.group_by("__dato").agg(
            pl.len().cast(pl.Float64).alias("__innsats")
        )
        _kalender = pl.DataFrame(
            {
                "__dato": pl.date_range(
                    _periode_fra,
                    _periode_til,
                    interval="1d",
                    eager=True,
                )
            }
        )
        _sesonggrunnlag = (
            _kalender.join(_valgt_per_dato, on="__dato", how="left")
            .join(_innsats_per_dato, on="__dato", how="left")
            .with_columns(pl.col(["__valgt_verdi", "__innsats"]).fill_null(0.0))
            .with_columns(
                pl.date(
                    2000,
                    pl.col("__dato").dt.month(),
                    pl.col("__dato").dt.day(),
                ).alias("Sesongdato")
            )
            .group_by("Sesongdato")
            .agg(
                pl.col("__valgt_verdi").sum().alias("__verdi_sum"),
                pl.col("__innsats").sum().alias("__innsats_sum"),
                pl.len().cast(pl.Float64).alias("__datodager"),
            )
            .sort("Sesongdato")
        )

        _antall_sesongdager = _sesonggrunnlag.height
        _vindu = min(vindusstorrelse, _antall_sesongdager)
        _venstre = (_vindu - 1) // 2
        _hoeyre = _vindu - 1 - _venstre
        _utvidede_deler = []
        if _venstre:
            _utvidede_deler.append(_sesonggrunnlag.tail(_venstre))
        _utvidede_deler.append(_sesonggrunnlag)
        if _hoeyre:
            _utvidede_deler.append(_sesonggrunnlag.head(_hoeyre))
        _utvidet = pl.concat(_utvidede_deler)
        _rullerende = (
            _utvidet.select(
                pl.col("__verdi_sum").rolling_sum(window_size=_vindu).alias("Verdi i vindu"),
                pl.col("__innsats_sum")
                .rolling_sum(window_size=_vindu)
                .alias("Alle artsobservasjoner i vindu"),
                pl.col("__datodager").rolling_sum(window_size=_vindu).alias("Datodager i vindu"),
            )
            .slice(_vindu - 1, _antall_sesongdager)
        )
        _maanedsnavn = {
            1: "jan.",
            2: "feb.",
            3: "mars",
            4: "apr.",
            5: "mai",
            6: "juni",
            7: "juli",
            8: "aug.",
            9: "sep.",
            10: "okt.",
            11: "nov.",
            12: "des.",
        }

        return (
            _sesonggrunnlag.hstack(_rullerende)
            .with_columns(
                (pl.col("Verdi i vindu") / pl.col("Datodager i vindu")).alias("Råverdi per dag"),
                pl.when(pl.col("Alle artsobservasjoner i vindu") > 0)
                .then(pl.col("Verdi i vindu") / pl.col("Alle artsobservasjoner i vindu") * 1_000)
                .otherwise(None)
                .alias("Innsatsjustert verdi per 1 000"),
            )
            .with_columns(
                pl.col("Sesongdato").dt.ordinal_day().cast(pl.Int64).alias("Dag i året"),
                pl.concat_str(
                    [
                        pl.col("Sesongdato").dt.day().cast(pl.String),
                        pl.lit(". "),
                        pl.col("Sesongdato")
                        .dt.month()
                        .replace_strict(_maanedsnavn, return_dtype=pl.String),
                    ]
                ).alias("Dato"),
                pl.lit(metrikk).alias("Metrikk"),
            )
            .select(
                "Sesongdato",
                "Dag i året",
                "Dato",
                "Metrikk",
                "Verdi i vindu",
                "Alle artsobservasjoner i vindu",
                "Datodager i vindu",
                "Råverdi per dag",
                "Innsatsjustert verdi per 1 000",
            )
        )

    def beregn_sesongprofil_observasjoner(
        tabellgrunnlag: pl.DataFrame,
        innsatsgrunnlag: pl.DataFrame,
        artens_id: int,
        vindusstorrelse: int = 15,
    ) -> pl.DataFrame:
        """Beregn sesongprofil for observasjoner av valgt art.

        Råprofilen er gjennomsnittlig antall observasjonsrader per kalenderdag.
        Den justerte profilen er observasjoner av arten per 1 000
        artsobservasjoner i samme rullerende datovindu.
        """
        return _beregn_sesongprofil(
            tabellgrunnlag,
            innsatsgrunnlag,
            artens_id,
            vindusstorrelse,
            metrikk="Observasjoner",
            verdikolonne=None,
        )

    def beregn_sesongprofil_individer(
        tabellgrunnlag: pl.DataFrame,
        innsatsgrunnlag: pl.DataFrame,
        artens_id: int,
        vindusstorrelse: int = 15,
    ) -> pl.DataFrame:
        """Beregn sesongprofil for individer av valgt art.

        Råprofilen er gjennomsnittlig antall individer per kalenderdag. Den
        justerte profilen er individer av arten per 1 000 artsobservasjoner i
        samme rullerende datovindu.
        """
        return _beregn_sesongprofil(
            tabellgrunnlag,
            innsatsgrunnlag,
            artens_id,
            vindusstorrelse,
            metrikk="Individer",
            verdikolonne="Antall",
        )

    return beregn_sesongprofil_individer, beregn_sesongprofil_observasjoner


@app.cell(hide_code=True)
def _():
    def lag_sesongprofilfigur(
        sesongprofil_df: pl.DataFrame,
        metrikk: str,
        artstekst: str,
        vindusstorrelse: int,
    ) -> alt.VConcatChart:
        """Lag to samkjørte Altair-paneler for rå og innsatsjustert sesongprofil."""
        paakrevde_kolonner = {
            "Dag i året",
            "Dato",
            "Verdi i vindu",
            "Alle artsobservasjoner i vindu",
            "Datodager i vindu",
            "Råverdi per dag",
            "Innsatsjustert verdi per 1 000",
        }
        manglende_kolonner = sorted(paakrevde_kolonner - set(sesongprofil_df.columns))
        if manglende_kolonner:
            raise ValueError("Mangler kolonner for sesongprofilfiguren: " + ", ".join(manglende_kolonner))
        if sesongprofil_df.is_empty():
            raise ValueError("Kan ikke lage sesongprofilfigur fra tomt datagrunnlag")
        if metrikk not in {"Observasjoner", "Individer"}:
            raise ValueError("Metrikk må være `Observasjoner` eller `Individer`")

        _er_observasjoner = metrikk == "Observasjoner"
        _raatt_aksetittel = "Gj.snitt observasjoner per dag" if _er_observasjoner else "Gj.snitt individer per dag"
        _justert_aksetittel = (
            "Observasjoner per 1 000 artsobservasjoner"
            if _er_observasjoner
            else "Individer per 1 000 artsobservasjoner"
        )
        _verditittel = "Observasjoner i vinduet" if _er_observasjoner else "Individer i vinduet"
        _maanedsstarter = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
        _maanedsuttrykk = (
            "datum.value === 1 ? 'Jan' : datum.value === 32 ? 'Feb' : "
            "datum.value === 61 ? 'Mar' : datum.value === 92 ? 'Apr' : "
            "datum.value === 122 ? 'Mai' : datum.value === 153 ? 'Jun' : "
            "datum.value === 183 ? 'Jul' : datum.value === 214 ? 'Aug' : "
            "datum.value === 245 ? 'Sep' : datum.value === 275 ? 'Okt' : "
            "datum.value === 306 ? 'Nov' : datum.value === 336 ? 'Des' : ''"
        )
        _hover = alt.selection_point(
            name="sesongprofil_hover",
            fields=["Dag i året"],
            nearest=True,
            on="pointerover",
            empty=False,
            clear="pointerout",
        )

        def _xakse(*, vis_etiketter: bool) -> alt.X:
            return alt.X(
                "Dag i året:Q",
                title="Dato i kalenderåret" if vis_etiketter else None,
                scale=alt.Scale(domain=[1, 366], nice=False),
                axis=alt.Axis(
                    values=_maanedsstarter,
                    labelExpr=_maanedsuttrykk,
                    labels=vis_etiketter,
                    ticks=vis_etiketter,
                    domain=vis_etiketter,
                    grid=True,
                    labelPadding=8,
                    titlePadding=12,
                ),
            )

        _raabase = alt.Chart(sesongprofil_df).encode(
            x=_xakse(vis_etiketter=False),
            y=alt.Y(
                "Råverdi per dag:Q",
                title=_raatt_aksetittel,
                scale=alt.Scale(zero=True, nice=True),
            ),
        )
        _raatooltip = [
            alt.Tooltip("Dato:N", title="Dato"),
            alt.Tooltip("Råverdi per dag:Q", title=_raatt_aksetittel, format=".3f"),
            alt.Tooltip("Verdi i vindu:Q", title=_verditittel, format=",.0f"),
            alt.Tooltip("Datodager i vindu:Q", title="Datodager i vinduet", format=",.0f"),
        ]
        _raapanel = alt.layer(
            _raabase.mark_area(color="#64748B", opacity=0.12),
            _raabase.mark_line(color="#475569", strokeWidth=2.4),
            _raabase.mark_point(opacity=0),
            _raabase.mark_circle(color="#334155", size=70).encode(
                opacity=alt.condition(_hover, alt.value(1), alt.value(0)),
                tooltip=_raatooltip,
            ),
            _raabase.mark_rule(color="#94A3B8", strokeWidth=1).encode(
                opacity=alt.condition(_hover, alt.value(0.7), alt.value(0))
            ),
        ).properties(
            height=220,
            width="container",
            title=alt.TitleParams(
                text="Rå sesongprofil",
                subtitle="Gjennomsnitt per kalenderdag på tvers av år",
                anchor="start",
                color="#334155",
                fontSize=15,
                subtitleColor="#64748B",
                subtitleFontSize=12,
            ),
        )

        _justertbase = alt.Chart(sesongprofil_df).encode(
            x=_xakse(vis_etiketter=True),
            y=alt.Y(
                "Innsatsjustert verdi per 1 000:Q",
                title=_justert_aksetittel,
                scale=alt.Scale(zero=True, nice=True),
            ),
        )
        _justerttooltip = [
            alt.Tooltip("Dato:N", title="Dato"),
            alt.Tooltip(
                "Innsatsjustert verdi per 1 000:Q",
                title=_justert_aksetittel,
                format=".2f",
            ),
            alt.Tooltip("Verdi i vindu:Q", title=_verditittel, format=",.0f"),
            alt.Tooltip(
                "Alle artsobservasjoner i vindu:Q",
                title="Alle artsobservasjoner i vinduet",
                format=",.0f",
            ),
        ]
        _justertpanel = alt.layer(
            _justertbase.mark_area(color="#3B82F6", opacity=0.11),
            _justertbase.mark_line(color="#2563EB", strokeWidth=2.6),
            _justertbase.mark_point(opacity=0),
            _justertbase.mark_circle(color="#1D4ED8", size=70).encode(
                opacity=alt.condition(_hover, alt.value(1), alt.value(0)),
                tooltip=_justerttooltip,
            ),
            _justertbase.mark_rule(color="#60A5FA", strokeWidth=1).encode(
                opacity=alt.condition(_hover, alt.value(0.7), alt.value(0))
            ),
        ).properties(
            height=220,
            width="container",
            title=alt.TitleParams(
                text="Innsatsjustert sesongprofil",
                subtitle="Valgt art relativt til all artsrapportering i samme datovindu",
                anchor="start",
                color="#1E3A8A",
                fontSize=15,
                subtitleColor="#64748B",
                subtitleFontSize=12,
            ),
        )

        return (
            alt.vconcat(
                _raapanel,
                _justertpanel,
                spacing=24,
                title=alt.TitleParams(
                    text=f"Sesongprofil – {artstekst}",
                    subtitle=[
                        "Alle år er foldet til januar–desember",
                        f"Sentrert, sirkulært rullerende vindu: {vindusstorrelse} dager",
                    ],
                    anchor="start",
                    color="#0F172A",
                    fontSize=21,
                    fontWeight=600,
                    offset=18,
                    subtitleColor="#475569",
                    subtitleFontSize=12,
                    subtitlePadding=6,
                ),
            )
            .resolve_scale(x="shared", y="independent")
            .add_params(_hover)
            .configure_view(stroke=None)
            .configure_axis(
                domainColor="#CBD5E1",
                gridColor="#E2E8F0",
                gridOpacity=0.75,
                labelColor="#475569",
                labelFontSize=11,
                tickColor="#CBD5E1",
                titleColor="#334155",
                titleFontSize=12,
                titleFontWeight=500,
            )
            .configure_title(font="sans-serif")
        )

    return (lag_sesongprofilfigur,)


@app.cell(hide_code=True)
def _(beregn_sesongprofil_individer, beregn_sesongprofil_observasjoner):
    def test_sesongprofil_mtm_001():
        _grunnlag = pl.DataFrame(
            {
                "Artens ID": [1, 2, 1],
                "Antall": [4, 10, 2],
                "Observert dato": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 12, 31)],
            }
        )
        _resultat = beregn_sesongprofil_observasjoner(_grunnlag, _grunnlag, 1, vindusstorrelse=1)
        _januar = _resultat.filter(pl.col("Dato") == "1. jan.").row(0, named=True)

        assert _resultat.height == 365
        assert _januar["Råverdi per dag"] == 1.0
        assert _januar["Alle artsobservasjoner i vindu"] == 2.0
        assert _januar["Innsatsjustert verdi per 1 000"] == 500.0

    def test_sesongprofil_mtm_002():
        _grunnlag = pl.DataFrame(
            {
                "Artens ID": [1, 2, 1],
                "Antall": [4, 10, 2],
                "Observert dato": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 12, 31)],
            }
        )
        _resultat = beregn_sesongprofil_individer(_grunnlag, _grunnlag, 1, vindusstorrelse=1)
        _januar = _resultat.filter(pl.col("Dato") == "1. jan.").row(0, named=True)

        assert _januar["Råverdi per dag"] == 4.0
        assert _januar["Innsatsjustert verdi per 1 000"] == 2_000.0

    def test_sesongprofil_mtm_003():
        _grunnlag = pl.DataFrame(
            {
                "Artens ID": [1, 2, 1],
                "Antall": [4, 10, 2],
                "Observert dato": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 12, 31)],
            }
        )
        _resultat = beregn_sesongprofil_observasjoner(_grunnlag, _grunnlag, 1, vindusstorrelse=3)
        _januar = _resultat.filter(pl.col("Dato") == "1. jan.").row(0, named=True)

        assert _januar["Verdi i vindu"] == 2.0, "Vinduet skal koble desember og januar"
        assert _januar["Datodager i vindu"] == 3.0
        assert abs(_januar["Innsatsjustert verdi per 1 000"] - (2 / 3 * 1_000)) < 1e-9

    test_sesongprofil_mtm_001()
    test_sesongprofil_mtm_002()
    test_sesongprofil_mtm_003()
    return (
        test_sesongprofil_mtm_001,
        test_sesongprofil_mtm_002,
        test_sesongprofil_mtm_003,
    )


@app.cell(hide_code=True)
def _(lag_sesongprofilfigur, beregn_sesongprofil_observasjoner):
    def test_sesongprofil_mtm_004():
        _grunnlag = pl.DataFrame(
            {
                "Artens ID": [1, 2, 1],
                "Antall": [4, 10, 2],
                "Observert dato": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 12, 31)],
            }
        )
        _profil = beregn_sesongprofil_observasjoner(_grunnlag, _grunnlag, 1, vindusstorrelse=15)
        _figur = lag_sesongprofilfigur(_profil, "Observasjoner", "testart (Avis testus)", 15)
        _figurspesifikasjon = _figur.to_dict()

        assert len(_figurspesifikasjon["vconcat"]) == 2
        assert "Rå sesongprofil" in str(_figurspesifikasjon)
        assert "Innsatsjustert sesongprofil" in str(_figurspesifikasjon)

    test_sesongprofil_mtm_004()
    return (test_sesongprofil_mtm_004,)


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
def velg_art_for_innsatsjustert_maanedsprofil(arter_df, arter_df_lest_inn):
    _artsvalg_rader = (
        arter_df.filter(pl.col("Taksonomisk nivå") == "species")
        .group_by(["Artens ID", "Art"])
        .agg(
            pl.col("Navn").drop_nulls().first().alias("Navn"),
            pl.len().alias("Observasjoner"),
        )
        .sort(["Observasjoner", "Navn"], descending=[True, False])
    )
    mo.stop(
        _artsvalg_rader.is_empty(),
        mo.md("Ingen arter er tilgjengelige i det valgte tabellgrunnlaget for sammenligning av månedsprofiler."),
    )

    _artsvalg = {
        f"{_rad['Navn'] or 'Uten norsk navn'} ({_rad['Art']}; ID {_rad['Artens ID']})": _rad["Artens ID"]
        for _rad in _artsvalg_rader.iter_rows(named=True)
    }
    valgt_art_maanedsprofil = mo.ui.dropdown(
        options=_artsvalg,
        value=next(iter(_artsvalg)),
        searchable=True,
        allow_select_none=False,
        label="Velg art fra artsstatistikken",
        full_width=True,
    )

    _kontrollgrunnlag = arter_df.filter(
        (pl.col("Taksonomisk nivå") == "species") & pl.col("Observert dato").is_not_null()
    )
    _periode_fra_kontroll = _kontrollgrunnlag.get_column("Observert dato").min()
    _periode_til_kontroll = _kontrollgrunnlag.get_column("Observert dato").max()
    _kommuner_kontroll = _kontrollgrunnlag.get_column("Kommune").drop_nulls().unique().to_list()
    _kommuneuttrykk_kontroll = pl.col("Kommune").is_in(_kommuner_kontroll) if _kommuner_kontroll else pl.lit(True)
    _innsats_per_maaned_kontroll = (
        arter_df_lest_inn.filter(
            (pl.col("Taksonomisk nivå") == "species")
            & pl.col("Observert dato").is_between(
                _periode_fra_kontroll,
                _periode_til_kontroll,
                closed="both",
            )
            & _kommuneuttrykk_kontroll
        )
        .group_by(pl.col("Observert dato").dt.month())
        .len()
    )
    _maks_maanedsinnsats = int(_innsats_per_maaned_kontroll.get_column("len").max() or 100)
    _innsatssteg = 10 if _maks_maanedsinnsats <= 500 else 50 if _maks_maanedsinnsats <= 5_000 else 100
    _innsatsstopp = max(
        _innsatssteg,
        ((_maks_maanedsinnsats + _innsatssteg - 1) // _innsatssteg) * _innsatssteg,
    )

    valgt_maaned_forklaring = mo.ui.slider(
        start=1,
        stop=12,
        step=1,
        value=5,
        show_value=True,
        include_input=True,
        label="Måned i regneeksempelet (1 = jan., 12 = des.)",
        full_width=True,
    )
    minste_maanedsinnsats = mo.ui.slider(
        start=0,
        stop=_innsatsstopp,
        step=_innsatssteg,
        value=0,
        show_value=True,
        include_input=True,
        debounce=True,
        label="Minste antall artsregistreringer for å ta med en måned",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md(
                r"""
    ## Utforsk gammel og innsatsjustert månedsprofil

    Velg først en art. Bruk deretter kontrollene til å undersøke **hvordan** og
    **hvorfor** resultatet endrer seg. Ingen av kontrollene endrer originaldataene.
    """
            ),
            valgt_art_maanedsprofil,
            mo.hstack(
                [valgt_maaned_forklaring, minste_maanedsinnsats],
                widths="equal",
                align="start",
                wrap=True,
            ),
            mo.callout(
                mo.md(
                    "**Måned** styrer bare regneeksempelet. "
                    "**Minste datamengde** er en sensitivitetskontroll: måneder "
                    "under grensen skjules fra begge profilkurvene, men beholdes "
                    "i detaljtabellen. Start med `0` for å bruke alle måneder."
                ),
                kind="info",
                title="Slik bruker du kontrollene",
            ),
        ],
        gap=1,
    )
    return (
        minste_maanedsinnsats,
        valgt_art_maanedsprofil,
        valgt_maaned_forklaring,
    )


@app.cell(hide_code=True)
def sammenlign_maanedsprofiler(
    arter_df,
    arter_df_lest_inn,
    minste_maanedsinnsats,
    valgt_art_maanedsprofil,
    valgt_maaned_forklaring,
):
    mo.stop(
        valgt_art_maanedsprofil.value is None,
        mo.md("Velg en art for å lage sammenligningen."),
    )

    _maanedsnavn = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Mai",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Okt",
        "Nov",
        "Des",
    ]
    _tabellgrunnlag = arter_df.filter(
        (pl.col("Taksonomisk nivå") == "species") & pl.col("Observert dato").is_not_null()
    )
    mo.stop(
        _tabellgrunnlag.is_empty(),
        mo.md("Tabellgrunnlaget inneholder ingen artsobservasjoner med gyldig dato."),
    )

    _periode_fra = _tabellgrunnlag.get_column("Observert dato").min()
    _periode_til = _tabellgrunnlag.get_column("Observert dato").max()
    _kommuner = _tabellgrunnlag.get_column("Kommune").drop_nulls().unique().to_list()
    _kommuneuttrykk = pl.col("Kommune").is_in(_kommuner) if _kommuner else pl.lit(True)
    _innsatsgrunnlag = arter_df_lest_inn.filter(
        (pl.col("Taksonomisk nivå") == "species")
        & pl.col("Observert dato").is_between(
            _periode_fra,
            _periode_til,
            closed="both",
        )
        & _kommuneuttrykk
    )
    _art_info = (
        _tabellgrunnlag.filter(pl.col("Artens ID") == valgt_art_maanedsprofil.value)
        .select("Navn", "Art")
        .unique()
        .row(0, named=True)
    )

    _maaneder = pl.DataFrame(
        {
            "Månedsnummer": list(range(1, 13)),
            "Måned": _maanedsnavn,
        }
    )
    _alle_arter_per_maaned = _innsatsgrunnlag.group_by(pl.col("Observert dato").dt.month().alias("Månedsnummer")).agg(
        pl.len().alias("Alle artsobservasjoner")
    )
    _valgt_art_per_maaned = (
        _tabellgrunnlag.filter(pl.col("Artens ID") == valgt_art_maanedsprofil.value)
        .group_by(pl.col("Observert dato").dt.month().alias("Månedsnummer"))
        .agg(pl.len().alias("Artsobservasjoner"))
    )

    _maanedsgrunnlag = (
        _maaneder.join(
            _alle_arter_per_maaned,
            on="Månedsnummer",
            how="left",
        )
        .join(
            _valgt_art_per_maaned,
            on="Månedsnummer",
            how="left",
        )
        .with_columns(
            pl.col("Alle artsobservasjoner").fill_null(0),
            pl.col("Artsobservasjoner").fill_null(0),
        )
        .with_columns(
            (
                (pl.col("Alle artsobservasjoner") >= minste_maanedsinnsats.value)
                & (pl.col("Alle artsobservasjoner") > 0)
            ).alias("Brukes i profil"),
            pl.when(pl.col("Alle artsobservasjoner") > 0)
            .then(pl.col("Artsobservasjoner") / pl.col("Alle artsobservasjoner") * 1_000)
            .otherwise(None)
            .alias("Relativ frekvens per 1 000"),
        )
    )
    _brukt_maanedsgrunnlag = _maanedsgrunnlag.filter(pl.col("Brukes i profil"))
    _sum_artsobservasjoner_brukt = int(_brukt_maanedsgrunnlag.get_column("Artsobservasjoner").sum() or 0)
    _sum_relative_frekvenser_brukt = float(_brukt_maanedsgrunnlag.get_column("Relativ frekvens per 1 000").sum() or 0.0)
    _maanedssammenligning = _maanedsgrunnlag.with_columns(
        pl.when(pl.col("Brukes i profil") & (_sum_artsobservasjoner_brukt > 0))
        .then(pl.col("Artsobservasjoner") / _sum_artsobservasjoner_brukt * 100)
        .otherwise(None)
        .alias("Gammel profil (%)"),
        pl.when(pl.col("Brukes i profil") & (_sum_relative_frekvenser_brukt > 0))
        .then(pl.col("Relativ frekvens per 1 000") / _sum_relative_frekvenser_brukt * 100)
        .otherwise(None)
        .alias("Innsatsjustert profil (%)"),
    )

    _profilrader = _maanedssammenligning.filter(
        pl.col("Gammel profil (%)").is_not_null() & pl.col("Innsatsjustert profil (%)").is_not_null()
    )
    _profil_kan_vises = _profilrader.height > 0
    _antall_maaneder_brukt = _maanedssammenligning.filter(pl.col("Brukes i profil")).height
    _antall_maaneder_under_terskel = 12 - _antall_maaneder_brukt

    if _profil_kan_vises:
        _profil_lang = _profilrader.select(
            "Måned",
            "Gammel profil (%)",
            "Innsatsjustert profil (%)",
        ).unpivot(
            on=["Gammel profil (%)", "Innsatsjustert profil (%)"],
            index="Måned",
            variable_name="Metode",
            value_name="Andel av profil (%)",
        )
        _profilfigur = px.line(
            _profil_lang,
            x="Måned",
            y="Andel av profil (%)",
            color="Metode",
            line_dash="Metode",
            symbol="Metode",
            markers=True,
            category_orders={"Måned": _maanedsnavn},
            color_discrete_map={
                "Gammel profil (%)": "#6B7280",
                "Innsatsjustert profil (%)": "#2563EB",
            },
            line_dash_map={
                "Gammel profil (%)": "dot",
                "Innsatsjustert profil (%)": "solid",
            },
            symbol_map={
                "Gammel profil (%)": "circle",
                "Innsatsjustert profil (%)": "diamond",
            },
            title=(f"Gammel og innsatsjustert månedsprofil – {_art_info['Navn']} ({_art_info['Art']})"),
        )
        _profilfigur.update_layout(
            legend_title_text="",
            hovermode="x unified",
            yaxis_title="Andel av månedsprofil (%)",
            xaxis_title="Måned",
        )
        _profilfigur.add_hline(
            y=100 / _antall_maaneder_brukt,
            line_dash="dash",
            line_color="#9CA3AF",
            annotation_text="Jevnt fordelt profil",
            annotation_position="top right",
        )
        _profilvisualisering = _profilfigur
    else:
        _profilvisualisering = mo.callout(
            mo.md(
                "Ingen profil kan beregnes med denne terskelen. Senk "
                "**minste datamengde** til minst én måned med registreringer av "
                "arten inngår."
            ),
            kind="danger",
            title="Terskelen er for høy",
        )

    _innsatsvisning = _maanedssammenligning.with_columns(
        pl.when(pl.col("Brukes i profil"))
        .then(pl.lit("Brukes i profil"))
        .otherwise(pl.lit("Under terskel"))
        .alias("Datastatus")
    )
    _innsatsfigur = px.bar(
        _innsatsvisning,
        x="Måned",
        y="Alle artsobservasjoner",
        color="Datastatus",
        category_orders={
            "Måned": _maanedsnavn,
            "Datastatus": ["Brukes i profil", "Under terskel"],
        },
        color_discrete_map={
            "Brukes i profil": "#60A5FA",
            "Under terskel": "#F59E0B",
        },
        title="Månedlig rapporteringsmengde som brukes som innsatsproxy",
        labels={"Alle artsobservasjoner": "Alle artsregistreringer"},
    )
    _innsatsfigur.update_layout(
        legend_title_text="",
        xaxis_title="Måned",
        yaxis_title="Alle artsregistreringer",
    )
    if minste_maanedsinnsats.value > 0:
        _innsatsfigur.add_hline(
            y=minste_maanedsinnsats.value,
            line_dash="dash",
            line_color="#B45309",
            annotation_text="Valgt terskel",
            annotation_position="top right",
        )

    if _profil_kan_vises:
        _gammel_topp = _profilrader.sort(
            "Gammel profil (%)",
            descending=True,
        ).row(0, named=True)
        _justert_topp = _profilrader.sort(
            "Innsatsjustert profil (%)",
            descending=True,
        ).row(0, named=True)
        _maks_justering = float(
            _profilrader.select((pl.col("Innsatsjustert profil (%)") - pl.col("Gammel profil (%)")).abs().max()).item()
            or 0.0
        )
        if _gammel_topp["Måned"] != _justert_topp["Måned"]:
            _hovedbudskap = mo.callout(
                mo.md(
                    f"Den største månedsandelen flytter seg fra "
                    f"**{_gammel_topp['Måned']}** i råprofilen til "
                    f"**{_justert_topp['Måned']}** etter justering. Det viser at "
                    "månedlig rapporteringsmengde påvirker formen på råprofilen. "
                    "Det beviser ikke at den justerte toppen er artens sanne "
                    "biologiske topp."
                ),
                kind="info",
                title="Toppen flytter seg etter justering",
            )
        else:
            _hovedbudskap = mo.callout(
                mo.md(
                    f"Både råprofilen og den innsatsjusterte profilen har størst "
                    f"månedsandel i **{_gammel_topp['Måned']}**. Formen kan likevel "
                    f"være endret; største forskjell er "
                    f"**{_maks_justering:.1f} prosentpoeng**."
                ),
                kind="success",
                title="Samme toppmåned, men kontroller resten av kurven",
            )
    else:
        _maks_justering = 0.0
        _hovedbudskap = mo.callout(
            mo.md("Terskelen etterlater ikke nok data til å sammenligne profilene."),
            kind="danger",
            title="Ingen sammenlignbar profil",
        )

    _valgt_maanedsrad = _maanedssammenligning.filter(pl.col("Månedsnummer") == valgt_maaned_forklaring.value).row(
        0, named=True
    )
    _valgt_maanedsnavn = _valgt_maanedsrad["Måned"]
    _valgt_artsobservasjoner = int(_valgt_maanedsrad["Artsobservasjoner"])
    _valgt_maanedsinnsats = int(_valgt_maanedsrad["Alle artsobservasjoner"])
    _valgt_relativ_frekvens = _valgt_maanedsrad["Relativ frekvens per 1 000"]
    _valgt_gammel_profil = _valgt_maanedsrad["Gammel profil (%)"]
    _valgt_justert_profil = _valgt_maanedsrad["Innsatsjustert profil (%)"]
    _valgt_maaned_brukes = bool(_valgt_maanedsrad["Brukes i profil"])

    if _valgt_maaned_brukes:
        _maanedsstatus = mo.callout(
            mo.md(
                (
                    f"{_valgt_maanedsnavn} har {_valgt_maanedsinnsats:,} "
                    "artsregistreringer og er derfor med i begge profilene."
                ).replace(",", " ")
            ),
            kind="success",
            title="Måneden er over terskelen",
        )
        _normalisert_regnestykke = mo.md(
            rf"""
    ### Trinn 3: Sammenlign profilandelene

    Den gamle metoden deler månedens artsregistreringer på alle registreringer av
    arten i månedene som er med:

    \[
    100 \times \frac{{{_valgt_artsobservasjoner}}}
    {{{_sum_artsobservasjoner_brukt}}}
    = {_valgt_gammel_profil:.2f}\,\%
    \]

    Den nye metoden normaliserer de tolv relative frekvensene, slik at de månedene
    som er med til sammen utgjør 100 %:

    \[
    100 \times \frac{{{_valgt_relativ_frekvens:.2f}}}
    {{{_sum_relative_frekvenser_brukt:.2f}}}
    = {_valgt_justert_profil:.2f}\,\%
    \]

    Forskjellen skyldes at den nye metoden først spør hvor stor andel arten utgjør
    av **all rapportering i samme måned**, før månedsprofilen normaliseres.
    """
        )
    else:
        _maanedsstatus = mo.callout(
            mo.md(
                (
                    f"{_valgt_maanedsnavn} har {_valgt_maanedsinnsats:,} "
                    "artsregistreringer, som er under terskelen på "
                    f"{minste_maanedsinnsats.value:,}. Råverdiene vises fortsatt, men "
                    "måneden inngår ikke i de normaliserte profilene."
                ).replace(",", " ")
            ),
            kind="warn",
            title="Måneden er under terskelen",
        )
        _normalisert_regnestykke = mo.md(
            "### Trinn 3: Profilandel\n\n"
            "Profilandelen beregnes ikke for denne måneden med valgt terskel. "
            "Senk terskelen for å se den inngå i begge kurvene."
        )

    _valgt_frekvenstekst = f"{_valgt_relativ_frekvens:.2f}" if _valgt_relativ_frekvens is not None else "Ikke beregnbar"
    _regneeksempel = mo.vstack(
        [
            mo.md(
                rf"""
    ## Regn ut {_valgt_maanedsnavn} steg for steg

    ### Trinn 1: Identifiser teller og nevner

    - **Teller \(S_m\):** {_valgt_artsobservasjoner} registreringer av
      {_art_info["Navn"]} i {_valgt_maanedsnavn}.
    - **Nevner \(E_m\):** {_valgt_maanedsinnsats} registreringer av alle arter i
      samme måned, periode og kommuner.

    Nevneren er en **proxy for rapporteringsinnsats**. En måned med mange innsendte
    artsregistreringer gir større mulighet for at også den valgte arten blir
    registrert.

    ### Trinn 2: Beregn relativ rapporteringsfrekvens

    \[
    R_m = 1000 \times \frac{{S_m}}{{E_m}}
    = 1000 \times \frac{{{_valgt_artsobservasjoner}}}
    {{{_valgt_maanedsinnsats}}}
    = {_valgt_frekvenstekst}
    \]

    Dette leses som **{_valgt_frekvenstekst} registreringer av arten per 1 000
    artsregistreringer** i {_valgt_maanedsnavn}. Det er en relativ
    rapporteringsfrekvens, ikke et estimat på antall individer.
    """
            ),
            _maanedsstatus,
            _normalisert_regnestykke,
        ],
        gap=1,
    )

    _statistikkort = mo.hstack(
        [
            mo.stat(
                value=f"{_tabellgrunnlag.filter(pl.col('Artens ID') == valgt_art_maanedsprofil.value).height:,}".replace(
                    ",", " "
                ),
                label="Artsregistreringer",
                caption="Tellergrunnlaget for gammel profil",
                bordered=True,
            ),
            mo.stat(
                value=f"{_innsatsgrunnlag.height:,}".replace(",", " "),
                label="Alle artsregistreringer",
                caption="Samlet innsatsproxy i perioden",
                bordered=True,
            ),
            mo.stat(
                value=f"{_antall_maaneder_brukt} av 12",
                label="Måneder i kurvene",
                caption=f"{_antall_maaneder_under_terskel} under valgt terskel",
                bordered=True,
            ),
            mo.stat(
                value=f"{_maks_justering:.1f} pp",
                label="Største justering",
                caption="Største absolutt forskjell mellom kurvene",
                bordered=True,
            ),
        ],
        widths="equal",
        wrap=True,
    )

    _visningstabell = _innsatsvisning.select(
        "Måned",
        pl.col("Artsobservasjoner").alias("Artsobservasjoner (gammel)"),
        pl.col("Alle artsobservasjoner").alias("Alle artsobservasjoner (innsats)"),
        pl.col("Relativ frekvens per 1 000").round(2),
        pl.col("Gammel profil (%)").round(2),
        pl.col("Innsatsjustert profil (%)").round(2),
        "Datastatus",
    )
    _detaljtabell = mo.ui.table(
        _visningstabell,
        pagination=False,
        selection=None,
        show_data_types=False,
        show_column_summaries=False,
        show_search=False,
        show_download=True,
        text_justify_columns={
            "Artsobservasjoner (gammel)": "center",
            "Alle artsobservasjoner (innsats)": "center",
            "Relativ frekvens per 1 000": "center",
            "Gammel profil (%)": "center",
            "Innsatsjustert profil (%)": "center",
        },
        header_tooltip={
            "Artsobservasjoner (gammel)": "Antall rader for valgt art i gjeldende tabellgrunnlag.",
            "Alle artsobservasjoner (innsats)": "Alle artsrader i samme måned, periode og kommuner.",
            "Relativ frekvens per 1 000": "Artsobservasjoner delt på alle artsobservasjoner, multiplisert med 1 000.",
            "Gammel profil (%)": "Månedens andel av artens rå registreringer blant måneder over terskelen.",
            "Innsatsjustert profil (%)": "Månedens andel etter justering for rapporteringsmengde.",
            "Datastatus": "Viser om måneden inngår i profilkurvene med valgt terskel.",
        },
    )

    _metodeforklaring = mo.accordion(
        {
            "Hvorfor kan rå månedstall villede?": mo.md(
                r"""
    Hvis flere personer rapporterer, eller rapporteringssystemet leverer flere
    rader i én måned, øker sjansen for å få registreringer av alle arter. En topp i
    råtall kan derfor skyldes både biologisk sesongmønster og høy
    rapporteringsaktivitet. Den gamle profilen kan ikke skille disse mekanismene.
    """
            ),
            "Hva gjør justeringen matematisk?": mo.md(
                r"""
    For måned \(m\) beregnes

    \[
    R_m = 1000 \times \frac{S_m}{E_m},
    \]

    hvor \(S_m\) er registreringer av valgt art og \(E_m\) er alle
    artsregistreringer. \(R_m\) beholder en tolkbar skala: registreringer av arten
    per 1 000 artsregistreringer. For å sammenligne **formen** med den gamle
    nanoprofilen normaliseres deretter begge kurver til 100 %.
    """
            ),
            "Hva betyr dataterskelen?": mo.md(
                r"""
    Terskelen er en **sensitivitetsanalyse**, ikke en p-verdi eller formell
    kvalitetsgrense. Når du øker terskelen, fjernes måneder med lite
    rapporteringsgrunnlag fra begge kurvene. Hvis konklusjonen endrer seg kraftig,
    er månedsmønsteret følsomt for måneder med få data og bør beskrives med ekstra
    forsiktighet.
    """
            ),
            "Hva løser metoden ikke?": mo.md(
                r"""
    Metoden kjenner ikke antall observatørtimer, reiselengde, sjekklengde,
    dekningsareal, vær eller reelle nullobservasjoner. Artsregistreringer er heller
    ikke nødvendigvis uavhengige statistiske forsøk. Resultatet er derfor en
    **innsatsjustert relativ rapporteringsfrekvens**, ikke abundans,
    bestandstetthet, deteksjonssannsynlighet eller en kausal effekt.
    """
            ),
            "Hva kreves for en sterkere publikasjonsanalyse?": mo.md(
                r"""
    En sterkere analyse trenger strukturerte observasjonsøkter eller komplette
    sjekklister med både funn og ikke-funn, samt mål på varighet, område og
    observatørinnsats. Da kan man modellere deteksjon og biologisk forekomst
    separat, for eksempel med okkupasjonsmodeller eller modeller med en eksplisitt
    innsatskomponent. Denne notebook-metoden er et transparent mellomtrinn for
    opportunistiske forekomstdata.
    """
            ),
        },
        multiple=False,
    )

    _sammenligningsfane = mo.vstack(
        [
            mo.md(
                f"""
    ## Hva skjer med mønsteret?

    Den **grå, prikkede kurven** er den gamle månedsprofilen. Den **blå, heltrukne
    kurven** justerer først for hvor mange artsregistreringer som finnes i hver
    måned. Med terskel `0` samsvarer den grå kurven med månedsfordelingen i
    artsstatistikken. Hever du terskelen, sammenlignes begge metoder på de samme
    månedene over terskelen.

    Dataperiode: **{_periode_fra}–{_periode_til}**. Kommuner:
    **{", ".join(sorted(_kommuner)) if _kommuner else "ikke avgrenset"}**.
    """
            ),
            _statistikkort,
            _hovedbudskap,
            _profilvisualisering,
            mo.callout(
                mo.md(
                    "Se først etter **om toppmåneden flytter seg**, og deretter "
                    "etter måneder der avstanden mellom kurvene er stor. Bruk "
                    "fanen *Regn på én måned* for å forklare en konkret forskjell."
                ),
                kind="neutral",
                title="Slik leser du figuren",
            ),
        ],
        gap=1,
    )

    _datagrunnlagsfane = mo.vstack(
        [
            mo.md(
                r"""
    ## Se nevneren før du tolker profilen

    Søylene viser hvor mange artsregistreringer som ligger bak nevneren i hver
    måned. En høy søyle betyr mer rapportering, ikke nødvendigvis flere fugler.
    Oransje måneder er under valgt terskel. Tabellen under gjør alle mellomregningene
    etterprøvbare og kan lastes ned.
    """
            ),
            _innsatsfigur,
            _detaljtabell,
        ],
        gap=1,
    )

    _metodefane = mo.vstack(
        [
            mo.callout(
                mo.md(
                    "Bruk betegnelsen **innsatsjustert relativ "
                    "rapporteringsfrekvens**. Unngå å omtale dette som korrigert "
                    "abundans eller sann forekomst."
                ),
                kind="warn",
                title="Anbefalt språk i en publikasjon",
            ),
            _metodeforklaring,
        ],
        gap=1,
    )

    mo.ui.tabs(
        {
            "1 · Sammenlign profilene": _sammenligningsfane,
            "2 · Regn på én måned": _regneeksempel,
            "3 · Undersøk datagrunnlaget": _datagrunnlagsfane,
            "4 · Metode og begrensninger": _metodefane,
        },
        value="1 · Sammenlign profilene",
        label="Læringssti for innsatsjustert månedsprofil",
    )
    return


@app.cell(hide_code=True)
def velg_sesongprofil(arter_df):
    _artsvalg_rader = (
        arter_df.filter(
            (pl.col("Taksonomisk nivå") == "species") & pl.col("Observert dato").is_not_null()
        )
        .group_by(["Artens ID", "Art"])
        .agg(
            pl.col("Navn").drop_nulls().first().alias("Navn"),
            pl.len().alias("Observasjoner"),
        )
        .sort(["Observasjoner", "Navn"], descending=[True, False])
    )
    mo.stop(
        _artsvalg_rader.is_empty(),
        mo.md("Ingen arter med gyldig dato er tilgjengelige for sesongprofilen."),
    )

    _artsvalg = {
        f"{_rad['Navn'] or 'Uten norsk navn'} ({_rad['Art']}; ID {_rad['Artens ID']})": _rad["Artens ID"]
        for _rad in _artsvalg_rader.iter_rows(named=True)
    }
    valgt_art_sesongprofil = mo.ui.dropdown(
        options=_artsvalg,
        value=next(iter(_artsvalg)),
        searchable=True,
        allow_select_none=False,
        label="Velg art",
        full_width=True,
    )
    valgt_metrikk_sesongprofil = mo.ui.radio(
        options=["Observasjoner", "Individer"],
        value="Observasjoner",
        inline=True,
        label="Vis sesongprofil for",
    )
    sesongprofil_vindu = mo.ui.slider(
        start=1,
        stop=31,
        step=1,
        value=15,
        show_value=True,
        include_input=True,
        debounce=True,
        label="Rullerende vindu (dager)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md(
                r"""
    ## Sesongprofil gjennom året

    Alle år foldes til ett kalenderår. Det øverste panelet viser den rå
    observasjonsmengden, mens det nederste justerer for all artsrapportering i
    samme datovindu. GT-tabellen og månedsprofilen endres ikke.
    """
            ),
            valgt_art_sesongprofil,
            mo.hstack(
                [valgt_metrikk_sesongprofil, sesongprofil_vindu],
                widths="equal",
                align="start",
                wrap=True,
            ),
        ],
        gap=1,
    )
    return valgt_art_sesongprofil, valgt_metrikk_sesongprofil, sesongprofil_vindu


@app.cell(hide_code=True)
def vis_sesongprofil(
    arter_df,
    arter_df_lest_inn,
    beregn_sesongprofil_individer,
    beregn_sesongprofil_observasjoner,
    lag_sesongprofilfigur,
    sesongprofil_vindu,
    valgt_art_sesongprofil,
    valgt_metrikk_sesongprofil,
):
    mo.stop(
        valgt_art_sesongprofil.value is None,
        mo.md("Velg en art for å lage sesongprofilen."),
    )
    _tabellgrunnlag = arter_df.filter(
        (pl.col("Taksonomisk nivå") == "species") & pl.col("Observert dato").is_not_null()
    )
    mo.stop(
        _tabellgrunnlag.is_empty(),
        mo.md("Tabellgrunnlaget inneholder ingen artsobservasjoner med gyldig dato."),
    )

    _periode_fra = _tabellgrunnlag.get_column("Observert dato").min()
    _periode_til = _tabellgrunnlag.get_column("Observert dato").max()
    _kommuner = _tabellgrunnlag.get_column("Kommune").drop_nulls().unique().to_list()
    _kommuneuttrykk = pl.col("Kommune").is_in(_kommuner) if _kommuner else pl.lit(True)
    _innsatsgrunnlag = arter_df_lest_inn.filter(
        (pl.col("Taksonomisk nivå") == "species")
        & pl.col("Observert dato").is_not_null()
        & pl.col("Observert dato").is_between(_periode_fra, _periode_til, closed="both")
        & _kommuneuttrykk
    )
    mo.stop(
        _innsatsgrunnlag.is_empty(),
        mo.callout(
            mo.md("Det finnes ingen artsregistreringer som kan brukes som innsatsgrunnlag."),
            kind="danger",
            title="Mangler innsatsgrunnlag",
        ),
    )

    _art_info = (
        _tabellgrunnlag.filter(pl.col("Artens ID") == valgt_art_sesongprofil.value)
        .select("Navn", "Art")
        .unique()
        .row(0, named=True)
    )
    if valgt_metrikk_sesongprofil.value == "Observasjoner":
        _profil = beregn_sesongprofil_observasjoner(
            _tabellgrunnlag,
            _innsatsgrunnlag,
            valgt_art_sesongprofil.value,
            sesongprofil_vindu.value,
        )
        _enhetsforklaring = "observasjoner av arten per 1 000 artsobservasjoner"
    else:
        _profil = beregn_sesongprofil_individer(
            _tabellgrunnlag,
            _innsatsgrunnlag,
            valgt_art_sesongprofil.value,
            sesongprofil_vindu.value,
        )
        _enhetsforklaring = "individer av arten per 1 000 artsobservasjoner"

    _artstekst = f"{_art_info['Navn'] or _art_info['Art']} ({_art_info['Art']})"
    _figur = lag_sesongprofilfigur(
        _profil,
        valgt_metrikk_sesongprofil.value,
        _artstekst,
        sesongprofil_vindu.value,
    )
    _antall_aar = _tabellgrunnlag.get_column("Observert dato").dt.year().n_unique()

    mo.vstack(
        [
            mo.ui.altair_chart(_figur),
            mo.callout(
                mo.md(
                    f"Det øverste panelet viser et rullerende gjennomsnitt per "
                    f"kalenderdag på tvers av **{_antall_aar} år**. Det nederste "
                    f"viser **{_enhetsforklaring}**. En topp i nederste panel betyr "
                    "at arten utgjør en større del av rapporteringen i perioden; "
                    "den er ikke et direkte mål på bestand eller tetthet."
                ),
                kind="neutral",
                title="Slik leses panelene",
            ),
        ],
        gap=1,
    )
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
