import json
import runpy
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).parents[1] / "arcgis_geoparquet.py"
SCRIPT_GLOBALS = runpy.run_path(SCRIPT)
CRS = SCRIPT_GLOBALS["CRS"]
main = SCRIPT_GLOBALS["main"]


def skriv_input(sti: Path, geometrier: list[str | None]) -> None:
    pd.DataFrame(
        {
            "Artens ID": range(1, len(geometrier) + 1),
            "Navn": [f"art-{indeks}" for indeks in range(len(geometrier))],
            "geometry": geometrier,
        }
    ).to_parquet(sti, index=False)


def test_deler_punkter_og_polygoner_til_geoparquet(tmp_path, capsys):
    input_sti = tmp_path / "observasjoner.parquet"
    skriv_input(
        input_sti,
        [
            "POINT (254222 6784502)",
            "POINT (254223 6784503)",
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
        ],
    )

    assert main([str(input_sti)]) == 0

    punkt_sti = tmp_path / "observasjoner_punkter.parquet"
    polygon_sti = tmp_path / "observasjoner_polygoner.parquet"
    punkter = gpd.read_parquet(punkt_sti)
    polygoner = gpd.read_parquet(polygon_sti)

    assert punkter["Artens ID"].tolist() == [1, 2]
    assert polygoner["Artens ID"].tolist() == [3]
    assert set(punkter.geometry.geom_type) == {"Point"}
    assert set(polygoner.geometry.geom_type) == {"Polygon"}
    assert punkter.crs.to_epsg() == 25833
    assert polygoner.crs.to_epsg() == 25833
    assert (
        pd.read_parquet(input_sti)["geometry"]
        .str.startswith(("POINT", "POLYGON"))
        .all()
    )

    utskrift = capsys.readouterr().out
    assert f"Koordinatsystem: {CRS}" in utskrift
    assert "2 rader, 2 unike geometrier" in utskrift
    assert "1 rader, 1 unike geometrier" in utskrift


def test_skriver_wkb_og_geoparquet_1_metadata(tmp_path):
    input_sti = tmp_path / "data.parquet"
    skriv_input(input_sti, ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])

    assert main([str(input_sti)]) == 0

    output_sti = tmp_path / "data_polygoner.parquet"
    metadata = pq.read_metadata(output_sti).metadata
    geo = json.loads(metadata[b"geo"])

    assert pa.types.is_binary(pq.read_schema(output_sti).field("geometry").type)
    assert geo["version"] == "1.0.0"
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"
    assert geo["columns"]["geometry"]["geometry_types"] == ["Polygon"]


def test_lager_bare_lag_for_geometrityper_som_finnes(tmp_path):
    input_sti = tmp_path / "bare_polygon.parquet"
    skriv_input(input_sti, ["POLYGON ((0 0, 1 0, 1 1, 0 0))"])

    assert main([str(input_sti)]) == 0

    assert (tmp_path / "bare_polygon_polygoner.parquet").exists()
    assert not (tmp_path / "bare_polygon_punkter.parquet").exists()


def test_overskriver_tidligere_avledet_fil(tmp_path):
    input_sti = tmp_path / "data.parquet"
    output_sti = tmp_path / "data_punkter.parquet"
    skriv_input(input_sti, ["POINT (1 2)"])
    output_sti.write_bytes(b"gammelt innhold")

    assert main([str(input_sti)]) == 0

    assert gpd.read_parquet(output_sti).geometry.to_wkt().tolist() == ["POINT (1 2)"]


def test_avviser_fil_uten_geometrykolonne(tmp_path, capsys):
    input_sti = tmp_path / "uten_geometry.parquet"
    pd.DataFrame({"id": [1]}).to_parquet(input_sti, index=False)

    assert main([str(input_sti)]) == 1
    assert "mangler kolonnen 'geometry'" in capsys.readouterr().err


def test_avviser_tom_fil(tmp_path, capsys):
    input_sti = tmp_path / "tom.parquet"
    pd.DataFrame({"geometry": pd.Series(dtype="str")}).to_parquet(
        input_sti, index=False
    )

    assert main([str(input_sti)]) == 1
    assert "inneholder ingen geometrier" in capsys.readouterr().err


def test_avviser_null_og_topologisk_ugyldig_geometri(tmp_path, capsys):
    null_sti = tmp_path / "null.parquet"
    ugyldig_sti = tmp_path / "ugyldig.parquet"
    skriv_input(null_sti, [None])
    skriv_input(ugyldig_sti, ["POLYGON ((0 0, 2 2, 0 2, 2 0, 0 0))"])

    assert main([str(null_sti)]) == 1
    assert "tomme eller ugyldige geometrier" in capsys.readouterr().err
    assert main([str(ugyldig_sti)]) == 1
    assert "tomme eller ugyldige geometrier" in capsys.readouterr().err


def test_avviser_ugyldig_wkt(tmp_path, capsys):
    input_sti = tmp_path / "ugyldig_wkt.parquet"
    skriv_input(input_sti, ["POLYGON dette er ikke WKT"])

    assert main([str(input_sti)]) == 1
    assert "Kunne ikke lese geometry-kolonnen som WKT" in capsys.readouterr().err


def test_avviser_geometritype_uten_eksportregel(tmp_path, capsys):
    input_sti = tmp_path / "linje.parquet"
    skriv_input(input_sti, ["LINESTRING (0 0, 1 1)"])

    assert main([str(input_sti)]) == 1
    assert "Geometrityper uten eksportregel: LineString" in capsys.readouterr().err
    assert not list(tmp_path.glob("linje_*.parquet"))


def test_rapporterer_manglende_inputfil(tmp_path, capsys):
    assert main([str(tmp_path / "finnes_ikke.parquet")]) == 1
    assert "Feil:" in capsys.readouterr().err


def test_kan_kjores_som_selvstendig_script(tmp_path):
    input_sti = tmp_path / "cli.parquet"
    skriv_input(input_sti, ["POINT (1 2)"])

    resultat = subprocess.run(
        [sys.executable, str(SCRIPT), str(input_sti)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert resultat.returncode == 0, resultat.stderr
    assert "Ferdig" in resultat.stdout
    assert (tmp_path / "cli_punkter.parquet").exists()
