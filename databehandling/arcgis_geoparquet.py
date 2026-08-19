#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas>=1.0",
#   "pandas>=2.2",
#   "pyarrow>=15",
# ]
# ///
"""Konverter behandlet Artskart-Parquet til ArcGIS-kompatibel GeoParquet."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

STANDARD_CRS = "EPSG:25833"
GEOMETRI_GRUPPER = {
    "Point": "punkter",
    "MultiPoint": "multipunkter",
    "LineString": "linjer",
    "MultiLineString": "linjer",
    "Polygon": "polygoner",
    "MultiPolygon": "polygoner",
}


def les_wkt_parquet(input_sti: Path, crs: str) -> gpd.GeoDataFrame:
    """Read a regular Parquet file and convert its WKT column to geometry."""
    if not input_sti.is_file():
        raise FileNotFoundError(f"Fant ikke inputfilen: {input_sti}")

    df = pd.read_parquet(input_sti)
    if "geometry" not in df.columns:
        raise ValueError("Inputfilen mangler kolonnen 'geometry'.")
    if df["geometry"].isna().any():
        antall = int(df["geometry"].isna().sum())
        raise ValueError(f"Geometry-kolonnen inneholder {antall} nullverdier.")

    try:
        geometry = gpd.GeoSeries.from_wkt(
            df.pop("geometry"), crs=crs, on_invalid="raise"
        )
    except Exception as exc:
        raise ValueError(f"Kunne ikke lese geometry-kolonnen som WKT: {exc}") from exc

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    tomme_geometrier = int(gdf.geometry.is_empty.sum())
    if tomme_geometrier:
        raise ValueError(
            f"Geometry-kolonnen inneholder {tomme_geometrier} tomme geometrier."
        )

    ugyldige_geometrier = int((~gdf.geometry.is_valid).sum())
    if ugyldige_geometrier:
        raise ValueError(
            f"Geometry-kolonnen inneholder {ugyldige_geometrier} ugyldige geometrier."
        )

    return gdf


def del_i_arcgis_lag(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split geometries into homogeneous ArcGIS-compatible layers."""
    geometrityper = gdf.geometry.geom_type
    ukjente = sorted(set(geometrityper.dropna()) - set(GEOMETRI_GRUPPER))
    if ukjente:
        raise ValueError(f"Geometrityper uten eksportregel: {', '.join(ukjente)}")

    lag: dict[str, gpd.GeoDataFrame] = {}
    for geometritype, suffix in GEOMETRI_GRUPPER.items():
        utvalg = gdf.loc[geometrityper == geometritype]
        if utvalg.empty:
            continue
        if suffix in lag:
            lag[suffix] = pd.concat([lag[suffix], utvalg], ignore_index=True)
            lag[suffix] = gpd.GeoDataFrame(
                lag[suffix], geometry="geometry", crs=gdf.crs
            )
        else:
            lag[suffix] = utvalg.copy()

    return lag


def skriv_geoparquet_atomisk(gdf: gpd.GeoDataFrame, output_sti: Path) -> None:
    """Write WKB GeoParquet 1.0 and replace the destination atomically."""
    output_sti.parent.mkdir(parents=True, exist_ok=True)
    filreferanse, midlertidig_navn = tempfile.mkstemp(
        prefix=f".{output_sti.stem}.",
        suffix=".parquet",
        dir=output_sti.parent,
    )
    os.close(filreferanse)
    midlertidig_sti = Path(midlertidig_navn)

    try:
        gdf.to_parquet(
            midlertidig_sti,
            index=False,
            geometry_encoding="WKB",
            schema_version="1.0.0",
        )
        midlertidig_sti.replace(output_sti)
    finally:
        midlertidig_sti.unlink(missing_ok=True)


def konverter(
    input_sti: Path,
    output_mappe: Path | None = None,
    crs: str = STANDARD_CRS,
    overskriv: bool = False,
) -> list[tuple[Path, int, int]]:
    """Convert WKT Parquet into one GeoParquet file per ArcGIS geometry type."""
    input_sti = input_sti.expanduser().resolve()
    output_mappe = (output_mappe or input_sti.parent).expanduser().resolve()

    gdf = les_wkt_parquet(input_sti, crs)
    lag = del_i_arcgis_lag(gdf)
    if not lag:
        raise ValueError("Inputfilen inneholder ingen geometrier som kan eksporteres.")

    output_filer = {
        suffix: output_mappe / f"{input_sti.stem}_{suffix}.parquet" for suffix in lag
    }
    eksisterende = [sti for sti in output_filer.values() if sti.exists()]
    if eksisterende and not overskriv:
        opplisting = ", ".join(str(sti) for sti in eksisterende)
        raise FileExistsError(
            f"Output finnes allerede: {opplisting}. Bruk --overskriv for å erstatte."
        )

    resultat: list[tuple[Path, int, int]] = []
    for suffix, lag_gdf in lag.items():
        output_sti = output_filer[suffix]
        skriv_geoparquet_atomisk(lag_gdf, output_sti)
        resultat.append((output_sti, len(lag_gdf), int(lag_gdf.geometry.nunique())))

    return resultat


def lag_argumentparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Konverter en behandlet Parquet-fil med WKT-geometri til separate "
            "ArcGIS-kompatible GeoParquet-filer."
        )
    )
    parser.add_argument(
        "input", type=Path, help="Behandlet Parquet-fil med kolonnen 'geometry'."
    )
    parser.add_argument(
        "--output-mappe",
        type=Path,
        help="Mappe for resultatene. Standard er samme mappe som inputfilen.",
    )
    parser.add_argument(
        "--crs",
        default=STANDARD_CRS,
        help=f"Koordinatsystem for WKT-koordinatene. Standard: {STANDARD_CRS}.",
    )
    parser.add_argument(
        "--overskriv",
        action="store_true",
        help="Erstatt GeoParquet-filer som allerede finnes.",
    )
    return parser


def main() -> int:
    args = lag_argumentparser().parse_args()
    try:
        resultater = konverter(
            input_sti=args.input,
            output_mappe=args.output_mappe,
            crs=args.crs,
            overskriv=args.overskriv,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"Feil: {exc}", file=sys.stderr)
        return 1

    print(f"Ferdig. Koordinatsystem: {args.crs}")
    for sti, antall_rader, unike_geometrier in resultater:
        print(f"  {sti} ({antall_rader} rader, {unike_geometrier} unike geometrier)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
