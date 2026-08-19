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

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

CRS = "EPSG:25833"
GEOMETRILAG = {"Point": "punkter", "Polygon": "polygoner"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lag separate ArcGIS-kompatible GeoParquet-filer for punkter og polygoner."
    )
    parser.add_argument(
        "input", type=Path, help="Parquet-fil med WKT-kolonnen 'geometry'."
    )
    args = parser.parse_args(argv)
    input_sti = args.input.expanduser().resolve()

    try:
        df = pd.read_parquet(input_sti)
        if "geometry" not in df:
            raise ValueError("Inputfilen mangler kolonnen 'geometry'.")
        try:
            geometry = gpd.GeoSeries.from_wkt(
                df.pop("geometry"), crs=CRS, on_invalid="raise"
            )
        except Exception as exc:
            raise ValueError(
                f"Kunne ikke lese geometry-kolonnen som WKT: {exc}"
            ) from exc

        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS)
        if gdf.empty:
            raise ValueError("Inputfilen inneholder ingen geometrier.")
        if gdf.geometry.isna().any() or (~gdf.geometry.is_valid).any():
            raise ValueError(
                "Geometry-kolonnen inneholder tomme eller ugyldige geometrier."
            )

        ukjente = sorted(set(gdf.geometry.geom_type) - set(GEOMETRILAG))
        if ukjente:
            raise ValueError(f"Geometrityper uten eksportregel: {', '.join(ukjente)}")

        resultater = []
        for geometritype, suffix in GEOMETRILAG.items():
            lag = gdf[gdf.geometry.geom_type == geometritype]
            if lag.empty:
                continue
            output_sti = input_sti.with_name(f"{input_sti.stem}_{suffix}.parquet")
            lag.to_parquet(
                output_sti,
                index=False,
                geometry_encoding="WKB",
                schema_version="1.0.0",
            )
            resultater.append((output_sti, len(lag), lag.geometry.nunique()))
    except (OSError, ValueError) as exc:
        print(f"Feil: {exc}", file=sys.stderr)
        return 1

    print(f"Ferdig. Koordinatsystem: {CRS}")
    for sti, antall, unike in resultater:
        print(f"  {sti} ({antall} rader, {unike} unike geometrier)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
