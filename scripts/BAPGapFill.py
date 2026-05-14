from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


def gapfill_target_bap(
    target_path,
    fill_paths,
    output_path,
    nodata=None,
):
    """
    Fill nodata pixels in target BAP using one or more backup BAP composites.

    Priority:
        1. target_path
        2. fill_paths[0]
        3. fill_paths[1]
        ...
    """

    with rasterio.open(target_path) as src:
        target = src.read()
        profile = src.profile.copy()
        if nodata is None:
            nodata = src.nodata

    if nodata is None:
        nodata = -9999

    final = target.copy()
    missing = np.all(final == nodata, axis=0)

    for fill_path in fill_paths:
        with rasterio.open(fill_path) as src:
            fill = src.read()

        fill_valid = ~np.all(fill == nodata, axis=0)
        update = missing & fill_valid

        final[:, update] = fill[:, update]
        missing = np.all(final == nodata, axis=0)

        if not np.any(missing):
            break

    profile.update(
        count=final.shape[0],
        dtype=final.dtype,
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(final)


if __name__ == "__main__":
    target_year = 2024
    fill_years = [2025, 2023]
    sensor = "S30"

    target_dir = Path(rf"F:\yukon\processed\{sensor}\{target_year}\bap_outputs")
    output_dir = Path(rf"F:\yukon\processed\{sensor}\gap_filled\{target_year}")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_files = list(target_dir.glob("*_bap_composite.tif"))

    for target_path in tqdm(
        target_files,
        desc=f"Gap Filling {target_year} tiles",
    ):
        tile = target_path.name.replace("_bap_composite.tif", "")

        fill_paths = [
            Path(
                rf"F:\yukon\processed\{sensor}\{yr}\bap_outputs\{tile}_bap_composite.tif"
            )
            for yr in fill_years
        ]

        fill_paths = [p for p in fill_paths if p.exists()]

        if not fill_paths:
            continue

        out_path = output_dir / f"{tile}_bap_gapfilled.tif"
        if out_path.exists():
            continue
        try:
            gapfill_target_bap(
                target_path=target_path,
                fill_paths=fill_paths,
                output_path=out_path,
            )
        except Exception as e:
            tqdm.write(f"Error processing {tile}: {e}")
