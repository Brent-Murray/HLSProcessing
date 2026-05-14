import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
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


def bounds_overlap(bounds1, bounds2):
    left1, bottom1, right1, top1 = bounds1
    left2, bottom2, right2, top2 = bounds2

    return not (
        right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1
    )


def merge_gapfilled_tiles_windowed(
    input_dir,
    output_path,
    pattern="*_bap_gapfilled.tif",
    block_size=1024,
    resampling=Resampling.nearest,
):
    input_dir = Path(input_dir)
    raster_paths = sorted(input_dir.glob(pattern))

    if len(raster_paths) == 0:
        raise RuntimeError(f"No rasters found in: {input_dir}")

    srcs = [rasterio.open(p) for p in tqdm(raster_paths, desc="Opening rasters")]

    try:
        ref = srcs[0]
        target_crs = ref.crs
        res_x, res_y = ref.res
        res_y = abs(res_y)

        bands = ref.count
        dtype = ref.dtypes[0]
        nodata = ref.nodata if ref.nodata is not None else -9999

        # Convert all bounds into target CRS before computing mosaic extent
        target_bounds = [
            transform_bounds(src.crs, target_crs, *src.bounds, densify_pts=21)
            for src in srcs
        ]

        left = min(b[0] for b in target_bounds)
        bottom = min(b[1] for b in target_bounds)
        right = max(b[2] for b in target_bounds)
        top = max(b[3] for b in target_bounds)

        width = math.ceil((right - left) / res_x)
        height = math.ceil((top - bottom) / res_y)

        transform = from_origin(left, top, res_x, res_y)

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            transform=transform,
            crs=target_crs,
            count=bands,
            dtype=dtype,
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            BIGTIFF="YES",
        )

        n_rows = math.ceil(height / block_size)
        n_cols = math.ceil(width / block_size)

        with rasterio.open(output_path, "w", **profile) as dst:
            for row in tqdm(range(n_rows), desc="Merging window rows"):
                for col in range(n_cols):
                    row_off = row * block_size
                    col_off = col * block_size

                    win_h = min(block_size, height - row_off)
                    win_w = min(block_size, width - col_off)

                    dst_window = Window(col_off, row_off, win_w, win_h)
                    dst_transform = dst.window_transform(dst_window)
                    dst_bounds = rasterio.windows.bounds(dst_window, transform)

                    out_block = np.full(
                        (bands, win_h, win_w),
                        nodata,
                        dtype=dtype,
                    )

                    for src, src_bounds in zip(srcs, target_bounds):
                        if not bounds_overlap(dst_bounds, src_bounds):
                            continue

                        tmp_block = np.full(
                            (bands, win_h, win_w),
                            nodata,
                            dtype=dtype,
                        )

                        for b in range(1, bands + 1):
                            reproject(
                                source=rasterio.band(src, b),
                                destination=tmp_block[b - 1],
                                src_transform=src.transform,
                                src_crs=src.crs,
                                src_nodata=src.nodata,
                                dst_transform=dst_transform,
                                dst_crs=target_crs,
                                dst_nodata=nodata,
                                resampling=resampling,
                            )

                        src_valid = ~((tmp_block == nodata).all(axis=0))
                        out_missing = (out_block == nodata).all(axis=0)

                        update = out_missing & src_valid
                        out_block[:, update] = tmp_block[:, update]

                    dst.write(out_block, window=dst_window)

        tqdm.write(f"Finished mosaic: {output_path}")

    finally:
        for src in srcs:
            src.close()


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

    merge_gapfilled_tiles_windowed(
        input_dir=output_dir, output_path=(output_dir / f"0_mosaic.tif")
    )
