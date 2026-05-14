# NOTE: This implementation is an adapted version of the Best Available Pixel (BAP)
# compositing approach described in White et al. (2014):
# https://doi.org/10.1080/07038992.2014.945827

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from math import ceil


def parse_hls_name(path):
    """
    Expected naming pattern:
        tile_yyyyddd_file.ext

    Examples:
        12ABC_2020180_composite.tif
        12ABC_2020180_fmask.tif
        12ABC_2020180_metadata.json
    """
    stem = Path(path).stem
    match = re.match(r"(.+?)_(\d{7})_(stack|fmask|metadata)$", stem)

    if match is None:
        raise ValueError(f"Could not parse filename: {path}")

    tile, yyyyddd, file_type = match.groups()

    year = int(yyyyddd[:4])
    doy = int(yyyyddd[4:])

    return tile, yyyyddd, year, doy, file_type


def read_cloud_cover(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return float(metadata["properties"]["eo:cloud_cover"])


def index_files(composite_dir, fmask_dir, metadata_dir):
    records = defaultdict(dict)
    skipped = []

    for path in Path(composite_dir).glob("*.tif"):
        try:
            tile, yyyyddd, year, doy, _ = parse_hls_name(path)
            records[(tile, yyyyddd)]["image_path"] = path
            records[(tile, yyyyddd)]["tile"] = tile
            records[(tile, yyyyddd)]["yyyyddd"] = yyyyddd
            records[(tile, yyyyddd)]["year"] = year
            records[(tile, yyyyddd)]["doy"] = doy
        except Exception as e:
            skipped.append((path, f"bad composite removed: {e}"))
            if Path(path).exists():
                Path(path).unlink()

    for path in Path(fmask_dir).glob("*.tif"):
        try:
            tile, yyyyddd, _, _, _ = parse_hls_name(path)
            records[(tile, yyyyddd)]["fmask_path"] = path
        except Exception as e:
            skipped.append((path, f"bad fmask removed: {e}"))
            if Path(path).exists():
                Path(path).unlink()

    for path in Path(metadata_dir).glob("*.json"):
        try:
            tile, yyyyddd, _, _, _ = parse_hls_name(path)
            key = (tile, yyyyddd)
            scene_cloud_pct = read_cloud_cover(path)
        except Exception as e:
            skipped.append((path, f"bad metadata removed: {e}"))
            if Path(path).exists():
                Path(path).unlink()
            continue

        records[key]["metadata_path"] = path
        records[key]["scene_cloud_pct"] = scene_cloud_pct

    complete = []

    for key, rec in records.items():
        required = {"image_path", "fmask_path", "metadata_path", "scene_cloud_pct"}
        missing = required - set(rec)

        if missing:
            for bad_path in [
                rec.get("image_path"),
                rec.get("fmask_path"),
                rec.get("metadata_path"),
            ]:
                if bad_path is not None and Path(bad_path).exists():
                    Path(bad_path).unlink()

            skipped.append((key, f"incomplete scene removed, missing: {missing}"))
        else:
            complete.append(rec)

    complete = sorted(complete, key=lambda x: (x["tile"], x["yyyyddd"]))

    return complete, skipped


def group_records_by_tile(records):
    grouped = defaultdict(list)

    for rec in records:
        grouped[rec["tile"]].append(rec)

    return grouped


def read_stack(paths):
    arrays = []
    profile = None

    for p in paths:
        with rasterio.open(p) as src:
            arrays.append(src.read())
            if profile is None:
                profile = src.profile

    return np.stack(arrays), profile  # time, bands, rows, cols


def gaussian_score(x, target, sigma):
    return np.exp(-0.5 * ((x - target) / sigma) ** 2)


def create_hls_clear_mask(qa, include_water=True, mask_medium_high_aerosol=True):
    """
    Decode HLS Fmask QA layer into a binary clear-pixel mask.

    The QA layer is an 8-bit integer where each bit represents a condition.
    We use bitwise operations to extract those conditions.

    Bit layout (0-indexed, least significant bit first):
        bit 1: cloud
        bit 2: adjacent to cloud/shadow (dilated region)
        bit 3: cloud shadow
        bit 4: snow/ice
        bit 5: water
        bits 6–7: aerosol level (2-bit value)
            00 = climatology
            01 = medium aerosol
            10 = low aerosol
            11 = high aerosol
    """

    # --- Extract individual bits ---
    # (1 << n) creates a mask where only bit n is set
    # qa & mask keeps only that bit
    # > 0 converts result to boolean

    cloud = (qa & (1 << 1)) > 0  # True where cloud bit is set
    adjacent = (qa & (1 << 2)) > 0  # True where adjacent-to-cloud bit is set
    shadow = (qa & (1 << 3)) > 0  # True where shadow bit is set
    snow = (qa & (1 << 4)) > 0  # True where snow/ice bit is set
    water = (qa & (1 << 5)) > 0  # True where water bit is set

    # --- Combine all "bad" conditions ---
    # Any pixel flagged as cloud, shadow, adjacent, or snow is considered bad

    bad = cloud | adjacent | shadow | snow

    # Optionally mask water pixels as well
    if not include_water:
        bad |= water  # "|=" means "add to existing bad mask"

    # --- Extract aerosol bits (bits 6–7) ---
    if mask_medium_high_aerosol:
        # Shift bits 6–7 to the lowest positions:
        # e.g., original bits XX...... → ......XX
        aerosol = (qa >> 6) & 0b11

        # Now aerosol is a 2-bit integer per pixel:
        # 0 = 00 (climatology)
        # 1 = 01 (medium)
        # 2 = 10 (low)
        # 3 = 11 (high)

        medium_aerosol = aerosol == 0b01
        high_aerosol = aerosol == 0b11

        # Add medium and high aerosol pixels to "bad"
        bad |= medium_aerosol | high_aerosol

    # --- Final mask ---
    # "clear" pixels are those NOT marked as bad

    return ~bad


def cloud_distance_score(clear_mask, max_distance=5):
    """
    Compute normalized distance from nearest bad pixel.

    max_distance is in pixels.
    For HLS 30 m data:
        5 pixels = 150 m
        10 pixels = 300 m
    """

    dist = distance_transform_edt(clear_mask)
    # return np.clip(dist / max_distance, 0, 1).astype(np.float32)
    return np.clip((dist / max_distance) ** 2, 0, 1).astype(np.float32)


def load_valid_hls_scenes(
    image_paths,
    fmask_paths,
    doy,
    scene_cloud_pct=None,
):
    valid_images = []
    valid_fmasks = []
    valid_doy = []
    valid_cloud = [] if scene_cloud_pct is not None else None
    skipped = []

    for i, (img_path, fmask_path) in enumerate(zip(image_paths, fmask_paths)):
        try:
            with rasterio.open(img_path) as src:
                img = src.read()
                profile = src.profile

            with rasterio.open(fmask_path) as src:
                qa = src.read()

            # Basic shape check
            if img.shape[1:] != qa.shape[1:]:
                raise ValueError(f"Shape mismatch: image {img.shape}, fmask {qa.shape}")

            valid_images.append(img)
            valid_fmasks.append(qa)
            valid_doy.append(doy[i])

            if scene_cloud_pct is not None:
                valid_cloud.append(scene_cloud_pct[i])

        except Exception as e:
            skipped.append(
                {
                    "image": img_path,
                    "fmask": fmask_path,
                    "error": str(e),
                }
            )

    if len(valid_images) == 0:
        raise RuntimeError("No valid image/Fmask pairs were available.")

    stack = np.stack(valid_images)
    qa_stack = np.stack(valid_fmasks)

    return stack, qa_stack, profile, np.asarray(valid_doy), valid_cloud, skipped


def best_available_pixel_hls(
    image_paths,
    fmask_paths,
    doy,
    scene_cloud_pct=None,
    target_doy=200,
    doy_sigma=30,
    max_cloud_distance=10,
    include_water=True,
    mask_medium_high_aerosol=True,
):
    """
    Create a best-available-pixel composite from HLS rasters.

    Parameters
    ----------
    image_paths : list
        Multiband raster paths, one per date.
    fmask_paths : list
        HLS Fmask raster paths, one per date.
    doy : list or array
        Day-of-year (1–366) for each raster.
    scene_cloud_pct : list or array, optional
        Scene-level cloud percentage (0–100) for each raster.
        Used to weight cleaner scenes more strongly.
    target_doy : int
        Preferred day of year for compositing (e.g., peak growing season).
    doy_sigma : float
        Standard deviation for Gaussian DOY weighting.
        Smaller values enforce stronger temporal consistency.
    max_cloud_distance : int
        Distance (in pixels) at which cloud-distance score saturates to 1.
        Larger values increase spatial smoothing (e.g., 10 px ≈ 300 m for HLS).
    include_water : bool
        If False, water pixels are masked out.
    mask_medium_high_aerosol : bool
        If True, pixels with medium (01) and high (11) aerosol levels
        from QA bits 6–7 are masked out.

    Returns
    -------
    composite : np.ndarray
        Output composite raster, shape = (bands, rows, cols).
    best_idx : np.ndarray
        Index of selected input image per pixel, shape = (rows, cols).
    profile : dict
        Rasterio profile for writing output.
    """

    stack, qa_stack, profile, doy, scene_cloud_pct, skipped = load_valid_hls_scenes(
        image_paths=image_paths,
        fmask_paths=fmask_paths,
        doy=doy,
        scene_cloud_pct=scene_cloud_pct,
    )

    n, bands, rows, cols = stack.shape

    doy_score = gaussian_score(doy, target=target_doy, sigma=doy_sigma)[
        :, None, None
    ].astype(np.float32)

    if scene_cloud_pct is None:
        scene_score = np.ones((n, 1, 1), dtype=np.float32)
    else:
        scene_cloud_pct = np.asarray(scene_cloud_pct, dtype=np.float32)
        scene_score = np.exp(-scene_cloud_pct / 20)[:, None, None].astype(np.float32)

    clear = np.empty((n, rows, cols), dtype=bool)
    dist_scores = np.empty((n, rows, cols), dtype=np.float32)

    for i in range(n):
        qa = qa_stack[i, 0].astype(np.uint8)

        clear_i = create_hls_clear_mask(
            qa,
            include_water=include_water,
            mask_medium_high_aerosol=mask_medium_high_aerosol,
        )

        clear[i] = clear_i
        dist_scores[i] = cloud_distance_score(clear_i, max_distance=max_cloud_distance)

    score = doy_score * scene_score * dist_scores
    score = np.where(clear, score, -np.inf)

    best_idx = np.argmax(score, axis=0)
    no_valid = np.all(~np.isfinite(score), axis=0)

    composite = np.empty((bands, rows, cols), dtype=stack.dtype)

    for b in range(bands):
        composite[b] = np.take_along_axis(
            stack[:, b, :, :], best_idx[None, :, :], axis=0
        )[0]

    nodata = profile.get("nodata", -9999)
    composite[:, no_valid] = nodata

    return composite, best_idx, profile, skipped


def expand_window(window, pad, height, width):
    col_off = max(0, window.col_off - pad)
    row_off = max(0, window.row_off - pad)

    col_end = min(width, window.col_off + window.width + pad)
    row_end = min(height, window.row_off + window.height + pad)

    return Window(
        col_off=col_off,
        row_off=row_off,
        width=col_end - col_off,
        height=row_end - row_off,
    )


def best_available_pixel_hls_windowed(
    image_paths,
    fmask_paths,
    doy,
    scene_cloud_pct=None,
    target_doy=200,
    doy_sigma=30,
    max_cloud_distance=10,
    include_water=True,
    mask_medium_high_aerosol=True,
):
    """
    Create a Best Available Pixel (BAP) composite using windowed processing.

    This version processes one raster block at a time instead of loading all
    scenes into memory. It keeps the original BAP scoring logic:

        score = DOY_score * scene_cloud_score * cloud_distance_score

    Parameters
    ----------
    image_paths : list
        Paths to multiband image rasters.
    fmask_paths : list
        Paths to corresponding HLS Fmask QA rasters.
    doy : list or array-like
        Day-of-year for each scene.
    scene_cloud_pct : list or array-like, optional
        Scene-level cloud cover percentage for each scene.
    target_doy : int
        Preferred day of year.
    doy_sigma : float
        Gaussian spread around target DOY.
    max_cloud_distance : int
        Distance in pixels where cloud-distance score saturates.
    include_water : bool
        If False, water pixels are masked out.
    mask_medium_high_aerosol : bool
        If True, medium/high aerosol pixels are masked out.

    Returns
    -------
    composite : np.ndarray
        Composite raster with shape (bands, rows, cols).
    best_idx : np.ndarray
        Selected scene index per pixel, shape (rows, cols).
    profile : dict
        Rasterio profile from the first image.
    skipped : list
        Empty list, included for compatibility.
    """

    doy = np.asarray(doy, dtype=np.float32)
    n = len(image_paths)

    doy_score = gaussian_score(doy, target_doy, doy_sigma).astype(np.float32)

    if scene_cloud_pct is None:
        scene_score = np.ones(n, dtype=np.float32)
    else:
        scene_cloud_pct = np.asarray(scene_cloud_pct, dtype=np.float32)
        scene_score = np.exp(-scene_cloud_pct / 20).astype(np.float32)

    base_score = doy_score * scene_score

    srcs = [rasterio.open(p) for p in image_paths]
    qa_srcs = [rasterio.open(p) for p in fmask_paths]

    try:
        profile = srcs[0].profile.copy()
        bands = srcs[0].count
        height = srcs[0].height
        width = srcs[0].width
        dtype = srcs[0].dtypes[0]
        nodata = profile.get("nodata", -9999)

        composite = np.full((bands, height, width), nodata, dtype=dtype)
        best_idx = np.full((height, width), -1, dtype=np.int16)

        block_height, block_width = srcs[0].block_shapes[0]
        n_windows = ceil(height / block_height) * ceil(width / block_width)

        for _, window in tqdm(
            srcs[0].block_windows(1),
            total=n_windows,
            desc="Processing tile windows",
            leave=False,
        ):
            row0 = int(window.row_off)
            col0 = int(window.col_off)
            h = int(window.height)
            w = int(window.width)

            padded_window = expand_window(
                window,
                pad=max_cloud_distance,
                height=height,
                width=width,
            )

            prow0 = int(window.row_off - padded_window.row_off)
            pcol0 = int(window.col_off - padded_window.col_off)

            block_best_score = np.full((h, w), -np.inf, dtype=np.float32)
            block_best_idx = np.full((h, w), -1, dtype=np.int16)
            block_composite = np.full((bands, h, w), nodata, dtype=dtype)

            for i in range(n):
                qa = qa_srcs[i].read(
                    1,
                    window=padded_window,
                    out_dtype="uint8",
                )

                clear_padded = create_hls_clear_mask(
                    qa,
                    include_water=include_water,
                    mask_medium_high_aerosol=mask_medium_high_aerosol,
                )

                dist_padded = cloud_distance_score(
                    clear_padded,
                    max_distance=max_cloud_distance,
                )

                clear = clear_padded[prow0 : prow0 + h, pcol0 : pcol0 + w]
                dist_score = dist_padded[prow0 : prow0 + h, pcol0 : pcol0 + w]

                score = base_score[i] * dist_score
                score[~clear] = -np.inf

                update = score > block_best_score

                if np.any(update):
                    img = srcs[i].read(window=window)

                    block_composite[:, update] = img[:, update]
                    block_best_score[update] = score[update]
                    block_best_idx[update] = i

            composite[:, row0 : row0 + h, col0 : col0 + w] = block_composite
            best_idx[row0 : row0 + h, col0 : col0 + w] = block_best_idx

        return composite, best_idx, profile, []

    finally:
        for src in srcs:
            src.close()

        for src in qa_srcs:
            src.close()


def write_raster(output_path, array, profile):
    profile = profile.copy()
    profile.update(count=array.shape[0], dtype=array.dtype, compress="deflate")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)


if __name__ == "__main__":
    composite_dir = Path(r"F:\yukon\processed\L30\2023\composites")
    fmask_dir = Path(r"F:\yukon\processed\L30\2023\fmask")
    metadata_dir = Path(r"F:\yukon\processed\L30\2023\metadata")
    output_dir = Path(r"F:\yukon\processed\L30\2023\bap_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = index_files(composite_dir, fmask_dir, metadata_dir)
    if skipped:
        print(f"Skipped {len(skipped)} incomplete scene groups.")
    records_by_tile = group_records_by_tile(records)

    for tile, tile_records in tqdm(records_by_tile.items(), desc="BAP Processing"):

        image_paths = [r["image_path"] for r in tile_records]
        fmask_paths = [r["fmask_path"] for r in tile_records]
        doy = [r["doy"] for r in tile_records]
        scene_cloud_pct = [r["scene_cloud_pct"] for r in tile_records]
        out_path = output_dir / f"{tile}_bap_composite.tif"
        if out_path.exists():
            continue
        try:
            # composite, best_idx, profile, skipped = best_available_pixel_hls(
            #     image_paths=image_paths,
            #     fmask_paths=fmask_paths,
            #     doy=doy,
            #     scene_cloud_pct=scene_cloud_pct,
            #     target_doy=213,
            #     doy_sigma=40,
            #     max_cloud_distance=10,
            #     include_water=True,
            #     mask_medium_high_aerosol=True,
            # )

            composite, best_idx, profile, skipped = best_available_pixel_hls_windowed(
                image_paths=image_paths,
                fmask_paths=fmask_paths,
                doy=doy,
                scene_cloud_pct=scene_cloud_pct,
                target_doy=213,
                doy_sigma=40,
                max_cloud_distance=10,
                include_water=True,
                mask_medium_high_aerosol=True,
            )

            out_path = output_dir / f"{tile}_bap_composite.tif"

            write_raster(out_path, composite, profile)
        except Exception as e:
            tqdm.write(f"Error processing {tile} {doy}: {e}")
