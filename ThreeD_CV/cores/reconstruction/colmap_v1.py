"""
An example for running incremental SfM on images with the pycolmap interface.
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import enlighten

import pycolmap
from pycolmap import logging


def incremental_mapping_with_pbar(database_path, image_path, sfm_path):
    try:
        with pycolmap.Database.open(str(database_path)) as database:
            num_images = database.num_images()
    except:
        database = pycolmap.Database(str(database_path))
        num_images = database.num_images
        
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
            )
    return reconstructions


def run():
    # output_path = Path("/HDD/etc/outputs/colmap/data/Fountain/outputs")
    # image_path = Path("/HDD/etc/outputs/colmap/data/Fountain/images")
    output_path = Path("/HDD/etc/outputs/colmap/data/custom_v1/outputs")
    image_path = Path("/HDD/etc/outputs/colmap/data/custom_v1/images")
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")

    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    recs = incremental_mapping_with_pbar(database_path, image_path, sfm_path)
    # alternatively, use:
    # import custom_incremental_pipeline
    # recs = custom_incremental_pipeline.main(
    #     database_path, image_path, sfm_path
    # )
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")


if __name__ == "__main__":
    run()