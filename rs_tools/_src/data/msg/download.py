from typing import Optional
from pathlib import Path
from eumdac.product import Product, ProductError
import shutil
from loguru import logger
from rs_tools._src.data.msg.query import msg_query_product
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import typer

app = typer.Typer()

MSG_EXTENSIONS = [".nat", ".grb"]


@app.command()
def msg_download_from_meta(
    meta_load_path: str,
    save_dir: str = "./",
    meta_save_path: Optional[str]=None
    
) -> pd.DataFrame:
    """
    Download data files based on metadata information.

    Args:
        meta_load_path (str): The path to the metadata file.
        save_dir (str, optional): The directory to save the downloaded files. Defaults to "./".
        meta_save_path (str, optional): The path to save the updated metadata file. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the downloaded data files' information.

    Raises:
        ValueError: If the file path extension is not recognized or the file type is unrecognized.

    """
    # load meta file
    # load file
    if isinstance(meta_load_path, pd.DataFrame | gpd.GeoDataFrame):
        df = meta_load_path
        
    elif isinstance(meta_load_path, Path | str):
        
        meta_load_path = Path(meta_load_path)
        
        assert meta_load_path.is_file()
        
        if meta_load_path.suffix == ".csv":
            df = pd.read_csv(meta_load_path, index_col=0)
            
        elif meta_load_path.suffix in [".geojson", ".json"]:
            df = gpd.read_file(meta_load_path, index_col=0)
        else:
            raise ValueError(f"Unrecognized filepath extension: {meta_load_path.suffix}")
    else:
        msg = f"Unrecognized filetype: {meta_load_path}"
        raise ValueError(msg)
    
    data_frames = list()
        
    pbar = tqdm(list(df.iterrows()))
    for irow, idata in pbar:
        
        pbar.set_description(f"Downloading: {idata['time']} | {idata['product_id']}")
        save_path = msg_download_from_product_id(
            product_id=idata["product_id"],
            collection_id=idata["collection_id"],
            save_dir=save_dir
        )
        idata["full_path"] = str(Path(save_path).absolute())
        data_frames.append(idata[1:])
    
    data_frames = pd.DataFrame(data_frames)
    data_frames = data_frames.drop_duplicates().reset_index(drop=True)
    
    if meta_save_path:
        Path(meta_save_path).parent.mkdir(parents=True, exist_ok=True)
        data_frames.to_csv(meta_save_path)
    
    return data_frames


@app.command()
def msg_download_from_product_id(
    product_id: str,
    collection_id: str,
    save_dir: str="./",
):
    """
    Downloads the MSG granule associated with the given product ID and collection ID.

    Args:
        product_id (str): The ID of the product.
        collection_id (str): The ID of the collection.
        save_dir (str, optional): The directory where the downloaded granule will be saved. Defaults to "./".

    Returns:
        bool: True if the granule was successfully downloaded, False otherwise.
    """
    product = msg_query_product(
        product_id=product_id,
        collection_id=collection_id
    )
    return msg_download_granule(product.granule, save_dir=save_dir)


def msg_download_granule(
    granule: Product,
    save_dir: str="./"
):
    """
    Downloads a specific file from a given granule and saves it to the specified directory.

    Args:
        granule (Product): The granule from which to download the file.
        save_dir (str, optional): The directory where the downloaded file will be saved. Defaults to "./".

    Returns:
        str: The path to the downloaded file.

    Raises:
        ProductError: If there is an error while downloading the file.
    """
    entries = list(granule.entries)
    filenames = list(filter(lambda x: Path(x).suffix in MSG_EXTENSIONS, entries))
    
    msg = f"No files to download"
    msg += f"\nEntries:\n{granule.entries}"
    assert len(filenames) == 1, msg
    
    filename = filenames[0]
    
    try:
        with granule.open(entry=filename) as fsrc:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir.joinpath(fsrc.name)
            
            with open(save_path, mode="wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            return save_path
    except ProductError as error:
        msg = f"Could not download {filename} from '{granule}': "
        msg += f"{error.msg}"
        print(msg)     


if __name__ == '__main__':
    app()