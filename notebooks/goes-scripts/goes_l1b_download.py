# Builtin modules
import os
import shutil
import subprocess
import warnings
from datetime import datetime, timedelta

# External modules
from google.cloud import storage
from dateutil.parser import parse as parse_date
import numpy as np
import xarray as xr

# set path to google cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'ADD GOOGLE CREDENTIALS FILE HERE'


if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    raise ImportError(
        """
        'GOOGLE_APPLICATION_CREDENTIALS' is not set, this is required for IO
        operations involving Google Cloud Storage!

        To continue, please set the 'GOOGLE_APPLICATION_CREDENTIALS' variable
        either:

        1. In your terminal:
        export GOOGLE_APPLICATION_CREDENTIALS='path-to-your-credentials-file.json'

        2. In python:
        import os
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path-to-your-credentials-file.json'

        Then import io once this is set
        """
    )

## cannot get project automatically, so it is provided as argument
storage_client = storage.Client(project = 'goes-olr')
goes_16_bucket = storage_client.get_bucket("gcp-public-data-goes-16")


def _test_subprocess_command(shell_command):
    """
    Test if a shell command can be successfully executed by suprocess

    inputs:
    -- shell_command (str): shell command that will be passed to subprocess

    outputs:
    -- boolean: True if command executed without errors. False if called process
        returned an error
    """
    try:
        test = subprocess.check_output(shell_command.split(" "))
    except subprocess.CalledProcessError:
        return False
    else:
        return True


# Check if we can use ncdump to check if files are correct:
ncdump_exists = _test_subprocess_command("ncdump")

if not ncdump_exists:
    warnings.warn(
        """Warning: ncdump shell command not found, resorting to xarray
                     for file checking"""
    )


def _check_file_size_against_blob(filename, blob, rtol=0.1, atol=1):
    """
    Check if a file size on disk is within toterance of a GCS blob

    inputs:
    -- filename (str): path to file
    -- blob (Blob): GCS blob of file to compare
    -- rtol (float; default=0.1): Relative tolerance in bytes (see numpy.isclose)
    -- atol (float; default=1): Absolute tolerance in bytes (see numpy.isclose)

    outputs:
    -- boolean: True if file size is within tolerance of the blob. False if
        otherwise
    """
    filesize = os.stat(filename).st_size
    blobsize = blob.size
    return np.isclose(filesize, blobsize, rtol, atol)


def _check_ncdump_is_valid(filename):
    """
    Checks if a netcdf file can be examined using ncdump

    inputs:
    -- filename (str): path to file

    outputs:
    -- boolean: True if file can be examined using ncdump. False if otherwise
    """
    if ncdump_exists:
        if _test_subprocess_command(f"ncdump -h {filename}"):
            return True
        else:
            return False
    else:
        raise RuntimeError("ncdump cannot be called")


def _check_xarray_is_valid(filename):
    """
    Checks if a netcdf file can be opened using xarray

    inputs:
    -- filename (str): path to file

    outputs:
    -- boolean: True if file can be opened using xarray. False if otherwise
    """
    try:
        with xr.open_dataset(filename) as ds:
            pass
    except OSError:
        return False
    else:
        return True


def _check_netcdf_file_is_valid(filename):
    """
    Checks if a netcdf file can be opened. Uses ncdump if available, otherwise
    attempts to open the file using xarrays

    inputs:
    -- filename (str): path to file

    outputs:
    -- boolean: True if file can be opened. False if otherwise
    """
    if ncdump_exists:
        return _check_ncdump_is_valid(filename)
    else:
        return _check_xarray_is_valid(filename)


def _check_free_space_for_blob(
    blob, save_path, relative_min_storage=1.25, absolute_min_storage=2**30
):
    """
    Checks if the avaialbe space on a storage volume is large enough to download
    a blob. Min storage required is the largest out of the relative_min_storage
    multiplied by the blob size, and the absolute_min_storage.

    inputs:
    -- blob (Blob): GCS blob of file to compare
    -- save_path (str): direfctory to save file to
    -- relative_min_storage (float; default=1.25): Relative tolerance in bytes (see numpy.isclose)
    -- absolute_min_storage (float; default=2**30 (1GB)): Absolute tolerance in bytes (see numpy.isclose)

    outputs:
    -- boolean: True if free storage is above min threhsold, otherwise False.
    """
    min_storage = np.maximum(blob.size * relative_min_storage, absolute_min_storage)

    if shutil.disk_usage(save_path).free > min_storage:
        return True
    else:
        return False


def _check_if_file_exists_and_is_valid(filename, blob, remove_corrupt=True):
    """
    Checks if a netcdf file exists and can be opened.

    inputs:
    -- filename (str): path to file
    -- blob (Blob): GCS blob of file to compare
    -- remove_corrupt (bool; default is True): If true, if the file exists but
        is not valid then delete the file. Otherwise leave in place

    outputs:
    -- boolean: True if file can be opened. False if otherwise
    """
    if os.path.exists(filename):
        if _check_file_size_against_blob(filename, blob):
            return True
        else:
            if remove_corrupt:
                os.remove(filename)

            return False
    else:
        return False


def _find_abi_blobs(date, satellite=16, product="Rad", view="C", mode=3, channel=1):
    """
    Find ABI file blobs for a single set of input parameters

    inputs:
    -- date (datetime): date to find files (hour)
    -- satellite (int; default=16): GOES satellite to get data for. 16 or 17
    -- product (str; default='Rad'): ABI data product to download
    -- view (str; default='C'): View to get data for
    -- mode (int; default=4): Scanning mode to get data for
    -- channel (int; default=1): Channel to get data for. Only used for Rad and
        CMIP products

    outputs:
    -- list of Blobs: list of blobs found using the prefix generated from the
        supplied inputs
    """
    if satellite == 16:
        goes_bucket = goes_16_bucket
    elif satellite == 17:
        goes_bucket = goes_17_bucket
    else:
        raise ValueError("Invalid input for satellite keyword")

    doy = (date - datetime(date.year, 1, 1)).days + 1

    level = "L1b" if product == "Rad" else "L2"
    
    blob_path = "ABI-%s-%s%.1s/%04d/%03d/%02d/" % (
        level,
        product,
        view,
        date.year,
        doy,
        date.hour,
    )

    if product == "Rad" or "CMIP":
        blob_prefix = "OR_ABI-%s-%s%s-M%1dC%02d_G%2d_s" % (
            level,
            product,
            view,
            mode,
            channel,
            satellite,
        )
        
    else:
        blob_prefix = "OR_ABI-%s-%s%s-M%1d_G%2d_s" % (
            level,
            product,
            view,
            mode,
            satellite,
        )
    print(f'path: {blob_path + blob_prefix}\n')
    blobs = list(goes_bucket.list_blobs(prefix=blob_path + blob_prefix, delimiter="/"))

    print(f'found blobs {blobs}\n\n')
    print(f'returning first blob {[blobs[0]]}\n\n')
    # return blobs
    return [blobs[0]]


def find_abi_blobs(
    dates, satellite=16, product="Rad", view="C", mode=[3, 4, 6], channel=1
):
    """
    Find ABI file blobs for a set of one or multiple input parameters

    inputs:
    -- dates (datetime): date or list of dates to find files (hour)
    -- satellite (int; default=16): GOES satellite to get data for. 16 or 17
    -- product (str; default='Rad'): ABI data product to download
    -- view (str; default='C'): View to get data for
    -- mode (int; default=[3,4,6]): Scanning mode to get data for
    -- channel (int; default=1): Channel to get data for. Only used for Rad and
        CMIP products

    outputs:
    -- list of Blobs: list of blobs found using the prefix generated from the
        supplied inputs
    """

    input_params = [
        m.ravel().tolist()
        for m in np.meshgrid(dates, satellite, product, view, mode, channel)
    ]
    n_params = len(input_params[0])

    blobs = sum(
        [
            _find_abi_blobs(
                input_params[0][i],
                satellite=input_params[1][i],
                product=input_params[2][i],
                view=input_params[3][i],
                mode=input_params[4][i],
                channel=input_params[5][i],
            )
            for i in range(n_params)
        ],
        [],
    )

    return blobs


def _get_download_destination(blob, save_dir, replicate_path=True):
    """
    Generate path to download destination for a blob

    inputs:
    -- blob (Blob): GCS blob to find download location for
    -- save_dir (str): root directory to save files in
    -- replicate_path (bool; default=True): If True, replicate the directory
        structure within the GCS store. Otherwise, provide a path to the file
        directly within the root directory

    outputs:
    -- str: save_file name and location for download
    """
    blob_path, blob_name = os.path.split(blob.name)

    if replicate_path:
        save_path = os.path.join(save_dir, blob_path)
    else:
        save_path = save_dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, blob_name)
    return save_file


def download_blob(
    blob,
    save_dir,
    replicate_path=True,
    check_download=False,
    n_attempts=1,
    clobber=False,
    min_storage=2**30,
    verbose=False,
    remove_corrupt=True,
):
    """
    Download a single blob from GCS

    inputs:
    -- blob (Blob): GCS blob to find download location for
    -- save_dir (str): root directory to save files in
    -- replicate_path (bool; default=True): If True, replicate the directory
        structure within the GCS store. Otherwise, provide a path to the file
        directly within the root directory
    -- check_download (bool; default=True)
    """
    save_path = _get_download_destination(blob, save_dir, replicate_path=replicate_path)
    if clobber or not os.path.exists(save_path):
        if clobber and os.path.exists(save_path):
            os.remove(save_path)
        if verbose:
            print(f"Downloading {save_path}", flush=True)
        if _check_free_space_for_blob(
            blob,
            os.path.split(save_path)[0],
            relative_min_storage=1.25,
            absolute_min_storage=min_storage,
        ):
            try:
                blob.download_to_filename(save_path)
            except OSError:
                if n_attempts > 1:
                    download_blob(
                        blob,
                        save_dir,
                        replicate_path=replicate_path,
                        check_download=check_download,
                        n_attempts=n_attempts - 1,
                        clobber=clobber,
                    )
                else:
                    raise RuntimeError(f"{save_path}: download failed")
            if check_download and not _check_if_file_exists_and_is_valid(
                save_path, blob, remove_corrupt=True
            ):
                if n_attempts > 1:
                    download_blob(
                        blob,
                        save_dir,
                        replicate_path=replicate_path,
                        check_download=check_download,
                        n_attempts=n_attempts - 1,
                        clobber=clobber,
                    )
                else:
                    if remove_corrupt:
                        if os.path.exists(save_path):
                            os.remove(save_path)
                    raise RuntimeError(f"{save_path}: downloaded file not valid")
        else:
            raise OSError("Not enough storage space available for download")
    if os.path.exists(save_path):
        return save_path
    else:
        raise RuntimeError(f"{save_path}: downloaded file not found")



def get_goes_date(filename: str) -> datetime:
    """
    Finds the centre point time from an ABI filename
    """
    base_string = os.path.split(filename)[-1]

    start_string = base_string.split("_s")[-1]
    start_date = parse_date(start_string[:4] + "0101" + start_string[7:13]) + timedelta(
        days=int(start_string[4:7]) - 1
    )
    end_string = base_string.split("_e")[-1]
    end_date = parse_date(end_string[:4] + "0101" + end_string[7:13]) + timedelta(
        days=int(end_string[4:7]) - 1
    )

    return start_date + (end_date - start_date) / 2


def find_abi_files(
    date,
    satellite=16,
    product="Rad",
    view="C",
    mode=[3, 4, 6],
    channel=1,
    save_dir="./",
    replicate_path=True,
    check_download=False,
    n_attempts=1,
    download_missing=False,
    clobber=False,
    min_storage=2**30,
    remove_corrupt=True,
    verbose=False,
):
    """
    Finds ABI files on the local system. Optionally downloads missing files from
    GCS

    inputs:
    # TODO:

    outputs:
        list: list of filenames on local system
    """
    blobs = find_abi_blobs(
        date,
        satellite=satellite,
        product=product,
        view=view,
        mode=mode,
        channel=channel,
    )
    files = []
    for blob in blobs:
        if download_missing:
            try:
                save_file = download_blob(
                    blob,
                    save_dir,
                    replicate_path=replicate_path,
                    check_download=check_download,
                    n_attempts=n_attempts,
                    clobber=clobber,
                    min_storage=min_storage,
                    verbose=verbose,
                )
            except OSError as e:
                warnings.warn(str(e.args[0]))
                download_missing = False
            except RuntimeError as e:
                warnings.warn(str(e.args[0]))
            else:
                if os.path.exists(save_file):
                    files += [save_file]
        else:
            local_file = _get_download_destination(
                blob, save_dir, replicate_path=replicate_path
            )
            if _check_if_file_exists_and_is_valid(
                local_file, blob, remove_corrupt=remove_corrupt
            ):
                files += [local_file]
    return files


def _find_glm_blobs(date, satellite=16):
    if satellite == 16:
        goes_bucket = goes_16_bucket
    elif satellite == 17:
        goes_bucket = goes_17_bucket
    else:
        raise ValueError("Invalid input for satellite keyword")

    doy = (date - datetime(date.year, 1, 1)).days + 1

    blob_path = "GLM-L2-LCFA/%04d/%03d/%02d/" % (date.year, doy, date.hour)
    blob_prefix = "OR_GLM-L2-LCFA_G%2d_s" % satellite

    blobs = list(goes_bucket.list_blobs(prefix=blob_path + blob_prefix, delimiter="/"))

    return blobs


def find_glm_blobs(dates, satellite=16):
    """
    Find GLM file blobs for a set of one or multiple input parameters

    inputs:
    -- dates (datetime): date or list of dates to find files (hour)
    -- satellite (int; default=16): GOES satellite to get data for. 16 or 17

    outputs:
    -- list of Blobs: list of blobs found using the prefix generated from the
        supplied inputs
    """

    input_params = [m.ravel().tolist() for m in np.meshgrid(dates, satellite)]
    n_params = len(input_params[0])

    blobs = sum(
        [
            _find_glm_blobs(input_params[0][i], satellite=input_params[1][i])
            for i in range(n_params)
        ],
        [],
    )

    return blobs


def find_glm_files(
    date,
    satellite=16,
    save_dir="./",
    replicate_path=True,
    check_download=False,
    n_attempts=1,
    download_missing=False,
    clobber=False,
    min_storage=2**30,
    remove_corrupt=True,
    verbose=False,
):
    """
    Finds GLM files on the local system. Optionally downloads missing files from
    GCS

    inputs:
    # TODO:

    outputs:
        list: list of filenames on local system
    """
    blobs = find_glm_blobs(date, satellite=satellite)
    files = []
    for blob in blobs:
        if download_missing:
            try:
                save_file = download_blob(
                    blob,
                    save_dir,
                    replicate_path=replicate_path,
                    check_download=check_download,
                    n_attempts=n_attempts,
                    clobber=clobber,
                    min_storage=min_storage,
                    verbose=verbose,
                )
            except OSError as e:
                warnings.warn(e.args[0])
                download_missing = False
            except RuntimeError:
                warnings.warn(e.args[0])
            else:
                if os.path.exists(save_file):
                    files += [save_file]
        else:
            save_file = _get_download_destination(
                blob, save_dir, replicate_path=replicate_path
            )
            if _check_if_file_exists_and_is_valid(save_file, blob):
                files += [save_file]
    return files


# def find_nexrad_blobs(date, site):
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket("gcp-public-data-nexrad-l2")

#     blob_path = "%04d/%02d/%02d/%s/" % (date.year, date.month, date.day, site)
#     blob_prefix = "NWS_NEXRAD_NXL2DPBL_%s_%04d%02d%02d%02d" % (
#         site,
#         date.year,
#         date.month,
#         date.day,
#         date.hour,
#     )

#     blobs = list(bucket.list_blobs(prefix=blob_path + blob_prefix, delimiter="/"))

#     return blobs


def download_blobs(
    blob_list, save_dir="./", replicate_path=True, n_attempts=0, clobber=False
):
    for blob in blob_list:
        blob_path, blob_name = os.path.split(blob.name)

        if replicate_path:
            save_path = os.path.join(save_dir, blob_path)
        else:
            save_path = save_dir
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_file = os.path.join(save_path, blob_name)
        if clobber or not os.path.exists(save_file):
            blob.download_to_filename(save_file)


# def find_nexrad_files(
#     date, site, save_dir="./", replicate_path=True, download_missing=False
# ):
#     blobs = find_nexrad_blobs(date, site)
#     files = []
#     for blob in blobs:
#         blob_path, blob_name = os.path.split(blob.name)

#         if replicate_path:
#             save_path = os.path.join(save_dir, blob_path)
#         else:
#             save_path = save_dir
#         if not os.path.isdir(save_path):
#             os.makedirs(save_path, exist_ok=True)

#         save_file = os.path.join(save_path, blob_name)
#         if os.path.exists(save_file):
#             files += [save_file]
#         elif download_missing:
#             download_blobs([blob], save_dir=save_dir, replicate_path=replicate_path)
#             if os.path.exists(save_file):
#                 files += [save_file]

#     return files