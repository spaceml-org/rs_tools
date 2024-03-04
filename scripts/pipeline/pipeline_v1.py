"""
A General Pipeline for create ML-Ready Data
- Downloading the Data
- Data Harmonization
- Normalizing
- Patching
"""



def main(stage: str="download"):

    # part 1
    if stage == "download":
        # download MODIS & GOES
        raise NotImplementedError()
    elif stage == "harmonize":
        # harmonize MODIS & GOES
        raise NotImplementedError()
    elif stage == "ml_processing":
        # normalize, Gap-fill, patch
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized stage - {stage}")

if __name__ == '__main__':
    main()