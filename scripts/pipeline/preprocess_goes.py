

@dataclass
class GOESGeoProcessing:
    resolution: float = 0.25 # in degrees?

    def geoprocess(self, params: GeoProcessingParams, files: List[str]):
        return None
        
    def parse_filenames(self, files: List[str]):
        # chunk the files
        # time, bands
        return None

@dataclass
class MLProcessingParams:
    # patching
    # normalization
    # gap-filling
    pass
