
reflectance_attrs = dict(
    name="reflectance",
    standard_name="toa_bidirectional_reflectance",
    full_name="TOA Bidirectional Reflectance",
    short_name="reflectance",
    units="%"
)

radiance_attrs = dict(
    name="radiance",
    standard_name="toa_outgoing_radiance_per_unit_wavelength",
    full_name="TOA Bidirectional Reflectance",
    short_name="radiance",
    units="Watts/m^2/micrometer/steradian"
)

counts_attrs = dict(
    name="counts",
    standard_name="counts",
    full_name="Counts",
    short_name="counts",
    units="counts"
)

brightness_temperature_attrs = dict(
    name="brightness_temperature",
    standard_name="toa_brightness_temperature",
    full_name="Brightness Temperature",
    short_name="brightness_temperature",
    units="K"
)

VARIABLE_ATTRS = dict(
    reflectance=reflectance_attrs,
    radiance=radiance_attrs,
    counts=counts_attrs,
    brightness_temperature=brightness_temperature_attrs
)
