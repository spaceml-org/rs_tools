MSG_WAVELENGTHS_TO_BANDS = {
    1.64: "IR_016",
    3.92: "IR_039",
    8.70: "IR_087",
    9.66: "IR_097",
    10.80: "IR_108",
    12.00: "IR_120",
    13.40: "IR_134",
    0.64: "VIS006",
    0.81: "VIS008",
    6.25: "WV_062",
    7.35: "WV_073",
}
# All wavelengths in micrometers
MSG_BANDS_TO_WAVELENGTHS = {
    "IR_016": 1.64,
    "IR_039": 3.92,
    "IR_087": 8.70,
    "IR_097": 9.66,
    "IR_108": 10.80,
    "IR_120": 12.00,
    "IR_134": 13.40,
    "VIS006": 0.64,
    "VIS008": 0.81,
    "WV_062": 6.25,
    "WV_073": 7.35,
}

MSG_BANDS_TO_BAND_NAME = {
    1: "IR_016",
    2: "IR_039",
    3: "IR_087",
    4: "IR_097",
    5: "IR_108",
    6: "IR_120",
    7: "IR_134",
    8: "VIS006",
    9: "VIS008",
    10: "WV_062",
    11: "WV_073",
}

MSG_WAVELENGTHS = list(MSG_WAVELENGTHS_TO_BANDS.keys())

MSG_BANDS = list(MSG_BANDS_TO_WAVELENGTHS.keys())
