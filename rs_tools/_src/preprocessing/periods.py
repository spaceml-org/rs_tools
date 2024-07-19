from typing import Optional, Iterable, List
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from dataclasses import dataclass
import pandas as pd
import geopandas as gpd


@dataclass
class TimeSeries:
    ts: List[Timestamp]
    time_delta: float | int = 1
    time_delta_unit: str = "minutes"

    @classmethod
    def init_from_geojson(
        cls,
        file_path: str,
        column_name: str = "time",
        time_delta: float = 15,
        time_delta_unit: str = "minutes",
    ):
        # read geojson
        gdf = gpd.read_file(file_path)
        # extract time
        ts = gdf["time"]
        # convert to timestamps
        ts = list(map(lambda x: pd.to_datetime(x), ts))
        # return class
        return cls(ts=ts, time_delta=time_delta, time_delta_unit=time_delta_unit)

    @classmethod
    def init_from_generation(
        cls,
        start_year: int = 2010,
        end_year: int = 2010,
        start_month: int = 10,
        end_month: int = 10,
        start_day: int = 1,
        end_day: int = 1,
        time_delta: float = 15,
        time_delta_unit: str = "minutes",
        start_hour: Optional[str] = None,
        end_hour: Optional[str] = None,
        start_minute: Optional[str] = None,
        end_minute: Optional[str] = None,
    ):
        # read geojson
        ts = generate_time_series(
            start_year=start_year, 
            end_year=end_year, 
            start_month=start_month,
            end_month=end_month,
            start_day=start_day,
            end_day=end_day,
            time_delta=time_delta,
            time_delta_unit=time_delta_unit,
            start_hour=start_hour,
            end_hour=end_hour,
            start_minute=start_minute,
            end_minute=end_minute,
            )
        # convert to timestamps
        ts = list(map(lambda x: pd.to_datetime(x), ts))
        # return class
        return cls(ts=ts, time_delta=time_delta, time_delta_unit=time_delta_unit)

    @property
    def datetime_index(self):
        return pd.DatetimeIndex(self.ts)

    @property
    def tmin(self):
        return self.datetime_index.min()

    @property
    def tmax(self):
        return self.datetime_index.min()

    def __iter__(self):
        for timestamp in self.ts:
            yield TimeQuery(timestamp, self.time_delta, self.time_delta_unit)


@dataclass
class TimeQuery:
    time_stamp: Timestamp
    time_delta: float | int = 1
    time_delta_unit: str = "minutes"

    @property
    def time_window(self):
        return pd.to_timedelta(self.time_delta, unit=self.time_delta_unit)

    @classmethod
    def init_from_date_string(
        cls, 
        time_stamp: str, 
        time_delta: float | int = 1,
        time_delta_unit: str = "minutes",
        ):

        time_stamp = pd.to_datetime(time_stamp)
        return cls(
            time_stamp=time_stamp,
            time_delta=time_delta,
            time_delta_unit=time_delta_unit,
            )


def generate_time_series(
        start_year: int = 2010,
        end_year: int = 2010,
        start_month: int = 10,
        end_month: int = 10,
        start_day: int = 1,
        end_day: int = 1,
        time_delta: float = 15,
        time_delta_unit: str = "minutes",
        start_hour: Optional[str] = None,
        end_hour: Optional[str] = None,
        start_minute: Optional[str] = None,
        end_minute: Optional[str] = None,
):
    # create start time
    start_date = f"{str(start_year).zfill(4)}"
    start_date += f"-{str(start_month).zfill(2)}"
    start_date += f"-{str(start_day).zfill(2)}"

    end_date = f"{str(end_year).zfill(4)}"
    end_date += f"-{str(end_month).zfill(2)}"
    end_date += f"-{str(end_day).zfill(2)}"

    freq = pd.Timedelta(value=time_delta, unit=time_delta_unit)
    ts = pd.date_range(start_date, end_date, freq=freq, )

    # filter
    if start_hour is not None:
        ts = ts[(ts.hour >= int(start_hour))]

    if end_hour is not None:
        ts = ts[(ts.hour <= int(end_hour))]

    if start_minute is not None:
        ts = ts[(ts.minute >= int(start_minute))]

    if end_minute is not None:
        ts = ts[(ts.minute <= int(end_minute))]

    return ts


def filter_time_series(
    ts: Iterable[Timestamp],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    start_month: Optional[int] = None,
    end_month: Optional[int] = None,
    start_day: Optional[int] = None,
    end_day: Optional[int] = None,
    start_hour: Optional[str] = None,
    end_hour: Optional[str] = None,
    start_minute: Optional[str] = None,
    end_minute: Optional[str] = None,
):
    # convert to datetime index
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    # YEAR
    if start_year is not None:
        ts = ts[(ts.year >= int(start_year))]
    if end_year is not None:
        ts = ts[(ts.year <= int(end_year))]

    # MONTH
    if start_month is not None:
        ts = ts[(ts.month >= int(start_month))]
    if end_month is not None:
        ts = ts[(ts.month <= int(end_month))]

    # DAY
    if start_day is not None:
        ts = ts[(ts.day >= int(start_day))]
    if end_day is not None:
        ts = ts[(ts.day <= int(end_day))]

    # HOUR
    if start_hour is not None:
        ts = ts[(ts.hour >= int(start_hour))]
    if end_hour is not None:
        ts = ts[(ts.hour <= int(end_hour))]

    # MINUTE
    if start_minute is not None:
        ts = ts[(ts.minute >= int(start_minute))]
    if end_minute is not None:
        ts = ts[(ts.minute <= int(end_minute))]

    return ts


def resample_datetime_index(
        ts: Iterable[Timestamp],
        time_delta: float = 15,
        time_delta_unit: str = "minutes",
):
    freq = pd.Timedelta(value=time_delta, unit=time_delta_unit)
    ts = pd.date_range(ts.min(), ts.max(), freq=freq, )

    return ts


def change_frequency(
        ts: Iterable[Timestamp],
        time_delta: float = 15,
        time_delta_unit: str = "minutes",
):
    freq = pd.Timedelta(value=time_delta, unit=time_delta_unit)
    ts = pd.date_range(ts.min(), ts.max(), freq=freq, )

    return ts