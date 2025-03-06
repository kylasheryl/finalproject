#!usr/bin/python3
# -*- coding: utf-8 -*-
"""
convert year month day hour min sec to decimal year and vice versa

@author tgoebel - UC Santa Cruz
"""

import numpy as np
import time
import datetime
import calendar
from datetime import datetime as dt

def mo_to_sec(value):
    return value * (aveDyYr() / 12) * 24 * 3600

def sec_to_mo(value):
    return value / ((aveDyMo()) * 24 * 3600)

def dy_to_sec(value):
    return value * 24 * 3600

def sec_to_dy(value):
    return value / (24 * 3600)

def aveDyYr():
    """ how many days in a year"""
    return 365 + 1/4 - 1/100 + 1/400

def aveDyMo(): 
    """ how many days in a month """
    return aveDyYr() / 12

def checkDateTime(dateTime):
    """ check that hour != 24, MN != 60, SC != 60 """
    YR, MO, DY, HR, MN, SC = (
        int(dateTime[0]), int(dateTime[1]), int(dateTime[2]), 
        int(dateTime[3]), int(dateTime[4]), float(dateTime[5])
    )

    if SC < 0:
        SC = 0
    elif SC >= 60:
        MN += int(SC // 60)
        SC -= 60 * int(SC // 60)

    if MN < 0:
        MN = 0
    elif MN >= 60:
        HR += int(MN // 60)
        MN -= 60 * int(MN // 60)

    if HR < 0:
        HR = 0
    elif HR >= 24:
        HR = 23
        MN = 59
        SC = 59.999

    return YR, MO, DY, HR, MN, SC

# Konversi tanggal ke format desimal
def dateTime2decYr(datetime_in):
    """
    Convert array containing time columns year - second to decimal year.
    """
    try:
        o_dt = datetime.datetime(
            int(datetime_in[0]), int(datetime_in[1]), int(datetime_in[2]), 
            int(datetime_in[3]), int(datetime_in[4]), int(round(datetime_in[5]) - 1e-3)
        )
    except ValueError as e:
        error_msg = f"datetime array not valid - {datetime_in}; check if date and time are correct."
        raise ValueError(error_msg) from e

    time_sc = o_dt.hour * 3600 + o_dt.minute * 60 + o_dt.second
    dayOfYear_seconds = (o_dt.timetuple().tm_yday - 1) * 86400.0 + time_sc

    if calendar.isleap(o_dt.year):
        year_fraction = dayOfYear_seconds / (86400.0 * 366)
    else:
        year_fraction = dayOfYear_seconds / (86400.0 * 365)

    return o_dt.year + year_fraction
