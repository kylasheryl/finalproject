from datetime import datetime, timedelta
import calendar

class TimeConverter:
    SECONDS_PER_DAY = 86400.0
    
    @staticmethod
    def get_days_in_year():
        """Average days in a year accounting for leap years"""
        return 365.0 + 1/4.0 - 1/100.0 + 1/400.0
    
    @staticmethod
    def get_days_in_month():
        """Average days in a month"""
        return TimeConverter.get_days_in_year() / 12.0
    
    @staticmethod
    def convert_month_to_seconds(months):
        """Convert months to seconds"""
        return months * (TimeConverter.get_days_in_year() / 12.0) * TimeConverter.SECONDS_PER_DAY
    
    @staticmethod
    def convert_seconds_to_months(seconds):
        """Convert seconds to months"""
        return seconds / (TimeConverter.get_days_in_month() * TimeConverter.SECONDS_PER_DAY)
        
    @staticmethod
    def validate_datetime(year, month, day, hour, minute, second):
        """
        Validates and adjusts datetime components to ensure they're within valid ranges
        Returns: tuple of corrected (year, month, day, hour, minute, second)
        """
        # Handle seconds overflow
        minute_overflow, second = divmod(max(0, second), 60)
        minute += int(minute_overflow)
        
        # Handle minutes overflow
        hour_overflow, minute = divmod(max(0, minute), 60)
        hour += int(hour_overflow)
        
        # Handle hours overflow/invalid values
        if hour < 0:
            hour = 0
        elif hour >= 24:
            return (year, month, day, 23, 59, 59.999)
            
        return (year, month, day, hour, minute, second)
    
    @staticmethod
    def datetime_to_decimal_year(year, month, day, hour, minute, second):
        """
        Converts datetime components to decimal year
        Returns: float representing decimal year
        """
        try:
            # Validate input
            year, month, day, hour, minute, second = TimeConverter.validate_datetime(
                int(year), int(month), int(day), int(hour), int(minute), float(second)
            )
            
            # Create datetime object
            dt = datetime(year, month, day, hour, minute, int(second))
            
            # Calculate seconds into the year
            seconds_into_day = hour * 3600 + minute * 60 + second
            day_of_year = dt.timetuple().tm_yday - 1
            total_seconds = day_of_year * TimeConverter.SECONDS_PER_DAY + seconds_into_day
            
            # Calculate year fraction based on leap year status
            days_in_year = 366.0 if calendar.isleap(year) else 365.0
            year_fraction = total_seconds / (TimeConverter.SECONDS_PER_DAY * days_in_year)
            
            return year + year_fraction
            
        except ValueError as e:
            raise ValueError(f"Invalid datetime components: {e}")
            
    @staticmethod
    def decimal_year_to_datetime(decimal_year):
        """
        Converts decimal year back to datetime components
        Returns: tuple of (year, month, day, hour, minute, second)
        """
        year = int(decimal_year)
        year_fraction = decimal_year - year
        
        days_in_year = 366.0 if calendar.isleap(year) else 365.0
        seconds_in_year = days_in_year * TimeConverter.SECONDS_PER_DAY
        
        total_seconds = int(year_fraction * seconds_in_year)
        
        days = total_seconds // TimeConverter.SECONDS_PER_DAY
        remaining_seconds = total_seconds % TimeConverter.SECONDS_PER_DAY
        
        dt = datetime(year, 1, 1) + timedelta(days=days, seconds=remaining_seconds)
        
        return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)