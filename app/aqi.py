"""
Air Quality API client and data models for Open-Meteo.

Handles fetching real-time AQI, PM2.5, PM10, and hourly forecasts.
Includes in-memory caching with TTL.
"""

import logging
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OPEN_METEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

# In-memory cache with TTL
_cache = {}
CACHE_TTL_SECONDS = 3600  # 1 hour for AQI; 86400 (24h) for geocoding


class GeocodingResult(BaseModel):
    """Geocoding result from Open-Meteo."""

    name: str
    latitude: float
    longitude: float
    country: str
    admin1: str | None = None
    timezone: str | None = None


class AirQualityData(BaseModel):
    """Current and hourly air quality data."""

    pm25: float = Field(..., description="PM2.5 in µg/m³")
    pm10: float = Field(..., description="PM10 in µg/m³")
    us_aqi: float = Field(..., description="US AQI scale (0–500+)")
    european_aqi: float | None = Field(None, description="European AQI")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class HourlyForecast(BaseModel):
    """Hourly forecast for air quality."""

    hour: int = Field(..., description="Hour 0–23 UTC")
    pm25: float = Field(..., description="PM2.5 µg/m³")
    pm10: float = Field(..., description="PM10 µg/m³")


async def geocode_city(city: str) -> GeocodingResult | None:
    """
    Convert city name to lat/lon using Open-Meteo Geocoding API.

    Args:
        city: City name (e.g., "Delhi")

    Returns:
        GeocodingResult with lat/lon, or None if not found

    Note:
        Results are cached for 24 hours by city name.
    """
    cache_key = f"geo:{city.lower()}"
    if cache_key in _cache:
        cached, cached_at = _cache[cache_key]
        if datetime.now() - cached_at < timedelta(seconds=86400):
            logger.info(f"Geocoding cache hit for {city}")
            return cached

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPEN_METEO_GEOCODING_URL,
                params={"name": city, "count": 1, "language": "en"},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("results"):
                result = data["results"][0]
                geocoding = GeocodingResult(
                    name=result.get("name"),
                    latitude=result.get("latitude"),
                    longitude=result.get("longitude"),
                    country=result.get("country"),
                    admin1=result.get("admin1"),
                    timezone=result.get("timezone"),
                )
                _cache[cache_key] = (geocoding, datetime.now())
                logger.info(f"Geocoded {city} to ({geocoding.latitude}, {geocoding.longitude})")
                return geocoding
    except Exception as e:
        logger.error(f"Geocoding error for {city}: {e}")

    return None


async def fetch_air_quality(
    latitude: float, longitude: float, include_hourly: bool = True
) -> tuple[AirQualityData, list[HourlyForecast]] | None:
    """
    Fetch current and hourly air quality from Open-Meteo.

    Args:
        latitude: Location latitude
        longitude: Location longitude
        include_hourly: Whether to fetch 7-day hourly forecast

    Returns:
        Tuple of (current AirQualityData, list of HourlyForecast), or None on error

    Note:
        Results cached for 1 hour by lat/lon.
        Requests hourly PM2.5 and PM10 for next 7 days.
    """
    cache_key = f"aq:{latitude:.2f}:{longitude:.2f}"
    if cache_key in _cache:
        cached, cached_at = _cache[cache_key]
        if datetime.now() - cached_at < timedelta(seconds=CACHE_TTL_SECONDS):
            logger.info(f"AQ cache hit for ({latitude}, {longitude})")
            return cached

    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm2_5,pm10,us_aqi,european_aqi",
        }
        if include_hourly:
            params["hourly"] = "pm2_5,pm10"
            params["forecast_days"] = 7

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(OPEN_METEO_AQ_URL, params=params)
            response.raise_for_status()
            data = response.json()

        current = data.get("current", {})
        current_aq = AirQualityData(
            pm25=current.get("pm2_5", 0.0),
            pm10=current.get("pm10", 0.0),
            us_aqi=current.get("us_aqi", 0.0),
            european_aqi=current.get("european_aqi"),
            timestamp=current.get("time", datetime.now().isoformat()),
        )

        hourly_forecasts = []
        if include_hourly:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            pm25_list = hourly.get("pm2_5", [])
            pm10_list = hourly.get("pm10", [])

            for i, time_str in enumerate(times[:168]):  # 7 days * 24 hours
                try:
                    dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                    hour = dt.hour
                    pm25 = pm25_list[i] if i < len(pm25_list) else 0.0
                    pm10 = pm10_list[i] if i < len(pm10_list) else 0.0
                    hourly_forecasts.append(
                        HourlyForecast(hour=hour, pm25=max(pm25, 0), pm10=max(pm10, 0))
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing hourly forecast at index {i}: {e}")

        result = (current_aq, hourly_forecasts)
        _cache[cache_key] = (result, datetime.now())
        logger.info(f"Fetched AQ for ({latitude}, {longitude}): PM2.5={current_aq.pm25:.1f}")
        return result

    except Exception as e:
        logger.error(f"AQ fetch error for ({latitude}, {longitude}): {e}")

    return None


def clear_cache() -> None:
    """Clear all cached data (for testing)."""
    _cache.clear()
    logger.info("AQ cache cleared")
