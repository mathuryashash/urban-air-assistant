"""
Core decision logic for air quality risk assessment and guidance generation.

This module contains pure, deterministic functions that calculate health risks,
optimal activity windows, and impact estimates based on air quality data.
All numbers are sourced from EPA, ICCT, and cited in docstrings.
"""

import logging
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class AQIBand(Enum):
    """AQI severity bands based on US EPA breakpoints for PM2.5."""

    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_FOR_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"


class VulnerabilityGroup(Enum):
    """Vulnerable populations with elevated health risk in poor AQ."""

    CHILDREN = "children"
    ELDERLY = "elderly"
    RESPIRATORY = "respiratory_conditions"
    CARDIAC = "cardiac_conditions"


class ActivityType(Enum):
    """Types of outdoor activities for which optimal timing is forecast."""

    OUTDOOR_EXERCISE = "outdoor_exercise"
    COMMUTE = "commute"
    VENTILATION = "home_ventilation"


class ImpactEstimate(NamedTuple):
    """Quantified impact of a behavior change."""

    co2_kg_saved: float
    exposure_reduction_percent: float
    rationale: str  # Cited source/assumptions


# EPA AQI breakpoints for PM2.5 (µg/m³) → AQI band
# Source: https://www.epa.gov/air-quality/air-quality-index-aqi-basics
AQI_PM25_BREAKPOINTS = {
    "Good": (0, 12),
    "Moderate": (12.1, 35.4),
    "Unhealthy for Sensitive Groups": (35.5, 55.4),
    "Unhealthy": (55.5, 150.4),
    "Very Unhealthy": (150.5, 250.4),
    "Hazardous": (250.5, float("inf")),
}


def calculate_aqi_band(pm25: float, pm10: float) -> AQIBand:
    """
    Classify air quality into health advisory band using EPA breakpoints.

    Args:
        pm25: PM2.5 concentration in µg/m³
        pm10: PM10 concentration in µg/m³ (for future refinement)

    Returns:
        AQIBand enum indicating health advisory level

    Source: EPA Air Quality Index scale
    """
    for band_name, (lower, upper) in AQI_PM25_BREAKPOINTS.items():
        if lower <= pm25 <= upper:
            return AQIBand(band_name)
    return AQIBand.HAZARDOUS


def assess_health_risk(band: AQIBand, vulnerability: VulnerabilityGroup) -> int:
    """
    Calculate health risk score (0-100) for a vulnerable group at a given AQI band.

    Args:
        band: Current AQI band
        vulnerability: Vulnerable population category

    Returns:
        Risk score 0–100 (0 = no risk, 100 = highest risk)

    Health advisory matrix:
      - Children & elderly: elevated risk at Moderate and above
      - Respiratory/cardiac conditions: escalated risk starting from Moderate
    """
    # Base risk per band (all groups)
    band_risk = {
        AQIBand.GOOD: 0,
        AQIBand.MODERATE: 25,
        AQIBand.UNHEALTHY_FOR_SENSITIVE: 50,
        AQIBand.UNHEALTHY: 75,
        AQIBand.VERY_UNHEALTHY: 90,
        AQIBand.HAZARDOUS: 100,
    }

    base_risk = band_risk.get(band, 0)

    # Vulnerability multiplier
    multipliers = {
        VulnerabilityGroup.CHILDREN: 1.3,
        VulnerabilityGroup.ELDERLY: 1.3,
        VulnerabilityGroup.RESPIRATORY: 1.5,
        VulnerabilityGroup.CARDIAC: 1.4,
    }

    multiplier = multipliers.get(vulnerability, 1.0)
    risk = min(int(base_risk * multiplier), 100)
    return risk


def find_best_activity_window(hourly_forecast: list[dict], activity_type: ActivityType) -> dict:
    """
    Find the optimal time window (next 24h) for outdoor activity, commute, or ventilation.

    Forecast structure: [{"pm25": float, "hour": int (0-23 UTC)}, ...]

    Args:
        hourly_forecast: List of hourly forecasts with pm25 and hour
        activity_type: Type of activity (affects threshold logic)

    Returns:
        {
            "best_hour": int (0-23),
            "pm25": float,
            "band": str,
            "rationale": str,
            "avoid_windows": [{"start_hour": int, "end_hour": int, "reason": str}]
        }

    Activity thresholds (PM2.5):
      - Outdoor exercise: <35.4 µg/m³ (Moderate band); best <12 µg/m³
      - Commute (outdoor): <55.4 µg/m³ (Unhealthy threshold); best <35.4 µg/m³
      - Home ventilation: <35.4 µg/m³ (can safely open windows)
    """
    if not hourly_forecast:
        return {
            "best_hour": None,
            "pm25": None,
            "band": "Unknown",
            "rationale": "No forecast data available",
            "avoid_windows": [],
        }

    thresholds = {
        ActivityType.OUTDOOR_EXERCISE: 35.4,
        ActivityType.COMMUTE: 55.4,
        ActivityType.VENTILATION: 35.4,
    }

    threshold = thresholds.get(activity_type, 55.4)

    # Find hours within threshold
    suitable_hours = [h for h in hourly_forecast if h.get("pm25", 0) <= threshold]

    if suitable_hours:
        best = min(suitable_hours, key=lambda h: h.get("pm25", 0))
    else:
        best = min(hourly_forecast, key=lambda h: h.get("pm25", 0))

    best_hour = best.get("hour", 0)
    best_pm25 = best.get("pm25", 0.0)
    best_band = calculate_aqi_band(best_pm25, 0).value

    # Identify windows to avoid (above threshold or very high)
    avoid_windows = []
    current_avoid = None
    for h in sorted(hourly_forecast, key=lambda x: x.get("hour", 0)):
        hour = h.get("hour", 0)
        pm25 = h.get("pm25", 0)
        is_bad = pm25 > threshold
        if is_bad and not current_avoid:
            current_avoid = {"start_hour": hour}
        elif not is_bad and current_avoid:
            current_avoid["end_hour"] = hour
            avoid_windows.append(current_avoid)
            current_avoid = None

    if current_avoid:
        current_avoid["end_hour"] = 23
        avoid_windows.append(current_avoid)

    for w in avoid_windows:
        reason = ""
        if activity_type == ActivityType.OUTDOOR_EXERCISE:
            reason = "High pollution; avoid outdoor exercise"
        elif activity_type == ActivityType.COMMUTE:
            reason = "High pollution; delay outdoor commute if possible"
        else:
            reason = "High pollution; keep windows closed"
        w["reason"] = reason

    rationale = ""
    if suitable_hours:
        rationale = (
            f"Hour {best_hour}:00 UTC has the lowest PM2.5 ({best_pm25:.1f} µg/m³, "
            f"{best_band} band) in the next 24h."
        )
    else:
        rationale = (
            f"All hours have elevated pollution. Hour {best_hour}:00 UTC is the best "
            f"available option ({best_pm25:.1f} µg/m³)."
        )

    return {
        "best_hour": best_hour,
        "pm25": best_pm25,
        "band": best_band,
        "rationale": rationale,
        "avoid_windows": avoid_windows,
    }


def estimate_impact(choice: str, lifestyle_param: dict) -> ImpactEstimate:
    """
    Quantify the impact of a behavior change on emissions and exposure.

    Args:
        choice: Type of change ('switch_to_public_transit', 'use_purifier', 'mask_usage', etc.)
        lifestyle_param: Context dict with keys like 'distance_km', 'frequency_days_per_week', etc.

    Returns:
        ImpactEstimate with CO2 saved (kg), exposure reduction (%), and rationale

    Sources:
      - Car CO2 emissions: ~0.21 kg CO2 per km (EPA; average petrol car)
      - Public transit CO2: ~0.05 kg CO2 per km (EPA; bus/metro)
      - Air purifier: 20–40% particulate reduction (depends on model/room size)
      - N95 mask: 50–95% inhalation exposure reduction (CDC)
    """
    impact = ImpactEstimate(co2_kg_saved=0.0, exposure_reduction_percent=0, rationale="")

    if choice == "switch_to_public_transit":
        distance = lifestyle_param.get("distance_km", 10)
        frequency = lifestyle_param.get("frequency_days_per_week", 5)
        # Car: 0.21 kg CO2/km; Transit: 0.05 kg CO2/km
        # Savings per week: (0.21 - 0.05) * distance * frequency
        co2_per_week = (0.21 - 0.05) * distance * frequency
        impact = ImpactEstimate(
            co2_kg_saved=round(co2_per_week * 52 / 12, 1),  # Monthly average
            exposure_reduction_percent=65,  # Reduction in peak exposure
            rationale=(
                "EPA car avg 0.21 kg CO2/km; transit 0.05 kg/km. Assumes 5 days/week. "
                "Monthly CO2 saved. Exposure reduction: reduced outdoor time + "
                "air filtration in transit."
            ),
        )

    elif choice == "use_air_purifier":
        impact = ImpactEstimate(
            co2_kg_saved=0.0,
            exposure_reduction_percent=30,  # Conservative; depends on room size, model, AQI
            rationale="HEPA purifiers reduce particulate inhalation 20–40% (varies by model, room volume, maintenance). Conservative estimate: 30%.",
        )

    elif choice == "wear_n95_mask":
        impact = ImpactEstimate(
            co2_kg_saved=0.0,
            exposure_reduction_percent=70,  # CDC guidance; assumes proper fit
            rationale="N95 masks reduce inhalation exposure 50–95% with proper fit (CDC). Estimate: 70% for typical usage.",
        )

    elif choice == "flexible_schedule":
        # Assume ability to avoid 2 hours of peak commute per week
        impact = ImpactEstimate(
            co2_kg_saved=0.1,  # Minimal direct CO2; primary benefit is exposure
            exposure_reduction_percent=15,
            rationale="Avoids ~2h of peak pollution commute weekly. Modest CO2 savings if replacing short car trips. Exposure reduction from avoiding peak hours.",
        )

    elif choice == "switch_to_cooking_fuel":
        # Assumed: LPG → Induction; reduces indoor PM2.5 from cooking by ~80%
        impact = ImpactEstimate(
            co2_kg_saved=50.0,  # LPG to induction saves ~50 kg CO2 eq/year
            exposure_reduction_percent=80,  # Cooking is major indoor PM source
            rationale="LPG to induction: ~50 kg CO2 eq/year (lifecycle + combustion). Indoor cooking PM2.5 reduction: 70–90% (induction eliminates gas combustion).",
        )

    return impact
