"""
FastAPI application for Urban Air Quality Assistant.

Endpoints:
  GET /api/health - Health check
  POST /api/location - Geocode city to lat/lon
  POST /api/forecast - Fetch air quality forecast
  POST /api/advice - Generate personalized guidance (main endpoint)
  GET / - Serve static HTML UI
"""

import logging
import os
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from app import aqi, core, gemini

# Local runs read GEMINI_API_KEY from .env; on Cloud Run it comes from Secret Manager.
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# httpx logs every request URL at INFO; keep it quiet so upstream URLs stay out of the logs.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(
    title="Urban Air Quality Assistant",
    description="AI-powered air quality guidance for Indian families",
    version="1.0.0",
)

# CORS configuration (not *)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' https://api.open-meteo.com https://geocoding-api.open-meteo.com"
    )
    return response


# Pydantic models
class LocationRequest(BaseModel):
    """Request to geocode a city."""

    city: str = Field(..., min_length=1, max_length=100)

    @field_validator("city")
    @classmethod
    def city_no_injection(cls, v):
        """Basic injection defense: city is data, not code."""
        if any(char in v for char in [";", "--", "/*", "*/"]):
            raise ValueError("Invalid characters in city name")
        return v.strip()


class LocationResponse(BaseModel):
    """Response with geocoded location."""

    name: str
    latitude: float
    longitude: float
    timezone: str | None = None


class ForecastRequest(BaseModel):
    """Request for air quality forecast."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class AirQualityResponse(BaseModel):
    """Current air quality snapshot."""

    pm25: float
    pm10: float
    us_aqi: float
    band: str
    timestamp: str


class AdviceRequest(BaseModel):
    """Request for personalized air quality guidance."""

    city: str = Field(..., min_length=1, max_length=100)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    vulnerable_groups: list[str] = Field(
        default_factory=list
    )  # 'children', 'elderly', 'respiratory_conditions', 'cardiac_conditions'
    commute_mode: str = Field(default="car", pattern="^(car|public_transit|cycle|walk)$")
    outdoor_habits: str = Field(default="moderate", pattern="^(low|moderate|high)$")
    cooking_fuel: str = Field(default="lpg", pattern="^(lpg|induction|wood|charcoal)$")
    home_ventilation: str = Field(default="basic", pattern="^(basic|air_purifier|natural)$")

    @field_validator("city")
    @classmethod
    def city_no_injection(cls, v):
        """Basic injection defense."""
        if any(char in v for char in [";", "--", "/*", "*/"]):
            raise ValueError("Invalid characters in city name")
        return v.strip()


class AdviceResponse(BaseModel):
    """Personalized guidance response."""

    location: str
    aqi_band: str
    risk_score: int
    best_activity_window: dict
    health_summary: str
    recommended_routine: str
    sustainable_alternatives: list[str]
    immediate_actions: list[str]
    generated_at: str


# Health check
@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "urban-air-assistant",
    }


# Geocoding endpoint
@app.post("/api/location", response_model=LocationResponse, tags=["location"])
async def geocode_location(request: LocationRequest):
    """Geocode a city name to latitude and longitude."""
    result = await aqi.geocode_city(request.city)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"City '{request.city}' not found. Please check spelling or try a larger city.",
        )
    return LocationResponse(
        name=result.name,
        latitude=result.latitude,
        longitude=result.longitude,
        timezone=result.timezone,
    )


# Air quality forecast endpoint
@app.post("/api/forecast", response_model=AirQualityResponse, tags=["forecast"])
async def get_forecast(request: ForecastRequest):
    """Fetch current air quality and forecast."""
    result = await aqi.fetch_air_quality(request.latitude, request.longitude)
    if not result:
        raise HTTPException(
            status_code=503,
            detail="Unable to fetch air quality data. Please try again later.",
        )

    current, hourly = result
    band = core.calculate_aqi_band(current.pm25, current.pm10).value

    return AirQualityResponse(
        pm25=round(current.pm25, 1),
        pm10=round(current.pm10, 1),
        us_aqi=round(current.us_aqi, 1),
        band=band,
        timestamp=current.timestamp,
    )


# Main advice endpoint
@app.post("/api/advice", response_model=AdviceResponse, tags=["advice"])
async def get_advice(request: AdviceRequest):
    """Generate personalized air quality guidance."""
    # Geocode if needed
    if not request.latitude or not request.longitude:
        geo_result = await aqi.geocode_city(request.city)
        if not geo_result:
            raise HTTPException(status_code=404, detail=f"City '{request.city}' not found")
        request.latitude = geo_result.latitude
        request.longitude = geo_result.longitude

    # Fetch AQ data
    aq_result = await aqi.fetch_air_quality(request.latitude, request.longitude)
    if not aq_result:
        raise HTTPException(status_code=503, detail="Unable to fetch air quality data")

    current, hourly = aq_result

    # Calculate AQI band and risk
    band = core.calculate_aqi_band(current.pm25, current.pm10)
    risk_scores = []
    for group in request.vulnerable_groups:
        try:
            vuln = core.VulnerabilityGroup(group)
            risk = core.assess_health_risk(band, vuln)
            risk_scores.append(risk)
        except ValueError:
            logger.warning(f"Unknown vulnerability group: {group}")

    aggregate_risk = max(risk_scores) if risk_scores else 0

    # Find best activity window
    activity = core.ActivityType.OUTDOOR_EXERCISE
    if request.commute_mode != "walk":
        activity = core.ActivityType.COMMUTE
    best_window = core.find_best_activity_window([h.model_dump() for h in hourly], activity)

    # Estimate impact of potential changes
    impact_data = {}
    if request.commute_mode == "car":
        impact = core.estimate_impact(
            "switch_to_public_transit",
            {"distance_km": 15, "frequency_days_per_week": 5},
        )
        impact_data["transit_switch"] = {
            "co2_kg_saved": impact.co2_kg_saved,
            "exposure_reduction_percent": impact.exposure_reduction_percent,
            "rationale": impact.rationale,
        }

    if request.cooking_fuel != "induction":
        impact = core.estimate_impact("switch_to_cooking_fuel", {"frequency_days_per_week": 7})
        impact_data["cooking_switch"] = {
            "co2_kg_saved": impact.co2_kg_saved,
            "exposure_reduction_percent": impact.exposure_reduction_percent,
            "rationale": impact.rationale,
        }

    if request.home_ventilation != "air_purifier":
        impact = core.estimate_impact("use_air_purifier", {})
        impact_data["purifier"] = {
            "co2_kg_saved": impact.co2_kg_saved,
            "exposure_reduction_percent": impact.exposure_reduction_percent,
            "rationale": impact.rationale,
        }

    # Generate routine with Gemini
    lifestyle = {
        "commute_mode": request.commute_mode,
        "outdoor_habits": request.outdoor_habits,
        "cooking_fuel": request.cooking_fuel,
        "home_ventilation": request.home_ventilation,
    }

    gemini_key = os.getenv("GEMINI_API_KEY")
    routine = await gemini.generate_routine_with_gemini(
        aqi_band=band.value,
        risk_score=aggregate_risk,
        lifestyle=lifestyle,
        vulnerable_groups=request.vulnerable_groups,
        impact_data=impact_data,
        api_key=gemini_key,
        activity_window=best_window,
    )

    return AdviceResponse(
        location=request.city,
        aqi_band=band.value,
        risk_score=aggregate_risk,
        best_activity_window=best_window,
        health_summary=routine.health_summary,
        recommended_routine=routine.recommended_routine,
        sustainable_alternatives=routine.sustainable_alternatives,
        immediate_actions=routine.immediate_actions,
        generated_at=datetime.now().isoformat(),
    )


# Serve static HTML UI
@app.get("/", response_class=HTMLResponse, tags=["ui"])
async def serve_ui():
    """Serve the main HTML UI."""
    try:
        with open("app/static/index.html") as f:
            return f.read()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="UI not found") from e


# Catch-all 404
@app.get("/{full_path:path}")
async def not_found(full_path: str):
    """Catch undefined routes."""
    raise HTTPException(status_code=404, detail=f"Path /{full_path} not found")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
