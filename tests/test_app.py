"""
Comprehensive test suite for Urban Air Quality Assistant.

Tests cover:
- Decision logic (core.py)
- AQI data fetching and caching (aqi.py)
- Gemini integration with fallback (gemini.py)
- API endpoints (main.py)

Run: pytest --cov=app --cov-report=html
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import aqi, core, gemini, main

# =============================================================================
# TESTS FOR CORE DECISION LOGIC (core.py)
# =============================================================================


class TestAQIBandCalculation:
    """Tests for AQI band classification."""

    def test_calculate_aqi_band_good(self):
        """PM2.5 in Good range (0–12 µg/m³)."""
        band = core.calculate_aqi_band(10.0, 20.0)
        assert band == core.AQIBand.GOOD

    def test_calculate_aqi_band_moderate(self):
        """PM2.5 in Moderate range (12.1–35.4 µg/m³)."""
        band = core.calculate_aqi_band(25.0, 50.0)
        assert band == core.AQIBand.MODERATE

    def test_calculate_aqi_band_sensitive(self):
        """PM2.5 in Unhealthy for Sensitive Groups range (35.5–55.4 µg/m³)."""
        band = core.calculate_aqi_band(45.0, 90.0)
        assert band == core.AQIBand.UNHEALTHY_FOR_SENSITIVE

    def test_calculate_aqi_band_unhealthy(self):
        """PM2.5 in Unhealthy range (55.5–150.4 µg/m³)."""
        band = core.calculate_aqi_band(100.0, 200.0)
        assert band == core.AQIBand.UNHEALTHY

    def test_calculate_aqi_band_very_unhealthy(self):
        """PM2.5 in Very Unhealthy range (150.5–250.4 µg/m³)."""
        band = core.calculate_aqi_band(200.0, 300.0)
        assert band == core.AQIBand.VERY_UNHEALTHY

    def test_calculate_aqi_band_hazardous(self):
        """PM2.5 in Hazardous range (>250.5 µg/m³)."""
        band = core.calculate_aqi_band(300.0, 400.0)
        assert band == core.AQIBand.HAZARDOUS

    def test_calculate_aqi_band_boundary_good_to_moderate(self):
        """Test boundary between Good and Moderate."""
        assert core.calculate_aqi_band(12.0, 0) == core.AQIBand.GOOD
        assert core.calculate_aqi_band(12.1, 0) == core.AQIBand.MODERATE


class TestHealthRiskAssessment:
    """Tests for health risk scoring."""

    def test_assess_health_risk_children_good_band(self):
        """Children in Good band should have minimal risk."""
        risk = core.assess_health_risk(core.AQIBand.GOOD, core.VulnerabilityGroup.CHILDREN)
        assert risk == 0

    def test_assess_health_risk_children_moderate_band(self):
        """Children in Moderate band should have elevated risk."""
        risk = core.assess_health_risk(core.AQIBand.MODERATE, core.VulnerabilityGroup.CHILDREN)
        assert risk > 25  # Multiplier 1.3 on base 25

    def test_assess_health_risk_respiratory_condition(self):
        """Respiratory condition at Moderate should have higher multiplier."""
        risk_base = core.assess_health_risk(core.AQIBand.MODERATE, core.VulnerabilityGroup.CHILDREN)
        risk_resp = core.assess_health_risk(
            core.AQIBand.MODERATE, core.VulnerabilityGroup.RESPIRATORY
        )
        assert risk_resp > risk_base

    def test_assess_health_risk_maximum_capped(self):
        """Risk should be capped at 100."""
        risk = core.assess_health_risk(core.AQIBand.HAZARDOUS, core.VulnerabilityGroup.RESPIRATORY)
        assert risk <= 100

    def test_assess_health_risk_elderly(self):
        """Elderly should have elevated risk similar to children."""
        risk_child = core.assess_health_risk(
            core.AQIBand.MODERATE, core.VulnerabilityGroup.CHILDREN
        )
        risk_elderly = core.assess_health_risk(
            core.AQIBand.MODERATE, core.VulnerabilityGroup.ELDERLY
        )
        assert abs(risk_child - risk_elderly) <= 5  # Roughly equal (both 1.3x)


class TestBestActivityWindow:
    """Tests for optimal time window identification."""

    def test_find_best_activity_window_outdoor_exercise(self):
        """Find best hour for outdoor exercise (threshold <35.4 µg/m³)."""
        forecast = [
            {"hour": 0, "pm25": 50.0},
            {"hour": 6, "pm25": 20.0},
            {"hour": 12, "pm25": 60.0},
        ]
        result = core.find_best_activity_window(forecast, core.ActivityType.OUTDOOR_EXERCISE)
        assert result["best_hour"] == 6
        assert result["pm25"] == 20.0

    def test_find_best_activity_window_commute(self):
        """Find best hour for commute (threshold <55.4 µg/m³)."""
        forecast = [
            {"hour": 0, "pm25": 100.0},
            {"hour": 8, "pm25": 50.0},
            {"hour": 18, "pm25": 70.0},
        ]
        result = core.find_best_activity_window(forecast, core.ActivityType.COMMUTE)
        assert result["best_hour"] == 8

    def test_find_best_activity_window_ventilation(self):
        """Find best hour for home ventilation (threshold <35.4 µg/m³)."""
        forecast = [
            {"hour": 2, "pm25": 30.0},
            {"hour": 14, "pm25": 25.0},
        ]
        result = core.find_best_activity_window(forecast, core.ActivityType.VENTILATION)
        assert result["pm25"] <= 35.4

    def test_find_best_activity_window_no_suitable_hours(self):
        """When all hours have high pollution, pick the best available."""
        forecast = [
            {"hour": 0, "pm25": 200.0},
            {"hour": 6, "pm25": 180.0},
        ]
        result = core.find_best_activity_window(forecast, core.ActivityType.OUTDOOR_EXERCISE)
        assert result["best_hour"] == 6

    def test_find_best_activity_window_empty_forecast(self):
        """Empty forecast should return graceful response."""
        result = core.find_best_activity_window([], core.ActivityType.OUTDOOR_EXERCISE)
        assert result["best_hour"] is None


class TestImpactEstimation:
    """Tests for behavior change impact quantification."""

    def test_estimate_impact_switch_to_transit(self):
        """Switching to public transit should reduce CO2 and exposure."""
        impact = core.estimate_impact(
            "switch_to_public_transit",
            {"distance_km": 10, "frequency_days_per_week": 5},
        )
        assert impact.co2_kg_saved > 0
        assert impact.exposure_reduction_percent > 0
        assert "EPA" in impact.rationale

    def test_estimate_impact_use_purifier(self):
        """Air purifier should reduce exposure but not CO2."""
        impact = core.estimate_impact("use_air_purifier", {})
        assert impact.co2_kg_saved == 0.0
        assert impact.exposure_reduction_percent > 0

    def test_estimate_impact_wear_mask(self):
        """N95 mask should provide high exposure reduction."""
        impact = core.estimate_impact("wear_n95_mask", {})
        assert impact.co2_kg_saved == 0.0
        assert impact.exposure_reduction_percent >= 50

    def test_estimate_impact_switch_cooking_fuel(self):
        """Switching from LPG to induction should save CO2."""
        impact = core.estimate_impact("switch_to_cooking_fuel", {"frequency_days_per_week": 7})
        assert impact.co2_kg_saved > 0
        assert impact.exposure_reduction_percent > 0

    def test_estimate_impact_includes_rationale(self):
        """All impacts should have a cited rationale."""
        impact = core.estimate_impact("switch_to_public_transit", {"distance_km": 15})
        assert len(impact.rationale) > 10

    @pytest.mark.asyncio
    @patch("app.aqi.httpx.AsyncClient")
    async def test_geocode_city_network_error(self, mock_client_class):
        """Handle network error during geocoding."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        aqi.clear_cache()
        result = await aqi.geocode_city("Delhi")
        assert result is None


# =============================================================================
# TESTS FOR AQI DATA FETCHING (aqi.py)
# =============================================================================


class TestGeocoding:
    """Tests for city geocoding."""

    @pytest.mark.asyncio
    @patch("app.aqi.httpx.AsyncClient")
    async def test_geocode_city_success(self, mock_client_class):
        """Successfully geocode a city."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "name": "Delhi",
                    "latitude": 28.7041,
                    "longitude": 77.1025,
                    "country": "India",
                    "admin1": "Delhi",
                    "timezone": "Asia/Kolkata",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        # Create a mock client that works as async context manager
        mock_client = MagicMock()
        mock_client.get = mock_get

        # Setup AsyncClient to return a proper async context manager
        async def async_context_manager():
            return mock_client

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        aqi.clear_cache()
        result = await aqi.geocode_city("Delhi")
        assert result.name == "Delhi"
        assert result.latitude == 28.7041
        assert result.longitude == 77.1025

    @pytest.mark.asyncio
    @patch("app.aqi.httpx.AsyncClient")
    async def test_geocode_city_not_found(self, mock_client_class):
        """Handle city not found."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        # Create a mock client that works as async context manager
        mock_client = MagicMock()
        mock_client.get = mock_get

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        aqi.clear_cache()
        result = await aqi.geocode_city("UnknownCity123")
        assert result is None

    @pytest.mark.asyncio
    async def test_geocode_city_cache(self):
        """Geocoding results should be cached."""
        with patch("app.aqi.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "results": [
                    {
                        "name": "Bengaluru",
                        "latitude": 12.9716,
                        "longitude": 77.5946,
                        "country": "India",
                    }
                ]
            }
            mock_response.raise_for_status = MagicMock()

            call_count = 0

            async def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return mock_response

            # Create a mock client that works as async context manager
            mock_client = MagicMock()
            mock_client.get = mock_get

            mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

            aqi.clear_cache()
            result1 = await aqi.geocode_city("Bengaluru")
            result2 = await aqi.geocode_city("Bengaluru")

            assert result1 == result2
            # Should only call API once (cached second time)
            assert call_count == 1

    @pytest.mark.asyncio
    @patch("app.aqi.httpx.AsyncClient")
    async def test_fetch_air_quality_success(self, mock_client_class):
        """Successfully fetch air quality data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "current": {
                "pm2_5": 25.0,
                "pm10": 50.0,
                "us_aqi": 70.0,
                "time": "2024-11-26T12:00Z",
            },
            "hourly": {
                "time": ["2024-11-26T12:00Z", "2024-11-26T13:00Z"],
                "pm2_5": [25.0, 30.0],
                "pm10": [50.0, 55.0],
            },
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client = MagicMock()
        mock_client.get = mock_get

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        aqi.clear_cache()
        result = await aqi.fetch_air_quality(28.7, 77.1)
        assert result is not None
        current, hourly = result
        assert current.pm25 == 25.0
        assert len(hourly) == 2

    @pytest.mark.asyncio
    @patch("app.aqi.httpx.AsyncClient")
    async def test_fetch_air_quality_network_error(self, mock_client_class):
        """Handle network error during air quality fetch."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Network error"))

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        aqi.clear_cache()
        result = await aqi.fetch_air_quality(28.7, 77.1)
        assert result is None


# =============================================================================
# TESTS FOR GEMINI INTEGRATION (gemini.py)
# =============================================================================


class TestGeminiIntegration:
    """Tests for Gemini API integration."""

    def test_get_fallback_routine_good_band(self):
        """Fallback routine for Good AQI band."""
        routine = gemini.get_fallback_routine("Good", 0)
        assert "good" in routine.health_summary.lower()
        assert len(routine.sustainable_alternatives) > 0
        assert len(routine.immediate_actions) > 0

    def test_get_fallback_routine_unhealthy_band(self):
        """Fallback routine for Unhealthy band."""
        routine = gemini.get_fallback_routine("Unhealthy", 75)
        assert "unhealthy" in routine.health_summary.lower()
        assert "indoors" in routine.recommended_routine.lower()

    def test_get_fallback_routine_includes_risk_score(self):
        """Fallback routine should reference risk score."""
        routine = gemini.get_fallback_routine("Moderate", 50)
        assert "50" in routine.health_summary

    @pytest.mark.asyncio
    async def test_generate_routine_no_key_uses_fallback(self):
        """Without API key, should use fallback."""
        routine = await gemini.generate_routine_with_gemini(
            aqi_band="Moderate",
            risk_score=25,
            lifestyle={"commute_mode": "car"},
            vulnerable_groups=[],
            impact_data={},
            api_key=None,
        )
        assert len(routine.health_summary) > 0
        assert isinstance(routine.sustainable_alternatives, list)

    @pytest.mark.asyncio
    @patch("app.gemini.httpx.AsyncClient")
    async def test_generate_routine_with_gemini_success(self, mock_client_class):
        """Successfully generate routine with Gemini."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "health_summary": "Air quality is moderate.",
                                        "recommended_routine": "Limit outdoor activity.",
                                        "sustainable_alternatives": ["Use public transit"],
                                        "immediate_actions": ["Check forecast"],
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_post(*args, **kwargs):
            return mock_response

        # Create a mock client that works as async context manager
        mock_client = MagicMock()
        mock_client.post = mock_post

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        routine = await gemini.generate_routine_with_gemini(
            aqi_band="Moderate",
            risk_score=25,
            lifestyle={},
            vulnerable_groups=[],
            impact_data={},
            api_key="test-key",
        )
        assert routine.health_summary == "Air quality is moderate."

    @pytest.mark.asyncio
    @patch("app.gemini.httpx.AsyncClient")
    async def test_generate_routine_gemini_failure_fallback(self, mock_client_class):
        """Gemini failure should fall back to template."""
        # Create a mock client that works as async context manager
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("API error"))

        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        routine = await gemini.generate_routine_with_gemini(
            aqi_band="Good",
            risk_score=0,
            lifestyle={},
            vulnerable_groups=[],
            impact_data={},
            api_key="test-key",
        )
        assert len(routine.health_summary) > 0  # Should have fallback


# =============================================================================
# TESTS FOR API ENDPOINTS (main.py)
# =============================================================================


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """FastAPI test client."""
        from starlette.testclient import TestClient

        return TestClient(main.app)

    def test_health_check(self, client):
        """Health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    @patch("app.aqi.geocode_city")
    def test_location_endpoint_success(self, mock_geocode, client):
        """Geocode location endpoint."""
        mock_geocode.return_value = aqi.GeocodingResult(
            name="Delhi",
            latitude=28.7041,
            longitude=77.1025,
            country="India",
        )

        response = client.post("/api/location", json={"city": "Delhi"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Delhi"
        assert data["latitude"] == 28.7041

    @patch("app.aqi.geocode_city")
    def test_location_endpoint_not_found(self, mock_geocode, client):
        """Location not found."""
        mock_geocode.return_value = None

        response = client.post("/api/location", json={"city": "UnknownCity"})
        assert response.status_code == 404

    def test_location_endpoint_injection_defense(self, client):
        """Location endpoint should reject SQL injection attempts."""
        response = client.post("/api/location", json={"city": "Delhi'; DROP TABLE--"})
        assert response.status_code == 422  # Validation error

    @patch("app.aqi.fetch_air_quality")
    def test_forecast_endpoint(self, mock_fetch, client):
        """Forecast endpoint."""
        current = aqi.AirQualityData(
            pm25=25.0, pm10=50.0, us_aqi=70.0, timestamp=datetime.now().isoformat()
        )
        hourly = [aqi.HourlyForecast(hour=0, pm25=20.0, pm10=40.0)]
        mock_fetch.return_value = (current, hourly)

        response = client.post("/api/forecast", json={"latitude": 28.7041, "longitude": 77.1025})
        assert response.status_code == 200
        data = response.json()
        assert data["pm25"] == 25.0
        assert "band" in data

    def test_advice_endpoint_input_validation(self, client):
        """Advice endpoint should validate inputs."""
        response = client.post(
            "/api/advice",
            json={
                "city": "Delhi",
                "latitude": 200.0,  # Invalid latitude
                "longitude": 77.1025,
            },
        )
        assert response.status_code == 422

    @patch("app.aqi.geocode_city")
    @patch("app.aqi.fetch_air_quality")
    @patch("app.gemini.generate_routine_with_gemini")
    def test_advice_endpoint_success(self, mock_routine, mock_fetch, mock_geocode, client):
        """Advice endpoint integration test."""
        mock_geocode.return_value = aqi.GeocodingResult(
            name="Delhi",
            latitude=28.7041,
            longitude=77.1025,
            country="India",
        )

        current = aqi.AirQualityData(
            pm25=35.0, pm10=60.0, us_aqi=100.0, timestamp=datetime.now().isoformat()
        )
        hourly = [
            aqi.HourlyForecast(hour=0, pm25=40.0, pm10=70.0),
            aqi.HourlyForecast(hour=6, pm25=30.0, pm10=55.0),
        ]
        mock_fetch.return_value = (current, hourly)

        mock_routine.return_value = gemini.RoutineAdvice(
            health_summary="Moderate pollution.",
            recommended_routine="Limit outdoor activity.",
            sustainable_alternatives=["Use transit"],
            immediate_actions=["Check forecast"],
        )

        response = client.post(
            "/api/advice",
            json={
                "city": "Delhi",
                "vulnerable_groups": ["children"],
                "commute_mode": "car",
                "outdoor_habits": "moderate",
                "cooking_fuel": "lpg",
                "home_ventilation": "basic",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "Delhi"
        assert "aqi_band" in data
        assert "risk_score" in data
        assert "health_summary" in data

    def test_ui_endpoint(self, client):
        """UI endpoint serves HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=app", "--cov-report=html"])
