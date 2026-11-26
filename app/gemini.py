"""
Gemini API integration for generating personalized air quality guidance.

Handles prompt construction, API calls, response parsing, and fallback templates.
Input to Gemini is always structured JSON (no free-form user text).
"""

import json
import logging

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoutineAdvice(BaseModel):
    """Structured response from Gemini or fallback template."""
    health_summary: str = Field(..., description="Plain-language health risk summary")
    recommended_routine: str = Field(..., description="Daily routine tailored to current AQI")
    sustainable_alternatives: list[str] = Field(
        ..., description="List of behavior/product alternatives"
    )
    immediate_actions: list[str] = Field(
        ..., description="Actions to take now (next 2–4 hours)"
    )


def get_fallback_routine(aqi_band: str, risk_score: int) -> RoutineAdvice:
    """
    Deterministic fallback routine when Gemini API fails or no key available.

    Args:
        aqi_band: AQI band label ('Good', 'Moderate', etc.)
        risk_score: Aggregate risk score 0–100

    Returns:
        RoutineAdvice with deterministic content
    """
    fallbacks = {
        "Good": {
            "health_summary": "Air quality is good. Outdoor activities are safe for all groups.",
            "routine": "Maintain regular outdoor activities. Keep windows open for natural ventilation.",
            "alternatives": [
                "No restrictions needed; enjoy outdoor activities as normal.",
                "Open windows during the day for fresh air circulation.",
                "Continue outdoor exercise routines.",
            ],
            "actions": [
                "Maintain baseline hydration.",
                "No special precautions needed.",
                "Resume normal outdoor schedules.",
            ],
        },
        "Moderate": {
            "health_summary": f"Air quality is moderate (risk score: {risk_score}/100). Sensitive groups should limit prolonged outdoor exposure.",
            "routine": "Limit outdoor activity to short periods. Vulnerable groups should wear N95 masks for outdoor activity.",
            "alternatives": [
                "Shift outdoor exercise to early morning or late evening (typically lower pollution).",
                "Consider indoor exercise alternatives (yoga, dance, strength training).",
                "Use HEPA air purifiers in bedrooms and common areas.",
                "Switch to induction cooking (if using gas).",
            ],
            "actions": [
                "Check the hourly forecast before planning outdoor activity.",
                "Ensure air purifiers are running in bedrooms.",
                "Have N95 masks readily available.",
            ],
        },
        "Unhealthy for Sensitive Groups": {
            "health_summary": f"Air quality is unhealthy for sensitive groups (risk score: {risk_score}/100). Children, elderly, and those with respiratory/cardiac conditions should avoid outdoor activity.",
            "routine": "Restrict outdoor exposure. Vulnerable groups should stay indoors. Everyone should wear N95 masks if outdoors.",
            "alternatives": [
                "Shift work/school commutes to public transit (reduces outdoor exposure time).",
                "Use video conferencing to reduce commute frequency.",
                "Install HEPA air purifiers in all living spaces.",
                "Cook indoors only during low-pollution hours or switch to induction.",
            ],
            "actions": [
                "Keep vulnerable household members indoors.",
                "Wear N95 masks if outdoor activity is necessary.",
                "Close windows and use air purifiers.",
                "Monitor symptoms; seek medical care if breathing difficulty develops.",
            ],
        },
        "Unhealthy": {
            "health_summary": f"Air quality is unhealthy for all groups (risk score: {risk_score}/100). Everyone should minimize outdoor exposure.",
            "routine": "Avoid outdoor activity. Stay indoors with air purifiers running. Wear N95 masks only if outdoors is unavoidable.",
            "alternatives": [
                "Work/study from home if possible.",
                "Replace commutes with public transit or carpool to minimize outdoor time.",
                "Use online delivery services for errands.",
                "Meal prep indoors at off-peak pollution hours.",
            ],
            "actions": [
                "Keep all household members indoors.",
                "Run HEPA air purifiers continuously.",
                "Wear N95 masks if any outdoor exposure is necessary.",
                "Contact healthcare provider if vulnerable household members show symptoms.",
            ],
        },
        "Very Unhealthy": {
            "health_summary": f"Air quality is very unhealthy for all groups (risk score: {risk_score}/100). Outdoor exposure should be minimized to emergencies only.",
            "routine": "Mandatory indoor protocols. No outdoor activity except emergencies.",
            "alternatives": [
                "Conduct all essential errands during off-peak pollution windows.",
                "Use home delivery exclusively.",
                "Work/study from home exclusively.",
                "Consider temporary relocation if vulnerable household members present.",
            ],
            "actions": [
                "All household members remain indoors.",
                "Seal windows; run HEPA air purifiers continuously.",
                "Medical monitoring for vulnerable groups; seek immediate care if symptoms worsen.",
                "Stock essential supplies to avoid outdoor trips.",
            ],
        },
        "Hazardous": {
            "health_summary": f"Air quality is hazardous for all groups (risk score: {risk_score}/100). No outdoor activity. Emergency protocols.",
            "routine": "Complete lockdown: indoors only. Emergency response mode.",
            "alternatives": [
                "No alternatives; prioritize health and safety.",
                "Seek shelter in well-sealed, filtered indoor spaces.",
                "Consider temporary evacuation if infrastructure allows.",
            ],
            "actions": [
                "All household members must remain indoors.",
                "Seal all windows and doors.",
                "Run all available air purification continuously.",
                "Seek emergency medical care immediately if any respiratory distress.",
            ],
        },
    }

    band_key = aqi_band if aqi_band in fallbacks else "Moderate"
    fb = fallbacks[band_key]

    return RoutineAdvice(
        health_summary=fb["health_summary"],
        recommended_routine=fb["routine"],
        sustainable_alternatives=fb["alternatives"],
        immediate_actions=fb["actions"],
    )


async def generate_routine_with_gemini(
    aqi_band: str,
    risk_score: int,
    lifestyle: dict,
    vulnerable_groups: list[str],
    impact_data: dict,
    api_key: str | None = None,
) -> RoutineAdvice:
    """
    Generate personalized guidance using Google Gemini API.

    Args:
        aqi_band: Current AQI band
        risk_score: Aggregate risk score 0–100
        lifestyle: User profile (commute_mode, outdoor_habits, cooking_fuel, ventilation)
        vulnerable_groups: List of vulnerable household members
        impact_data: Quantified impact of behavior changes
        api_key: Gemini API key; if None, returns fallback

    Returns:
        RoutineAdvice structured output

    Note:
        Input to Gemini is structured JSON (no raw user text).
        Output is parsed from JSON response; falls back if parsing fails.
    """
    if not api_key:
        logger.warning("No Gemini API key; using fallback routine")
        return get_fallback_routine(aqi_band, risk_score)

    prompt_data = {
        "aqi_band": aqi_band,
        "risk_score": risk_score,
        "lifestyle": lifestyle,
        "vulnerable_groups": vulnerable_groups,
        "impact_estimates": impact_data,
    }

    system_message = """You are an expert in air quality health guidance for Indian families during winter smog.
Your role is to generate personalized, actionable daily routines and sustainable alternatives based on:
1. Current AQI band and health risk score
2. Household composition and vulnerabilities
3. Lifestyle factors (commute mode, exercise habits, cooking fuel, home ventilation)
4. Quantified impact of behavior changes (CO2 saved, exposure reduction %)

CRITICAL INSTRUCTIONS:
- Respond ONLY with valid JSON matching this structure:
{
  "health_summary": "Plain-language summary of current risk (1–2 sentences)",
  "recommended_routine": "Tailored daily routine for today's AQI (3–4 sentences)",
  "sustainable_alternatives": ["alternative 1", "alternative 2", ...],
  "immediate_actions": ["action 1", "action 2", ...]
}
- Cite emission factors and health thresholds when quantifying impact.
- If a vulnerable group is present, emphasize their specific risks and mitigations.
- Recommend low-pollution time windows and outdoor alternatives based on the data provided.
- Keep language accessible to a general Indian family (avoid jargon).
"""

    user_message = f"""Given this air quality and lifestyle data, generate a personalized routine:

{json.dumps(prompt_data, indent=2)}

Respond with ONLY valid JSON, no markdown, no explanation."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Google Gemini API call
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": user_message}],
                        }
                    ],
                    "system_instruction": {"parts": [{"text": system_message}]},
                    "generationConfig": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_output_tokens": 1024,
                    },
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    text = parts[0].get("text", "")
                    # Parse JSON from response
                    # Try to extract JSON object even if wrapped in markdown
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0]

                    parsed = json.loads(text.strip())
                    logger.info("Gemini routine generated successfully")
                    return RoutineAdvice(**parsed)

    except Exception as e:
        logger.error(f"Gemini API error: {e}; falling back to template")

    # Fallback if Gemini fails or returns invalid JSON
    return get_fallback_routine(aqi_band, risk_score)
