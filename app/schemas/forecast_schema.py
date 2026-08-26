"""Pydantic schemas for POST /forecast."""
from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    districtId: int = Field(
        ...,
        ge=0,
        le=25,
        description=(
            "District identifier — the model's internal rdhs_id (0-based, "
            "alphabetical). "
            "Mapping: 0=Ampara, 1=Anuradhapura, 2=Badulla, 3=Batticaloa, "
            "4=Colombo, 5=Galle, 6=Gampaha, 7=Hambantota, 8=Jaffna, "
            "9=Kalmunai, 10=Kalutara, 11=Kandy, 12=Kegalle, 13=Kilinochchi, "
            "14=Kurunegala, 15=Mannar, 16=Matale, 17=Matara, 18=Monaragala, "
            "19=Mullaitivu, 20=Nuwara Eliya, 21=Polonnaruwa, 22=Puttalam, "
            "23=Ratnapura, 24=Trincomalee, 25=Vavuniya."
        ),
    )


class WeekForecast(BaseModel):
    weekAhead: int          # 1, 2, 3, or 4
    predictedCases: float   # real (unscaled) predicted case count


class ForecastResponse(BaseModel):
    districtId: int
    forecast: list[WeekForecast]
