from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from collections import Counter
from threading import Lock
import math
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from chatbot import process_message, create_empty_state, configure_runtime

MODEL_PATH = Path(__file__).with_name("real_estate_model.joblib")


# Step 1: Load the trained artifact.
artifact = joblib.load(MODEL_PATH)
MODEL = artifact["model"]
FEATURE_COLS = artifact["features"]
MAPS = artifact["context_maps"]
TYPE_CATEGORIES = artifact["type_categories"]
CITY_CATEGORIES = artifact["city_categories"]
PAYMENT_CATEGORIES = artifact["payment_categories"]
FINISHING_CATEGORIES = artifact["finishing_categories"]
VIEW_CATEGORIES = artifact["view_categories"]
AREA_BINS = artifact["area_bins"]
AREA_LABELS = artifact["area_labels"]
BATHROOM_LIMITS = artifact["bathroom_limits"]
MODEL_VERSION = artifact.get("model_version", "9.0.0")
METRICS = artifact.get("metrics", {})
UNCERTAINTY = artifact.get("uncertainty", {})
MODEL_LOADED_AT = datetime.now(timezone.utc).isoformat()

SEA_CITIES = {
    "North Coast",
    "Red Sea",
    "Alexandria",
    "Suez",
    "South Sainai",
    "Matrouh",
    "Demyat",
}
NILE_CITIES = {"Cairo", "Giza", "Luxor", "Aswan"}
MIN_COMPOUND_TYPE_SUPPORT = 3
MIN_OPTION_SUPPORT = 2

PropertyType = Enum(
    "PropertyType",
    {value.replace(" ", "_"): value for value in TYPE_CATEGORIES},
)
City = Enum(
    "City",
    {value.replace(" ", "_").replace("'", ""): value for value in CITY_CATEGORIES},
)
PaymentMethod = Enum(
    "PaymentMethod",
    {value.replace(" ", "_"): value for value in PAYMENT_CATEGORIES},
)
FinishingStatus = Enum(
    "FinishingStatus",
    {value: value for value in FINISHING_CATEGORIES},
)
ViewType = Enum(
    "ViewType",
    {value: value for value in VIEW_CATEGORIES},
)


# Step 2: Create the API.
app = FastAPI(
    title="EstateIQ — Property Intelligence API",
    version=MODEL_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store chatbot conversation state by session.
# These sessions reset whenever the development server restarts.
chat_sessions = {}

# Anonymous, process-local counters. No request bodies, IPs, or session IDs are stored.
# Values reset whenever the server instance restarts or scales down.
usage_analytics = {
    "prediction_count": 0,
    "predicted_price_total_egp": 0.0,
    "governorates": Counter(),
    "property_types": Counter(),
}
analytics_lock = Lock()

FEATURE_LABELS = {
    "area_sqm": "Property area",
    "bedrooms_num": "Bedrooms",
    "bathrooms_per_100sqm": "Bathroom density",
    "bedrooms_per_100sqm": "Bedroom density",
    "bathrooms_per_bedroom": "Bathrooms per bedroom",
    "context_price_encoded": "Local market price",
    "context_price_per_sqm": "Local price per sqm",
    "log_context_price": "Local market level",
    "is_furnished": "Furnished",
    "ready_to_move": "Ready to move",
    "has_pool": "Pool",
    "has_garden": "Garden",
    "has_clubhouse": "Clubhouse",
    "is_standalone": "Standalone property",
    "has_maid_room": "Maid room",
    "prime_location": "Prime location",
}


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.time() - start) * 1000:.1f}"
    return response


# Step 3: Build reusable market helpers.
def get_area_band(area_sqm: float) -> str:
    for index in range(len(AREA_BINS) - 1):
        if AREA_BINS[index] <= area_sqm <= AREA_BINS[index + 1]:
            return AREA_LABELS[index]
    return AREA_LABELS[-1]


def shrink_value(item, parent: float, strength: int) -> float:
    if not item:
        return float(parent)

    median, count = item
    return float((median * count + parent * strength) / (count + strength))


def get_context_price(city: str, property_type: str, area_band: str, compound=None):
    global_price = float(MAPS["global"])

    city_type = MAPS["city_type"].get((city, property_type))
    city_base = shrink_value(city_type, global_price, 30)

    city_area = MAPS["city_type_area"].get((city, property_type, area_band))
    local_base = shrink_value(city_area, city_base, 20)

    if not compound:
        support = int(MAPS["city_type_area"].get((city, property_type, area_band), (0, 0))[1])
        return local_base, {"level": "city_type_area", "support": support}

    compound_type = MAPS["compound_type"].get((compound, property_type))
    compound_base = shrink_value(compound_type, local_base, 20)

    compound_area = MAPS["compound_type_area"].get(
        (compound, property_type, area_band)
    )
    specific_price = shrink_value(compound_area, compound_base, 12)

    area_support = int(
        MAPS.get("compound_type_area_support", {}).get(
            (compound, property_type, area_band),
            0,
        )
    )
    type_support = int(
        MAPS.get("compound_type_support", {}).get((compound, property_type), 0)
    )

    level = "compound_type_area" if area_support else "compound_type"
    return specific_price, {
        "level": level,
        "support": area_support or type_support,
    }


def get_bathroom_rule(property_type: str, area_sqm: float):
    matches = [
        row
        for row in BATHROOM_LIMITS
        if row["property_type"] == property_type
        and row["area_min"] <= area_sqm <= row["area_max"]
    ]
    if matches:
        return matches[0]

    same_type = [
        row for row in BATHROOM_LIMITS if row["property_type"] == property_type
    ]
    if not same_type:
        return None

    return min(
        same_type,
        key=lambda row: min(
            abs(area_sqm - row["area_min"]),
            abs(area_sqm - row["area_max"]),
        ),
    )


def is_known_compound(name: Optional[str]) -> bool:
    return bool(name and name in MAPS.get("compound_city", {}))


def get_support_score(city, property_type, area_band, compound, view_type):
    city_type = int(
        MAPS.get("city_type_support", {}).get(
            (city, property_type),
            MAPS["city_type"].get((city, property_type), (0, 0))[1],
        )
    )
    city_area = int(
        MAPS.get("city_type_area_support", {}).get(
            (city, property_type, area_band),
            MAPS["city_type_area"].get((city, property_type, area_band), (0, 0))[1],
        )
    )

    compound_type = 0
    compound_area = 0

    if compound and is_known_compound(compound):
        compound_type = int(
            MAPS.get("compound_type_support", {}).get(
                (compound, property_type),
                0,
            )
        )
        compound_area = int(
            MAPS.get("compound_type_area_support", {}).get(
                (compound, property_type, area_band),
                0,
            )
        )

    score = min(1.0, math.log1p(max(city_area, 1)) / math.log(80))

    if compound:
        compound_score = min(
            1.0,
            math.log1p(max(compound_type, 1)) / math.log(50),
        )
        score = 0.40 * score + 0.60 * compound_score

        if compound_type < 10:
            score = min(score, 0.55)
        if compound_area < 3:
            score = min(score, 0.68)

    if compound_area >= 3:
        score = min(1.0, score + 0.10)

    if view_type != "none":
        view_support = int(
            MAPS.get("city_view_support", {}).get((city, view_type), 0)
        )
        view_factor = min(
            1.0,
            0.65 + 0.35 * math.log1p(max(view_support, 1)) / math.log(50),
        )
        score *= view_factor

    return max(0.15, min(1.0, score)), {
        "city_type": city_type,
        "city_type_area": city_area,
        "compound_type": compound_type,
        "compound_type_area": compound_area,
    }


# Step 4: Build dynamic input constraints.
def option(value, status="available", support=0, reason=""):
    return {
        "value": value,
        "status": status,
        "support": int(support),
        "reason": reason,
    }


def build_constraints(
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    area_sqm: Optional[float] = None,
    compound_name: Optional[str] = None,
    finishing_status: Optional[str] = None,
):
    compound = (compound_name or "").strip() or None

    property_options = []
    for value in TYPE_CATEGORIES:
        support = 0
        status = "available"
        reason = ""

        if compound and is_known_compound(compound):
            support = int(
                MAPS.get("compound_type_support", {}).get((compound, value), 0)
            )
            if support < MIN_COMPOUND_TYPE_SUPPORT:
                status = "insufficient_data"
                reason = "Not enough training examples for this property type in the selected compound."
        elif city:
            support = int(
                MAPS.get("city_type_support", {}).get(
                    (city, value),
                    MAPS["city_type"].get((city, value), (0, 0))[1],
                )
            )
            if support < MIN_OPTION_SUPPORT:
                status = "insufficient_data"
                reason = "Not enough training examples for this property type in the selected city."

        property_options.append(option(value, status, support, reason))

    view_options = []
    for value in VIEW_CATEGORIES:
        if value == "none":
            view_options.append(option(value))
            continue

        status = "available"
        support = 0
        reason = ""

        if city and value == "sea" and city not in SEA_CITIES:
            status = "not_applicable"
            reason = "Sea view is not geographically applicable to the selected city."
        elif city and value == "nile" and city not in NILE_CITIES:
            status = "not_applicable"
            reason = "Nile view is not geographically applicable to the selected city."
        elif compound and is_known_compound(compound):
            if property_type:
                support = int(
                    MAPS.get("compound_type_view_support", {}).get(
                        (compound, property_type, value),
                        0,
                    )
                )
            else:
                support = int(
                    MAPS.get("compound_view_support", {}).get((compound, value), 0)
                )

            if support < MIN_OPTION_SUPPORT:
                status = "insufficient_data"
                reason = "This view is not sufficiently represented for the selected compound and property type."
        elif city:
            if property_type:
                support = int(
                    MAPS.get("city_type_view_support", {}).get(
                        (city, property_type, value),
                        0,
                    )
                )
            else:
                support = int(
                    MAPS.get("city_view_support", {}).get((city, value), 0)
                )

            if support < MIN_OPTION_SUPPORT:
                status = "insufficient_data"
                reason = "This view is not sufficiently represented in the selected market segment."

        view_options.append(option(value, status, support, reason))

    finishing_options = []
    for value in FINISHING_CATEGORIES:
        status = "available"
        support = 0
        reason = ""

        if property_type == "Land" and value != "unspecified":
            status = "not_applicable"
            reason = "Finishing status does not apply to land."
        elif value != "unspecified" and property_type:
            if compound and is_known_compound(compound):
                support = int(
                    MAPS.get("compound_type_finish_support", {}).get(
                        (compound, property_type, value),
                        0,
                    )
                )
            elif city:
                support = int(
                    MAPS.get("city_type_finish_support", {}).get(
                        (city, property_type, value),
                        0,
                    )
                )

            if (compound or city) and support < MIN_OPTION_SUPPORT:
                status = "insufficient_data"
                reason = "This finishing status is not sufficiently represented in the selected market segment."

        finishing_options.append(option(value, status, support, reason))

    bathroom_rule = None
    if property_type and area_sqm is not None:
        bathroom_rule = get_bathroom_rule(property_type, area_sqm)

    feature_rules = {
        "is_furnished": {
            "allowed": property_type != "Land"
            and finishing_status not in {"semi_finished", "core_shell"},
            "reason": "Furnished is unavailable for land, semi-finished, or core-shell properties.",
        },
        "ready_to_move": {
            "allowed": property_type != "Land"
            and finishing_status not in {"semi_finished", "core_shell"},
            "reason": "Ready to move is unavailable for land, semi-finished, or core-shell properties.",
        },
        "is_standalone": {
            "allowed": property_type == "Villa",
            "reason": "Standalone is only available for villas.",
        },
        "has_maid_room": {
            "allowed": property_type != "Land",
            "reason": "Maid room does not apply to land.",
        },
    }

    return {
        "property_types": property_options,
        "view_types": view_options,
        "finishing_statuses": finishing_options,
        "bathrooms": bathroom_rule,
        "land_forces_zero_rooms": property_type == "Land",
        "feature_rules": feature_rules,
    }


# Step 5: Define request and response schemas.
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1, max_length=100)


class PropertyInput(BaseModel):
    area_sqm: float = Field(..., gt=10, le=5000)
    bedrooms: int = Field(..., ge=0, le=10)
    bathrooms: int = Field(..., ge=0, le=10)
    property_type: PropertyType
    city: City
    payment_method: PaymentMethod = PaymentMethod.Cash
    compound_name: Optional[str] = None
    finishing_status: FinishingStatus = FinishingStatus.unspecified
    view_type: ViewType = ViewType.none
    is_furnished: bool = False
    ready_to_move: bool = False
    has_pool: bool = False
    has_garden: bool = False
    has_clubhouse: bool = False
    is_standalone: bool = False
    has_maid_room: bool = False
    prime_location: bool = False

    model_config = ConfigDict(use_enum_values=True, validate_default=True)

    @model_validator(mode="after")
    def validate_input(self):
        compound = (self.compound_name or "").strip() or None

        if self.property_type == "Land":
            if self.bedrooms or self.bathrooms:
                raise ValueError("Land must use 0 bedrooms and 0 bathrooms.")
            if (
                self.finishing_status != "unspecified"
                or self.is_furnished
                or self.ready_to_move
                or self.is_standalone
                or self.has_maid_room
            ):
                raise ValueError("Land cannot use residential property options.")

        if self.finishing_status in {"semi_finished", "core_shell"}:
            if self.is_furnished or self.ready_to_move:
                raise ValueError(
                    "Semi-finished and core-shell properties cannot be furnished or ready to move."
                )

        if self.is_standalone and self.property_type != "Villa":
            raise ValueError("Standalone is only available for villas.")

        bathroom_rule = get_bathroom_rule(self.property_type, self.area_sqm)
        if bathroom_rule and self.bathrooms > int(bathroom_rule["recommended_max"]):
            raise ValueError(
                f"The supported bathroom maximum is {int(bathroom_rule['recommended_max'])} for this property type and area."
            )

        constraints = build_constraints(
            city=self.city,
            property_type=self.property_type,
            area_sqm=self.area_sqm,
            compound_name=compound,
            finishing_status=self.finishing_status,
        )

        property_status = next(
            item
            for item in constraints["property_types"]
            if item["value"] == self.property_type
        )
        if property_status["status"] != "available":
            raise ValueError(property_status["reason"])

        view_status = next(
            item
            for item in constraints["view_types"]
            if item["value"] == self.view_type
        )
        if view_status["status"] != "available":
            raise ValueError(view_status["reason"])

        finishing_status = next(
            item
            for item in constraints["finishing_statuses"]
            if item["value"] == self.finishing_status
        )
        if finishing_status["status"] == "not_applicable":
            raise ValueError(finishing_status["reason"])

        if compound and is_known_compound(compound):
            expected_city = MAPS["compound_city"].get(compound)
            if expected_city and expected_city != self.city:
                raise ValueError(
                    f"{compound} is represented under {expected_city}, not {self.city}."
                )

        return self


class PredictionOutput(BaseModel):
    predicted_price_egp: float
    predicted_price_formatted: str
    currency: str = "EGP"
    model_version: str
    market_context_egp: float
    estimate_low_egp: float
    estimate_high_egp: float
    confidence_score: float
    confidence_label: str
    evidence_level: str
    evidence_support: int
    support_counts: dict
    range_note: str
    explanation_method: str
    top_positive_contributors: list[dict]
    top_negative_contributors: list[dict]


class MetadataOutput(BaseModel):
    property_types: list[str]
    cities: list[str]
    payment_methods: list[str]
    finishing_statuses: list[str]
    view_types: list[str]
    n_known_compounds: int
    model_metrics: dict
    model_version: str


# Step 6: Build the model input row.
def build_feature_row(payload: PropertyInput):
    row = {column: 0 for column in FEATURE_COLS}
    area_band = get_area_band(payload.area_sqm)
    compound = (payload.compound_name or "").strip() or None

    context_price, evidence = get_context_price(
        payload.city,
        payload.property_type,
        area_band,
        compound,
    )

    row.update(
        {
            "area_sqm": payload.area_sqm,
            "bedrooms_num": payload.bedrooms,
            "bathrooms_per_100sqm": payload.bathrooms / payload.area_sqm * 100,
            "bedrooms_per_100sqm": payload.bedrooms / payload.area_sqm * 100,
            "bathrooms_per_bedroom": payload.bathrooms / max(payload.bedrooms, 1),
            "context_price_encoded": context_price,
            "context_price_per_sqm": context_price / max(payload.area_sqm, 1),
            "log_context_price": math.log1p(context_price),
        }
    )

    binary_features = [
        "is_furnished",
        "ready_to_move",
        "has_pool",
        "has_garden",
        "has_clubhouse",
        "is_standalone",
        "has_maid_room",
        "prime_location",
    ]
    for feature in binary_features:
        if feature in row:
            row[feature] = int(bool(getattr(payload, feature)))

    categorical_values = {
        "type": payload.property_type,
        "payment_method": payload.payment_method,
        "city": payload.city,
        "finishing_status": payload.finishing_status,
        "view_type": payload.view_type,
        "type_view": f"{payload.property_type}__{payload.view_type}",
        "type_finish": f"{payload.property_type}__{payload.finishing_status}",
        "city_type_interaction": f"{payload.city}__{payload.property_type}",
    }

    for prefix, value in categorical_values.items():
        column = f"{prefix}_{value}"
        if column in row:
            row[column] = 1

    frame = pd.DataFrame([row])[FEATURE_COLS]
    return frame, context_price, evidence, area_band


def humanize_feature(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    prefixes = {
        "city_type_interaction_": "Location and property type",
        "type_finish_": "Property type and finishing",
        "type_view_": "Property type and view",
        "finishing_status_": "Finishing status",
        "payment_method_": "Payment method",
        "property_type_": "Property type",
        "type_": "Property type",
        "view_type_": "View",
        "city_": "Governorate",
    }
    for prefix, label in prefixes.items():
        if name.startswith(prefix):
            value = name[len(prefix):].replace("__", " / ").replace("_", " ")
            return f"{label}: {value.title()}"
    return name.replace("_", " ").title()


def explain_prediction(features: pd.DataFrame):
    """Return native XGBoost TreeSHAP contributions without the SHAP package."""
    try:
        booster = MODEL.get_booster()
        matrix = xgb.DMatrix(features, feature_names=list(features.columns))
        values = booster.predict(matrix, pred_contribs=True)[0]
        # The final value is the bias term. Model output is log-residual, so
        # exp(contribution)-1 is an intuitive approximate percentage effect.
        contributors = []
        for name, value in zip(FEATURE_COLS, values[:-1]):
            value = float(value)
            if not math.isfinite(value) or abs(value) < 1e-7:
                continue
            pct = math.expm1(max(-3.0, min(3.0, value))) * 100
            contributors.append({
                "feature": name,
                "label": humanize_feature(name),
                "contribution_log": round(value, 6),
                "approx_effect_percent": round(pct, 1),
            })
        positive = sorted(
            (item for item in contributors if item["contribution_log"] > 0),
            key=lambda item: item["contribution_log"], reverse=True,
        )[:4]
        negative = sorted(
            (item for item in contributors if item["contribution_log"] < 0),
            key=lambda item: item["contribution_log"],
        )[:4]
        return positive, negative, "xgboost_pred_contribs"
    except Exception:
        # Prediction remains available if an older serialized estimator does
        # not expose a compatible booster API.
        return [], [], "unavailable"


def record_prediction(payload: PropertyInput, prediction: float):
    with analytics_lock:
        usage_analytics["prediction_count"] += 1
        usage_analytics["predicted_price_total_egp"] += prediction
        usage_analytics["governorates"][payload.city] += 1
        usage_analytics["property_types"][payload.property_type] += 1


# Step 7: Expose API endpoints.
@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(Path(__file__).with_name("frontend.html"))


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_version": MODEL_VERSION,
        "loaded_at": MODEL_LOADED_AT,
        "n_features": len(FEATURE_COLS),
        "metrics": METRICS,
    }


@app.get("/metadata", response_model=MetadataOutput)
def metadata():
    return MetadataOutput(
        property_types=TYPE_CATEGORIES,
        cities=CITY_CATEGORIES,
        payment_methods=PAYMENT_CATEGORIES,
        finishing_statuses=FINISHING_CATEGORIES,
        view_types=VIEW_CATEGORIES,
        n_known_compounds=len(MAPS.get("compound_city", {})),
        model_metrics=METRICS,
        model_version=MODEL_VERSION,
    )


@app.get("/constraints")
def constraints(
    city: Optional[str] = None,
    property_type: Optional[str] = None,
    area_sqm: Optional[float] = None,
    compound_name: Optional[str] = None,
    finishing_status: Optional[str] = None,
):
    if city and city not in CITY_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown city.")
    if property_type and property_type not in TYPE_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown property type.")
    if finishing_status and finishing_status not in FINISHING_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown finishing status.")

    return build_constraints(
        city=city,
        property_type=property_type,
        area_sqm=area_sqm,
        compound_name=compound_name,
        finishing_status=finishing_status,
    )


@app.get("/compounds")
def compounds(q: str = "", limit: int = 15, city: Optional[str] = None):
    names = sorted(MAPS.get("compound_city", {}))

    if city:
        names = [name for name in names if MAPS["compound_city"].get(name) == city]

    query = q.strip().lower()
    if query:
        names = [name for name in names if query in name.lower()]

    return {
        "query": q,
        "matches": names[: max(1, min(limit, 50))],
    }


@app.get("/market-context")
def market_context(
    city: str,
    property_type: str,
    area_sqm: float,
    compound_name: Optional[str] = None,
):
    if city not in CITY_CATEGORIES or property_type not in TYPE_CATEGORIES:
        raise HTTPException(status_code=422, detail="Unknown city or property type.")

    area_band = get_area_band(area_sqm)
    compound = (compound_name or "").strip() or None
    market_price, evidence = get_context_price(
        city,
        property_type,
        area_band,
        compound,
    )
    score, support_counts = get_support_score(
        city,
        property_type,
        area_band,
        compound,
        "none",
    )

    return {
        "market_context_egp": round(market_price, 2),
        "area_band": area_band,
        "evidence": evidence,
        "support_counts": support_counts,
        "confidence_score": round(score, 3),
    }


@app.get("/analytics")
def analytics():
    with analytics_lock:
        count = usage_analytics["prediction_count"]
        top_city = usage_analytics["governorates"].most_common(1)
        top_type = usage_analytics["property_types"].most_common(1)
        return {
            "prediction_count": count,
            "average_predicted_price_egp": round(
                usage_analytics["predicted_price_total_egp"] / count, 2
            ) if count else 0.0,
            "most_requested_governorate": top_city[0][0] if top_city else None,
            "most_requested_property_type": top_type[0][0] if top_type else None,
            "governorate_counts": dict(usage_analytics["governorates"]),
            "property_type_counts": dict(usage_analytics["property_types"]),
            "storage": "anonymous_in_memory",
            "reset_note": "Counters reset when this server instance restarts or scales down.",
        }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PropertyInput):
    try:
        features, context_price, evidence, area_band = build_feature_row(payload)

        residual = float(MODEL.predict(features)[0])
        prediction = context_price * math.exp(residual)
        prediction = float(
            np.clip(
                prediction,
                context_price * 0.45,
                context_price * 2.20,
            )
        )

        compound = (payload.compound_name or "").strip() or None
        score, support_counts = get_support_score(
            payload.city,
            payload.property_type,
            area_band,
            compound,
            payload.view_type,
        )

        base_error = float(UNCERTAINTY.get("p80_absolute_pct_error", 0.45))
        width = min(0.60, max(0.16, base_error * (1.25 - score * 0.45)))
        estimate_low = max(0, prediction * (1 - width))
        estimate_high = prediction * (1 + width)
        positive, negative, explanation_method = explain_prediction(features)

        if score >= 0.78:
            confidence_label = "High"
        elif score >= 0.50:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        record_prediction(payload, prediction)
        return PredictionOutput(
            predicted_price_egp=round(prediction, 2),
            predicted_price_formatted=f"{prediction:,.0f} EGP",
            model_version=MODEL_VERSION,
            market_context_egp=round(context_price, 2),
            estimate_low_egp=round(estimate_low, 2),
            estimate_high_egp=round(estimate_high, 2),
            confidence_score=round(score, 3),
            confidence_label=confidence_label,
            evidence_level=evidence["level"],
            evidence_support=int(evidence["support"]),
            support_counts=support_counts,
            range_note=(
                "Indicative valuation range based on model error and available market "
                "evidence; it is not a guaranteed statistical coverage interval."
            ),
            explanation_method=explanation_method,
            top_positive_contributors=positive,
            top_negative_contributors=negative,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/chat")
def chat(request: ChatRequest):
    # Avoid a loopback HTTP call on serverless deployments such as Vercel.
    # Both handlers reuse the exact same validation and prediction code as the API.
    configure_runtime(
        constraint_handler=lambda params: build_constraints(**params),
        prediction_handler=lambda data: predict(PropertyInput(**data)).model_dump(),
    )
    session_id = request.session_id.strip() or "default"

    if session_id not in chat_sessions:
        chat_sessions[session_id] = create_empty_state()

    state = chat_sessions[session_id]
    reply = process_message(request.message, state)

    return {
        "reply": reply,
        "session_id": session_id,
        "state": state,
    }


@app.delete("/chat/{session_id}")
def reset_chat_session(session_id: str):
    chat_sessions[session_id] = create_empty_state()

    return {
        "status": "reset",
        "session_id": session_id,
        "state": chat_sessions[session_id],
    }


# Step 8: Run the development server.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
