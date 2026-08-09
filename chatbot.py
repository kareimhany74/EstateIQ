import os
import re
import requests

PORT = os.getenv("PORT", "8000")
API_URL = os.getenv("ESTATEIQ_INTERNAL_API", f"http://127.0.0.1:{PORT}")
_constraint_handler = None
_prediction_handler = None


def configure_runtime(constraint_handler=None, prediction_handler=None):
    """Use in-process API handlers when chatbot and FastAPI share a runtime."""
    global _constraint_handler, _prediction_handler
    _constraint_handler = constraint_handler
    _prediction_handler = prediction_handler


def create_empty_state():
    return {
        "area_sqm": None,
        "bedrooms": None,
        "bathrooms": None,
        "property_type": None,
        "city": None,
        "payment_method": None,
        "compound_name": None,
        "finishing_status": None,
        "view_type": None,
        "is_furnished": False,
        "has_pool": False,
        "has_garden": False,
        "has_clubhouse": False,
        "is_standalone": False,
        "has_maid_room": False,
        "ready_to_move": False,
        "prime_location": False
    }


required_fields = [
    "area_sqm",
    "bedrooms",
    "bathrooms",
    "property_type",
    "city",
    "payment_method"
]


questions = {
    "area_sqm": "What is the property area in square meters?",
    "bedrooms": "How many bedrooms does the property have?",
    "bathrooms": "How many bathrooms does the property have?",
    "property_type": "What is the property type?",
    "city": "Which governorate is the property located in?",
    "payment_method": "What is the payment method: Cash or Installments?"
}


property_type_map = {
    "apartment": "Apartment",
    "villa": "Villa",
    "duplex": "Duplex",
    "penthouse": "Penthouse",
    "chalet": "Chalet",
    "townhouse": "Townhouse",
    "town house": "Townhouse",
    "twin house": "Twin House",
    "twinhouse": "Twin House",
    "ivilla": "iVilla",
    "i villa": "iVilla",
    "hotel apartment": "Hotel Apartment",
    "land": "Land"
}


governorate_map = {
    "cairo": "Cairo",
    "giza": "Giza",
    "alexandria": "Alexandria",
    "qalyubia": "Qalyubia",
    "qalubia": "Qalyubia",
    "north coast": "North Coast",
    "sahel": "North Coast",
    "red sea": "Red Sea",
    "matrouh": "Matrouh",
    "suez": "Suez",
    "sharqia": "Sharqia",
    "aswan": "Aswan",
    "asyut": "Asyut",
    "luxor": "Luxor",
    "demyat": "Demyat",
    "damietta": "Demyat",
    "al daqahlya": "Al Daqahlya",
    "dakahlia": "Al Daqahlya",
    "south sainai": "South Sainai",
    "south sinai": "South Sainai"
}


compound_map = {
    "madinaty": "Madinaty",
    "marassi": "Marassi",
    "mountain view": "Mountain View",
    "hyde park": "Hyde Park",
    "palm hills": "Palm Hills",
    "sodic": "SODIC",
    "o west": "O West"
}


def extract_property_type(message):
    message = message.lower()
    for value in sorted(property_type_map, key=len, reverse=True):
        if value in message:
            return property_type_map[value]
    return None


def extract_governorate(message):
    message = message.lower()
    for value in sorted(governorate_map, key=len, reverse=True):
        if value in message:
            return governorate_map[value]
    return None


def extract_compound(message):
    message = message.lower()
    for compound in sorted(compound_map, key=len, reverse=True):
        if compound in message:
            return compound_map[compound]
    return None


def extract_payment_method(message):
    message = message.lower()

    if "cash" in message:
        return "Cash"

    if (
        "installment" in message
        or "installments" in message
        or "installment plan" in message
    ):
        return "Installments"

    return None


def extract_advanced_features(message):
    message = message.lower()
    extracted = {}

    if "fully finished" in message or "super lux" in message:
        extracted["finishing_status"] = "fully_finished"
    elif "semi finished" in message or "semi-finished" in message:
        extracted["finishing_status"] = "semi_finished"
    elif (
        "core shell" in message
        or "core & shell" in message
        or "core and shell" in message
    ):
        extracted["finishing_status"] = "core_shell"

    if (
        "no view" in message
        or "without view" in message
        or "no specific view" in message
    ):
        extracted["view_type"] = "none"
    elif "sea view" in message:
        extracted["view_type"] = "sea"
    elif "nile view" in message:
        extracted["view_type"] = "nile"
    elif "lagoon" in message or "water view" in message:
        extracted["view_type"] = "lagoon_water"

    if (
        "unfurnished" in message
        or "not furnished" in message
        or "without furniture" in message
    ):
        extracted["is_furnished"] = False
    elif "furnished" in message:
        extracted["is_furnished"] = True

    if "no pool" in message or "without pool" in message:
        extracted["has_pool"] = False
    elif "pool" in message:
        extracted["has_pool"] = True

    if "no garden" in message or "without garden" in message:
        extracted["has_garden"] = False
    elif "garden" in message:
        extracted["has_garden"] = True

    if "no clubhouse" in message or "without clubhouse" in message:
        extracted["has_clubhouse"] = False
    elif "clubhouse" in message or "club house" in message:
        extracted["has_clubhouse"] = True

    if "no maid room" in message or "without maid room" in message:
        extracted["has_maid_room"] = False
    elif "maid room" in message:
        extracted["has_maid_room"] = True

    if "not ready to move" in message or "not ready" in message:
        extracted["ready_to_move"] = False
    elif "ready to move" in message:
        extracted["ready_to_move"] = True

    if "not prime location" in message:
        extracted["prime_location"] = False
    elif "prime location" in message:
        extracted["prime_location"] = True

    if "not standalone" in message or "not stand alone" in message:
        extracted["is_standalone"] = False
    elif "standalone" in message or "stand alone" in message:
        extracted["is_standalone"] = True

    return extracted


def extract_information(message):
    message_lower = message.lower()
    extracted = {}

    area_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(sqm|m2|m²|meter|meters)',
        message_lower
    )
    if area_match:
        area_value = float(area_match.group(1))
        extracted["area_sqm"] = int(area_value) if area_value.is_integer() else area_value

    bedroom_match = re.search(
        r'(\d+)\s*(bedroom|bedrooms|bed|beds)',
        message_lower
    )
    if bedroom_match:
        extracted["bedrooms"] = int(bedroom_match.group(1))

    bathroom_match = re.search(
        r'(\d+)\s*(bathroom|bathrooms|bath|baths)',
        message_lower
    )
    if bathroom_match:
        extracted["bathrooms"] = int(bathroom_match.group(1))

    property_type = extract_property_type(message)
    if property_type:
        extracted["property_type"] = property_type

    governorate = extract_governorate(message)
    if governorate:
        extracted["city"] = governorate

    payment_method = extract_payment_method(message)
    if payment_method:
        extracted["payment_method"] = payment_method

    compound = extract_compound(message)
    if compound:
        extracted["compound_name"] = compound

    extracted.update(extract_advanced_features(message))
    return extracted


def update_state(state, extracted_data):
    for key, value in extracted_data.items():
        if key in state:
            state[key] = value
    return state


def reset_conversation_state(state):
    state.clear()
    state.update(create_empty_state())
    return state


def get_missing_fields(state):
    return [field for field in required_fields if state[field] is None]


def get_next_question(state):
    missing_fields = get_missing_fields(state)
    if not missing_fields:
        return None
    return questions[missing_fields[0]]


def extract_contextual_answer(message, state):
    """Interpret short answers using the next missing required field."""
    message_clean = message.strip().lower()

    # Only apply contextual parsing to short, direct replies.
    if len(message_clean.split()) > 4:
        return {}

    missing_fields = get_missing_fields(state)
    if not missing_fields:
        return {}

    expected_field = missing_fields[0]
    extracted = {}

    if expected_field == "area_sqm":
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", message_clean)
        if match:
            value = float(match.group(1))
            extracted["area_sqm"] = int(value) if value.is_integer() else value

    elif expected_field == "bedrooms":
        match = re.fullmatch(r"\s*(\d+)\s*", message_clean)
        if match:
            extracted["bedrooms"] = int(match.group(1))

    elif expected_field == "bathrooms":
        match = re.fullmatch(r"\s*(\d+)\s*", message_clean)
        if match:
            extracted["bathrooms"] = int(match.group(1))

    elif expected_field == "property_type":
        property_type = extract_property_type(message)
        if property_type:
            extracted["property_type"] = property_type

    elif expected_field == "city":
        governorate = extract_governorate(message)
        if governorate:
            extracted["city"] = governorate

    elif expected_field == "payment_method":
        payment_method = extract_payment_method(message)
        if payment_method:
            extracted["payment_method"] = payment_method

    return extracted


def get_dynamic_constraints(state):
    params = {
        "property_type": state["property_type"],
        "area_sqm": state["area_sqm"],
        "city": state["city"],
        "compound_name": state["compound_name"] or ""
    }

    if _constraint_handler is not None:
        try:
            return _constraint_handler(params)
        except Exception as exc:
            return {"error": str(exc).replace("Value error, ", "")}

    try:
        response = requests.get(
            f"{API_URL}/constraints",
            params=params,
            timeout=15
        )
    except requests.RequestException:
        return {"error": "Unable to connect to the EstateIQ API."}

    if response.status_code == 200:
        return response.json()

    try:
        error_data = response.json()
        if "detail" in error_data:
            return {"error": str(error_data["detail"])}
    except ValueError:
        pass

    return {"error": "Unable to load property constraints."}


def validate_dynamic_constraints(state):
    constraints = get_dynamic_constraints(state)

    if "error" in constraints:
        return False, constraints["error"]

    max_bathrooms = constraints.get("max_bathrooms")
    if max_bathrooms is not None and state["bathrooms"] > max_bathrooms:
        return False, (
            f"The supported bathroom maximum is "
            f"{max_bathrooms} for this property."
        )

    allowed_property_types = constraints.get("allowed_property_types")
    if allowed_property_types and state["property_type"] not in allowed_property_types:
        return False, (
            f"{state['property_type']} is not sufficiently supported "
            f"for the selected market."
        )

    allowed_views = constraints.get("allowed_views")
    if allowed_views:
        current_view = state["view_type"] or "none"
        if current_view not in allowed_views:
            return False, (
                f"{current_view.replace('_', ' ').title()} is not "
                f"supported for the selected location."
            )

    standalone_allowed = constraints.get("standalone_allowed")
    if state["is_standalone"] and standalone_allowed is False:
        return False, (
            "Standalone configuration is not supported "
            "for this property type."
        )

    return True, None


def validate_state(state):
    is_valid, message = validate_dynamic_constraints(state)
    if not is_valid:
        return False, message

    if state["property_type"] == "Land":
        if state["bedrooms"] != 0 or state["bathrooms"] != 0:
            return False, "Land properties cannot have bedrooms or bathrooms."

    if state["finishing_status"] in ["semi_finished", "core_shell"]:
        if state["is_furnished"]:
            return False, (
                "Semi-finished or core-shell properties "
                "cannot be marked as furnished."
            )
        if state["ready_to_move"]:
            return False, (
                "Semi-finished or core-shell properties "
                "cannot be marked as ready to move."
            )

    return True, None


def get_prediction(state):
    payload = {
        "area_sqm": state["area_sqm"],
        "bedrooms": state["bedrooms"],
        "bathrooms": state["bathrooms"],
        "property_type": state["property_type"],
        "city": state["city"],
        "payment_method": state["payment_method"],
        "compound_name": state["compound_name"],
        "finishing_status": state["finishing_status"] or "unspecified",
        "view_type": state["view_type"] or "none",
        "is_furnished": state["is_furnished"],
        "has_pool": state["has_pool"],
        "has_garden": state["has_garden"],
        "has_clubhouse": state["has_clubhouse"],
        "is_standalone": state["is_standalone"],
        "has_maid_room": state["has_maid_room"],
        "ready_to_move": state["ready_to_move"],
        "prime_location": state["prime_location"]
    }

    if _prediction_handler is not None:
        try:
            return _prediction_handler(payload)
        except Exception as exc:
            detail = getattr(exc, "detail", str(exc))
            if isinstance(detail, list) and detail:
                detail = detail[0].get("msg", "Invalid property information.")
            return {"error": str(detail).replace("Value error, ", "")}

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=30
        )
    except requests.RequestException:
        return {"error": "Unable to connect to the EstateIQ API."}

    if response.status_code == 200:
        return response.json()

    try:
        error_data = response.json()
        if "detail" in error_data:
            detail = error_data["detail"]

            if isinstance(detail, list) and len(detail) > 0:
                message = detail[0].get("msg", "Invalid property information.")
                return {"error": message.replace("Value error, ", "")}

            if isinstance(detail, str):
                return {"error": detail.replace("Value error, ", "")}
    except ValueError:
        pass

    return {"error": "Unable to process the property information."}


def clean_user_message(message):
    message = re.sub(
        r'\bselected city\b',
        'selected governorate',
        message,
        flags=re.IGNORECASE
    )
    message = re.sub(
        r'\bcity\b',
        'governorate',
        message,
        flags=re.IGNORECASE
    )
    return message


def suggest_alternatives(state, validation_message):
    suggestions = []
    message = validation_message.lower()

    if "sea view" in message:
        suggestions.append("Choose 'No specific view'")
        suggestions.append(
            "Choose a coastal governorate such as North Coast, Alexandria, or Red Sea"
        )

    if "nile view" in message:
        suggestions.append("Choose 'No specific view'")
        suggestions.append("Choose Cairo or Giza")

    if "bathroom maximum" in message:
        suggestions.append("Reduce the number of bathrooms")

    if "standalone" in message:
        suggestions.append("Change the property type to Villa")
        suggestions.append("Disable the standalone option")

    if "furnished" in message:
        suggestions.append("Use Fully Finished")
        suggestions.append("Disable the furnished option")

    if "ready to move" in message:
        suggestions.append("Use Fully Finished")
        suggestions.append("Disable Ready to Move")

    return suggestions


def format_validation_response(message, suggestions):
    response = clean_user_message(message)

    if suggestions:
        response += "\n\nAvailable alternatives:\n"
        for suggestion in suggestions:
            response += f"- {suggestion}\n"

    return response.strip()


def format_prediction_response(prediction):
    price = prediction["predicted_price_formatted"]
    low = f'{prediction["estimate_low_egp"]:,.0f} EGP'
    high = f'{prediction["estimate_high_egp"]:,.0f} EGP'
    confidence = prediction["confidence_label"]
    support = prediction["evidence_support"]

    positive = prediction.get("top_positive_contributors", [])[:3]
    negative = prediction.get("top_negative_contributors", [])[:3]
    factors = []
    if positive:
        factors.append("Factors lifting the estimate: " + ", ".join(item["label"] for item in positive))
    if negative:
        factors.append("Factors lowering the estimate: " + ", ".join(item["label"] for item in negative))
    explanation = ("\n" + "\n".join(factors)) if factors else ""

    return (
        f"Estimated Market Value: {price}\n"
        f"Indicative Valuation Range: {low} - {high}\n"
        f"Valuation Confidence: {confidence}\n"
        f"Comparable Market Evidence: {support} listings"
        f"{explanation}\n"
        "Range is indicative and does not guarantee statistical coverage."
    )


def format_current_details(state):
    lines = [
        "Current Property Details:",
        "",
        f"Area: {state['area_sqm'] if state['area_sqm'] is not None else 'Not provided'} sqm",
        f"Bedrooms: {state['bedrooms'] if state['bedrooms'] is not None else 'Not provided'}",
        f"Bathrooms: {state['bathrooms'] if state['bathrooms'] is not None else 'Not provided'}",
        f"Property Type: {state['property_type'] or 'Not provided'}",
        f"Governorate: {state['city'] or 'Not provided'}",
        f"Payment Method: {state['payment_method'] or 'Not provided'}",
        f"Compound: {state['compound_name'] or 'Not specified'}",
        f"Finishing: {state['finishing_status'] or 'Not specified'}",
        f"View: {state['view_type'] or 'None'}",
        f"Furnished: {'Yes' if state['is_furnished'] else 'No'}",
        f"Pool: {'Yes' if state['has_pool'] else 'No'}",
        f"Garden: {'Yes' if state['has_garden'] else 'No'}",
        f"Clubhouse: {'Yes' if state['has_clubhouse'] else 'No'}",
        f"Standalone: {'Yes' if state['is_standalone'] else 'No'}",
        f"Maid Room: {'Yes' if state['has_maid_room'] else 'No'}",
        f"Ready to Move: {'Yes' if state['ready_to_move'] else 'No'}",
        f"Prime Location: {'Yes' if state['prime_location'] else 'No'}"
    ]
    return "\n".join(lines)


def process_message(message, state):
    message_lower = message.lower().strip()

    reset_commands = [
        "reset",
        "start over",
        "new property",
        "reset property",
        "clear"
    ]

    if message_lower in reset_commands:
        reset_conversation_state(state)
        return (
            "Property details have been reset.\n"
            "What is the property area in square meters?"
        )

    detail_commands = [
        "show details",
        "show current details",
        "current details",
        "current property",
        "show property"
    ]

    if message_lower in detail_commands:
        return format_current_details(state)

    extracted_data = extract_information(message)

    # If the user gives a short reply such as "140", interpret it
    # according to the question the chatbot is currently asking.
    contextual_data = extract_contextual_answer(message, state)
    extracted_data.update(contextual_data)

    update_state(state, extracted_data)

    next_question = get_next_question(state)
    if next_question:
        return next_question

    is_valid, validation_message = validate_state(state)
    if not is_valid:
        validation_message = clean_user_message(validation_message)
        suggestions = suggest_alternatives(state, validation_message)
        return format_validation_response(validation_message, suggestions)

    prediction = get_prediction(state)
    if "error" in prediction:
        return clean_user_message(prediction["error"])

    return format_prediction_response(prediction)
