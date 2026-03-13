import argparse
import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

DISTRICTS = [
    "raipur",
    "durg",
    "bilaspur",
    "raigarh",
    "jagdalpur",
    "korba",
    "ambikapur",
    "rajnandgaon",
]

GENDERS = ["male", "female"]
CASTES = ["SC", "ST", "OBC", "GENERAL", "EWS"]

SERVICE_CATEGORY = {
    "income_certificate": "certificate",
    "domicile_certificate": "certificate",
    "sc_st_certificate": "caste_certificate",
    "obc_certificate": "caste_certificate",
    "land_use_information": "land_record",
    "birth_certificate_correction": "civil_registration",
}

CSV_COLUMNS = [
    "application_id",
    "service_type",
    "service_category",
    "age",
    "gender",
    "caste",
    "district",
    "annual_income",
    "average_income_last_3_years",
    "missing_documents_count",
    "missing_fields_count",
    "field_mismatch_count",
    "document_quality_score",
    "age_eligible",
    "income_eligible",
    "district_valid",
    "risk_probability",
    "risk_score",
    "risk_level",
    "main_contributing_factors",
    "rejected",
    "uploaded_documents",
    "application_fields",
]


def load_registry(schema_path: Path) -> Dict:
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)["services"]


def random_dob_from_age(age: int) -> str:
    today = date.today()
    days = random.randint(0, 364)
    dob = date(today.year - age, today.month, min(today.day, 28)) - timedelta(days=days)
    return dob.isoformat()


def generate_demographics(service_name: str) -> Dict:
    if service_name == "birth_certificate_correction":
        age = random.randint(20, 45)
    elif service_name == "land_use_information":
        age = random.randint(25, 75)
    else:
        age = random.randint(18, 85)

    annual_income = random.randint(50000, 900000)
    avg_income_3y = int(annual_income * random.uniform(0.85, 1.15))

    return {
        "age": age,
        "gender": random.choice(GENDERS),
        "caste": random.choice(CASTES),
        "district": random.choice(DISTRICTS),
        "annual_income": annual_income,
        "average_income_last_3_years": avg_income_3y,
    }


def gather_accepted_doc_groups(doc_rules: Dict) -> List[Tuple[str, List[str]]]:
    groups = []
    for key, value in doc_rules.items():
        if key.startswith("accepted_") and isinstance(value, list):
            groups.append((key, value))
    return groups


def generate_documents(doc_rules: Dict, base_quality: float) -> Tuple[List[str], int]:
    uploaded = set()
    missing_count = 0

    mandatory = doc_rules.get("mandatory", [])
    optional = doc_rules.get("optional", [])
    accepted_groups = gather_accepted_doc_groups(doc_rules)

    for doc in mandatory:
        # Some mandatory keys represent a document category resolved by an accepted group.
        if doc in {"income_proof", "obc_proof", "caste_proof", "proof_of_15_years_residence"}:
            continue
        if random.random() < (0.88 * base_quality + 0.08):
            uploaded.add(doc)
        else:
            missing_count += 1

    for _, group_docs in accepted_groups:
        if random.random() < (0.85 * base_quality + 0.1):
            uploaded.add(random.choice(group_docs))
        else:
            missing_count += 1

    for doc in optional:
        if random.random() < 0.55:
            uploaded.add(doc)

    return sorted(uploaded), missing_count


def build_application_fields(service_name: str, demographics: Dict) -> Dict:
    fields = {
        "gender": demographics["gender"],
        "district": demographics["district"],
        "annual_income": demographics["annual_income"],
        "average_income_last_3_years": demographics["average_income_last_3_years"],
        "date_of_birth": random_dob_from_age(demographics["age"]),
    }

    if service_name == "domicile_certificate":
        fields["living_in_state_years"] = random.randint(2, 40)
    if service_name == "land_use_information":
        fields["is_applicant_owner"] = random.choice([0, 1, 1, 1])
    if service_name == "birth_certificate_correction":
        fields["application_number"] = f"BC-{random.randint(100000, 999999)}"
    if service_name in {"sc_st_certificate", "obc_certificate"}:
        fields["category"] = "SC/ST" if service_name == "sc_st_certificate" else "OBC"

    return fields


def eligibility_flags(service_name: str, row: Dict, app_fields: Dict) -> Tuple[int, int, int]:
    district_valid = 1 if random.random() > 0.06 else 0

    if service_name == "income_certificate":
        age_eligible = 1 if row["age"] >= 18 else 0
        income_eligible = 1 if row["annual_income"] <= 300000 else 0
    elif service_name == "domicile_certificate":
        age_eligible = 1
        income_eligible = 1 if app_fields.get("living_in_state_years", 0) >= 15 else 0
    elif service_name == "sc_st_certificate":
        age_eligible = 1
        income_eligible = 1 if row["caste"] in {"SC", "ST"} else 0
    elif service_name == "obc_certificate":
        age_eligible = 1
        income_eligible = 1 if (row["caste"] in {"OBC", "EWS"} and row["annual_income"] <= 800000) else 0
    elif service_name == "land_use_information":
        age_eligible = 1 if row["age"] >= 18 else 0
        income_eligible = app_fields.get("is_applicant_owner", 0)
    else:
        age_eligible = 1
        income_eligible = 1

    return age_eligible, income_eligible, district_valid


def compute_risk_and_label(row: Dict) -> Tuple[float, int, str]:
    contributions = {
        "missing_documents": min(0.46, row["missing_documents_count"] * 0.18),
        "missing_fields": min(0.18, row["missing_fields_count"] * 0.06),
        "field_mismatch": min(0.24, row["field_mismatch_count"] * 0.08),
        "document_quality": max(0.0, (1.0 - row["document_quality_score"]) * 0.35),
        "age_eligible": 0.24 if row["age_eligible"] == 0 else 0.0,
        "income_eligible": 0.22 if row["income_eligible"] == 0 else 0.0,
        "district_valid": 0.16 if row["district_valid"] == 0 else 0.0,
    }

    base = 0.06
    risk_probability = max(0.01, min(0.98, base + sum(contributions.values())))
    rejected = 1 if random.random() < risk_probability else 0

    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    top_factors = [name for name, score in ranked if score > 0][:3]
    if not top_factors:
        top_factors = ["no_major_issue_detected"]

    return risk_probability, rejected, ", ".join(top_factors)


def risk_level(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def build_row(index: int, service_name: str, schema: Dict) -> Dict:
    demographics = generate_demographics(service_name)
    app_fields = build_application_fields(service_name, demographics)

    doc_quality = round(random.uniform(0.55, 0.98), 3)
    uploaded_docs, missing_documents_count = generate_documents(schema["documents"], doc_quality)

    missing_fields_count = random.choices([0, 1, 2, 3], weights=[0.54, 0.28, 0.12, 0.06])[0]
    field_mismatch_count = random.choices([0, 1, 2], weights=[0.62, 0.28, 0.10])[0]

    base_row = {
        "application_id": f"APP-{index:06d}",
        "service_type": service_name,
        "service_category": SERVICE_CATEGORY[service_name],
        **demographics,
        "missing_documents_count": missing_documents_count,
        "missing_fields_count": missing_fields_count,
        "field_mismatch_count": field_mismatch_count,
        "document_quality_score": doc_quality,
    }

    age_eligible, income_eligible, district_valid = eligibility_flags(service_name, base_row, app_fields)
    base_row["age_eligible"] = age_eligible
    base_row["income_eligible"] = income_eligible
    base_row["district_valid"] = district_valid

    probability, rejected, top_factors = compute_risk_and_label(base_row)

    base_row["risk_probability"] = round(probability, 4)
    base_row["risk_score"] = int(round(probability * 100))
    base_row["risk_level"] = risk_level(probability)
    base_row["main_contributing_factors"] = top_factors
    base_row["rejected"] = rejected
    base_row["uploaded_documents"] = json.dumps(uploaded_docs, separators=(",", ":"))
    base_row["application_fields"] = json.dumps(app_fields, separators=(",", ":"))

    return base_row


def generate_dataset(registry: Dict, n_samples: int) -> List[Dict]:
    services = list(registry.keys())
    rows = []
    for i in range(1, n_samples + 1):
        service_name = random.choice(services)
        row = build_row(i, service_name, registry[service_name])
        rows.append(row)
    return rows


def write_csv(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rejection-risk dataset for eDistrict services.")
    parser.add_argument("--schema", default="schemas/service_registry.json", help="Path to service registry schema JSON.")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate.")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    registry = load_registry(Path(args.schema))
    rows = generate_dataset(registry, args.samples)
    write_csv(rows, Path(args.output))

    print(f"Generated {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
