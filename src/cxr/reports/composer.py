"""Deterministic Findings / Impression draft from the 14-class score vector with patient context.

Follows ACR / RSNA structured thoracic reporting standards:
1. Patient Demographics & Clinical Indication
2. Compartmentalized Findings:
   - LUNGS & AIRWAYS
   - PLEURAL SPACES & COSTOPHRENIC SULCI
   - CARDIOMEDIASTINUM & VASCULATURE
   - OSSEOUS STRUCTURES & SOFT TISSUES
3. Prioritized Impression with Differential Diagnoses
4. Evidence-Based Clinical Recommendations (IDSA/ATS, Fleischner 2017, ACC/AHA)
5. Plain-Language Patient Summary (Translating findings for patients and caregivers)
"""

from __future__ import annotations

from typing import Any
from cxr.constants import FINDING_LANGUAGE, PATHOLOGIES

COMPARTMENT_MAP = {
    "lungs": ["Pneumonia", "Consolidation", "Infiltration", "Atelectasis", "Nodule", "Mass", "Emphysema", "Fibrosis"],
    "pleura": ["Effusion", "Pneumothorax", "Pleural_Thickening"],
    "cardio": ["Cardiomegaly", "Edema"],
    "osseous": ["Hernia"],
}


def _severity(score: float) -> str:
    if score >= 0.85:
        return "severe"
    if score >= 0.65:
        return "moderate"
    return "mild"


def _build_patient_summary(
    positives: list[tuple[str, float]],
    age: Any,
    sex: str,
    indication: str,
) -> str:
    """Generates an empathetic, plain-language explanation written at an accessible reading level."""
    if not positives:
        return (
            "What Your X-Ray Shows:\n"
            "• Good news: Your chest X-ray looks clear! There are no signs of pneumonia, fluid buildup, "
            "collapsed lung, or enlarged heart.\n\n"
            "What This Means For You:\n"
            "• Your lungs and heart appear normal on this picture. If you are still feeling unwell, cough, or pain, "
            "your doctor will help determine if other common conditions (like a viral cold, muscle strain, or asthma) "
            "might be causing your symptoms.\n\n"
            "Questions You Can Ask Your Doctor:\n"
            "• 'Since my chest X-ray is clear, what else could be causing my symptoms?'\n"
            "• 'Are there any medicines or treatments to help me feel better while I recover?'"
        )

    findings_list = []
    for name, score in positives:
        disp = name.replace("_", " ")
        if name == "Pneumonia":
            findings_list.append("• Pneumonia: There is an area of inflammation and fluid/mucus in your lung, which is typical for a lung infection.")
        elif name == "Consolidation":
            findings_list.append("• Consolidation: A patch of your lung is temporarily filled with fluid or inflammatory cells rather than clear air.")
        elif name == "Infiltration":
            findings_list.append("• Infiltrate: A cloudy area in your lung tissue suggesting infection or inflammation.")
        elif name == "Cardiomegaly":
            findings_list.append("• Enlarged Heart: The silhouette of your heart appears larger than usual on the image, which can happen with high blood pressure or heart conditions.")
        elif name == "Edema":
            findings_list.append("• Fluid in Lungs (Pulmonary Edema): Extra fluid is collecting around the blood vessels in your lungs, which often causes shortness of breath.")
        elif name == "Effusion":
            findings_list.append("• Pleural Fluid: A pocket of fluid has collected in the space between your lung and your chest wall.")
        elif name == "Pneumothorax":
            findings_list.append("• Air Leak / Collapsed Lung (Pneumothorax): Air has leaked into the space surrounding your lung, preventing it from fully expanding.")
        elif name in ("Nodule", "Mass"):
            findings_list.append(f"• Spot / Shadow ({disp}): A small distinct spot was seen in your lung tissue that requires a clearer 3D picture (CT scan) to inspect closely.")
        elif name == "Atelectasis":
            findings_list.append("• Collapsed Air Sacs (Atelectasis): A small section of lung tissue isn't completely inflated with air, often from shallow breathing or coughing.")
        elif name == "Emphysema":
            findings_list.append("• Emphysema: Signs of stretched lung air spaces, commonly associated with smoking or COPD.")
        elif name == "Fibrosis":
            findings_list.append("• Lung Scarring (Fibrosis): Subtle scar-like markings seen in the lung tissue.")
        elif name == "Pleural_Thickening":
            findings_list.append("• Pleural Thickening: A thickening of the lining around the lungs, often from past inflammation.")
        elif name == "Hernia":
            findings_list.append("• Hernia: Part of the stomach or digestive tract is protruding slightly upward toward the chest.")
        else:
            findings_list.append(f"• {disp}: An area of changed lung tissue was detected.")

    findings_text = "\n".join(findings_list)

    next_steps = []
    if any(p in ["Pneumonia", "Consolidation", "Infiltration"] for p, _ in positives):
        next_steps.append("• Infection Treatment: Your doctor may prescribe antibiotics or breathing treatments and recommend plenty of fluids and rest.")
        next_steps.append("• Healing Follow-Up: Doctors often recommend a follow-up X-ray in about a month to verify that the lung has completely cleared.")
    if any(p in ["Cardiomegaly", "Edema"] for p, _ in positives):
        next_steps.append("• Heart Evaluation: Your care team may check your heart rhythm, perform an ultrasound of your heart (echocardiogram), or adjust medications (such as water pills/diuretics).")
    if any(p in ["Nodule", "Mass"] for p, _ in positives):
        next_steps.append("• Further Imaging: Having a spot on an X-ray is very common and often harmless (like a scar), but your doctor will likely order a CT scan to get a high-definition 3D view.")
    if any(p in ["Pneumothorax"] for p, _ in positives):
        next_steps.append("• Immediate Care: A doctor should examine you right away to ensure your breathing remains safe and decide if a small tube is needed to release trapped air.")
    if not next_steps:
        next_steps.append("• Discuss these findings with your doctor to decide the best treatment plan for your specific symptoms.")

    next_steps_text = "\n".join(next_steps)

    return (
        "What Was Detected On Your X-Ray:\n"
        f"{findings_text}\n\n"
        "What This Means & Typical Next Steps:\n"
        f"{next_steps_text}\n\n"
        "Important Questions To Ask Your Doctor:\n"
        "• 'What is the most likely cause of these findings for someone with my symptoms?'\n"
        "• 'Do I need any follow-up tests (such as blood work or a CT scan)?'\n"
        "• 'What symptoms should prompt me to seek immediate medical help?'"
    )


def _build_prescriptions(
    positives: list[tuple[str, float]],
    patient: dict[str, Any] | None = None,
) -> str:
    """Builds evidence-based clinical prescription and pharmacotherapy guidance tailored to findings and patient."""
    pos_map = {name: score for name, score in positives}
    p_age = patient.get("age") if patient else None
    p_comorbidities = (patient.get("comorbidities") or "").lower() if patient else ""

    rx_sections = []

    # 1. Pneumonia / Infiltration / Consolidation (IDSA/ATS Guidelines for CAP)
    if "Pneumonia" in pos_map or "Consolidation" in pos_map or "Infiltration" in pos_map:
        is_elderly = (p_age and int(p_age) >= 65)
        has_comorbid = any(k in p_comorbidities for k in ["diabet", "ckd", "kidney", "renal", "copd", "heart", "chf"])

        if is_elderly or has_comorbid:
            cap_rx = (
                "• PRIMARY ANTIMICROBIAL PRESCRIPTION (Patients with comorbidities / Age >= 65 per ATS/IDSA Guidelines):\n"
                "  - First-Line Combination: Amoxicillin-Clavulanate (Augmentin) 875/125 mg PO BID\n"
                "    PLUS Azithromycin 500 mg PO on Day 1, then 250 mg PO daily for 4 days (or Doxycycline 100 mg PO BID).\n"
                "  - Alternative (Beta-lactam allergic): Respiratory Fluoroquinolone — Levofloxacin 750 mg PO daily (adjust dosing for renal impairment/eGFR) or Moxifloxacin 400 mg PO daily.\n"
                "  - Duration: Minimum 5 days; patient should be afebrile for 48–72 hours prior to discontinuation.\n"
                "• ADJUNCTIVE SYMPTOMATIC THERAPY:\n"
                "  - Acetaminophen 500–1000 mg PO q6h PRN for pleuritic fever/discomfort (Max 3g/24h; monitor in hepatic impairment).\n"
                "  - Guaifenesin 600–1200 mg ER PO q12h with aggressive oral hydration to facilitate mucociliary clearance."
            )
        else:
            cap_rx = (
                "• PRIMARY ANTIMICROBIAL PRESCRIPTION (Uncomplicated Outpatient CAP per ATS/IDSA Guidelines):\n"
                "  - Standard First-Line: Amoxicillin 1 g PO TID (or Doxycycline 100 mg PO BID).\n"
                "  - Macrolide Option (in areas with pneumococcal macrolide resistance <25%): Azithromycin 500 mg Day 1, then 250 mg daily for 4 days.\n"
                "• ADJUNCTIVE THERAPY:\n"
                "  - Acetaminophen 500 mg PO q6h PRN for fever/pain.\n"
                "  - Adequate oral fluid intake and incentive spirometry."
            )
        rx_sections.append(cap_rx)

    # 2. Cardiomegaly / Pulmonary Edema (ACC/AHA Heart Failure Guidelines)
    if "Cardiomegaly" in pos_map or "Edema" in pos_map:
        hf_rx = (
            "• DECONGESTION & DIURETIC REGIMEN (ACC/AHA HF Guidelines for Volume Overload):\n"
            "  - Loop Diuretic: Furosemide (Lasix) 20–40 mg PO daily (or IV equivalent in acute decompensation), titrating to reach target dry weight and relieve orthopnea/edema.\n"
            "  - Electrolyte Supplementation: Potassium Chloride (KCl) 20 mEq PO daily with periodic serum K+ and creatinine checks.\n"
            "• GUIDELINE-DIRECTED MEDICAL THERAPY (GDMT) OPTIMIZATION:\n"
            "  - Evaluate for quadruple medical therapy initiation/titration: ARNI (Sacubitril/Valsartan), Beta-blocker (Carvedilol or Metoprolol Succinate), MRA (Spironolactone 25mg), and SGLT2i (Dapagliflozin 10mg / Empagliflozin 10mg).\n"
            "• CONTRAINDICATION ALERT:\n"
            "  - Strictly AVOID nonsteroidal anti-inflammatory drugs (NSAIDs like Ibuprofen/Naproxen) due to renal vasoconstriction and acute sodium/water retention."
        )
        rx_sections.append(hf_rx)

    # 3. Pulmonary Nodule / Mass (Fleischner Society Guidelines)
    if "Nodule" in pos_map or "Mass" in pos_map:
        nodule_rx = (
            "• PHARMACOTHERAPY & DIAGNOSTIC MEDICATION GUIDANCE (Fleischner Society 2017 Protocol):\n"
            "  - Antibiotic Stewardship: Do NOT prescribe empiric antibiotics for indeterminate solitary pulmonary nodules in the absence of active clinical infectious syndrome.\n"
            "  - Contrast CT Pre-Hydration: Ensure adequate oral hydration (500 mL water prior to IV iodinated contrast) if renal baseline is borderline.\n"
            "• SMOKING CESSATION PHARMACOTHERAPY (if active or recent smoker):\n"
            "  - First-Line: Varenicline (Chantix) 0.5 mg daily titrating to 1 mg BID, OR Combination Nicotine Replacement Therapy (NRT Transdermal Patch 21 mg/24h PLUS Nicotine Gum/Lozenge 2–4 mg PRN)."
        )
        rx_sections.append(nodule_rx)

    # 4. Pneumothorax (ACCP / British Thoracic Society Protocol)
    if "Pneumothorax" in pos_map:
        ptx_rx = (
            "• EMERGENCY ACUTE INTERVENTION & ANALGESIA:\n"
            "  - High-Flow Supplemental Oxygen: 10–15 L/min via non-rebreather mask (enhances pleural air absorption rate up to 4-fold via nitrogen washout).\n"
            "  - Analgesia Regimen: Multimodal pain control with Acetaminophen 1000 mg IV/PO q6h PLUS low-dose short-acting opioid (e.g. Oxycodone 5 mg PO q4-6h PRN severe pain); local intercostal anesthetic block (Lidocaine 1-2%) if procedural tube placement required.\n"
            "  - Procedural Decompression: Small-bore pigtail catheter (10–14 French) or needle aspiration for symptomatic or moderate/large collapse."
        )
        rx_sections.append(ptx_rx)

    # 5. Pleural Effusion (BTS Pleural Guidelines)
    if "Effusion" in pos_map and not ("Cardiomegaly" in pos_map or "Edema" in pos_map):
        effusion_rx = (
            "• PLEURAL FLUID MANAGEMENT & THERAPY:\n"
            "  - If suspect parapneumonic effusion: Proceed with targeted antibiotic therapy and diagnostic thoracentesis to assess fluid pH, LDH, protein, and gram stain.\n"
            "  - Procedural Local Anesthesia: 1% or 2% Lidocaine without epinephrine for bedside thoracentesis."
        )
        rx_sections.append(effusion_rx)

    # 6. Normal / Unremarkable
    if not rx_sections:
        return (
            "• NO CARDIOPULMONARY MEDICATIONS OR ANTIBIOTICS INDICATED:\n"
            "  - Avoid unnecessary empirical antimicrobial or diuretic therapy, preserving antimicrobial stewardship.\n"
            "  - For mild non-cardiopulmonary chest symptoms: Supportive symptomatic care with over-the-counter analgesics or saline throat lozenges as clinically appropriate."
        )

    return "\n\n".join(rx_sections)


def compose_report(
    scores: dict[str, float],
    threshold: float = 0.45,
    image_id: str | None = None,
    patient: dict[str, Any] | None = None,
) -> dict:
    positives = [(name, scores[name]) for name in PATHOLOGIES if scores.get(name, 0.0) >= threshold]
    positives.sort(key=lambda x: x[1], reverse=True)
    pos_map = {name: score for name, score in positives}

    # Normalize patient profile
    p_age = None
    p_sex = "Unspecified"
    p_setting = "Ambulatory Outpatient"
    p_indication = "Thoracic diagnostic evaluation"
    p_comorbidities = ""
    p_smoking = ""

    if patient:
        p_age = patient.get("age")
        p_sex = patient.get("sex", p_sex)
        p_setting = patient.get("setting", p_setting)
        p_indication = patient.get("indication", p_indication)
        p_comorbidities = patient.get("comorbidities", "")
        p_smoking = patient.get("smoking", "")

    # Build compartmentalized findings
    compartments: dict[str, str] = {}

    # 1. Lungs & Airways
    lung_pos = [name for name in COMPARTMENT_MAP["lungs"] if name in pos_map]
    if lung_pos:
        lung_phrases = []
        for name in lung_pos:
            s = pos_map[name]
            band = _severity(s)
            phrase = FINDING_LANGUAGE[name][band]
            lung_phrases.append(f"{phrase.capitalize()} (model score {s:.2f}).")

        # Clinical nuance for patient profile
        if "Pneumonia" in lung_pos or "Consolidation" in lung_pos:
            if p_age and int(p_age) >= 65:
                lung_phrases.append(f"In an elderly patient (age {p_age}), findings are concerning for acute community-acquired pneumonia or aspiration.")
            elif p_comorbidities and "copd" in p_comorbidities.lower():
                lung_phrases.append("Superimposed on underlying chronic lung disease, raises high concern for infectious acute exacerbation.")
        if "Nodule" in lung_pos or "Mass" in lung_pos:
            if p_smoking and "smok" in p_smoking.lower():
                lung_phrases.append(f"In a patient with documented tobacco history ({p_smoking}), focal parenchymal lesions carry elevated pre-test probability for neoplasm.")

        compartments["lungs"] = " ".join(lung_phrases)
    else:
        compartments["lungs"] = (
            "The lungs are well expanded and clear bilaterally without focal consolidation, airspace opacity, "
            "pulmonary mass, or suspicious nodule. Bronchovascular markings are normal in caliber."
        )

    # 2. Pleural Spaces & Sulci
    pleura_pos = [name for name in COMPARTMENT_MAP["pleura"] if name in pos_map]
    if pleura_pos:
        pleura_phrases = []
        for name in pleura_pos:
            s = pos_map[name]
            band = _severity(s)
            phrase = FINDING_LANGUAGE[name][band]
            pleura_phrases.append(f"{phrase.capitalize()} (model score {s:.2f}).")
        if "Pneumothorax" in pleura_pos:
            pleura_phrases.append("No gross mediastinal shift identified on frontal projection, but urgent clinical correlation for tension physiology is required.")
        compartments["pleura"] = " ".join(pleura_phrases)
    else:
        compartments["pleura"] = (
            "Costophrenic sulci and pleural spaces are sharp and clear. "
            "No evidence of pleural effusion or pneumothorax."
        )

    # 3. Cardiomediastinum & Vasculature
    cardio_pos = [name for name in COMPARTMENT_MAP["cardio"] if name in pos_map]
    if cardio_pos:
        cardio_phrases = []
        for name in cardio_pos:
            s = pos_map[name]
            band = _severity(s)
            phrase = FINDING_LANGUAGE[name][band]
            cardio_phrases.append(f"{phrase.capitalize()} (model score {s:.2f}).")
        if "Cardiomegaly" in cardio_pos:
            if p_comorbidities and any(k in p_comorbidities.lower() for k in ["heart failure", "chf"]):
                cardio_phrases.append("Findings correlate with known congestive cardiac history and clinical volume status.")
        compartments["cardio"] = " ".join(cardio_phrases)
    else:
        compartments["cardio"] = (
            "Cardiomediastinal silhouette and aortic contour are within normal limits for projection. "
            "Pulmonary vascular caliber and distribution are normal without vascular redistribution."
        )

    # 4. Osseous Structures & Soft Tissues
    osseous_pos = [name for name in COMPARTMENT_MAP["osseous"] if name in pos_map]
    if osseous_pos:
        osseous_phrases = []
        for name in osseous_pos:
            s = pos_map[name]
            band = _severity(s)
            phrase = FINDING_LANGUAGE[name][band]
            osseous_phrases.append(f"{phrase.capitalize()} (model score {s:.2f}).")
        osseous_phrases.append("Visualized thoracic skeleton otherwise demonstrates no acute displaced fracture.")
        compartments["osseous"] = " ".join(osseous_phrases)
    else:
        compartments["osseous"] = (
            "Visualized bony structures including ribs, clavicles, and thoracic spine are intact without acute fracture or "
            "destructive osseous lesion. Soft tissue envelopes and subdiaphragmatic regions appear unremarkable."
        )

    # Aggregate findings
    findings = (
        f"LUNGS & AIRWAYS: {compartments['lungs']}\n"
        f"PLEURA & SULCI: {compartments['pleura']}\n"
        f"CARDIOMEDIASTINUM: {compartments['cardio']}\n"
        f"OSSEOUS & SOFT TISSUES: {compartments['osseous']}"
    )

    # Build Impression
    critical_alert = False
    if not positives:
        impression = "No acute cardiopulmonary process."
        if p_indication and "pain" in p_indication.lower():
            impression += " Normal chest radiograph does not exclude non-radiopaque thoracic etiology or pulmonary embolism if high clinical suspicion."
    else:
        names = ", ".join(name.replace("_", " ") for name, _ in positives)
        imp_lines = []
        for i, (name, s) in enumerate(positives, start=1):
            disp_name = name.replace("_", " ")
            band = _severity(s)
            if name == "Pneumothorax":
                critical_alert = True
                imp_lines.append(f"{i}. {disp_name.upper()} ({band.capitalize()}, score {s:.2f}). Immediate clinical evaluation advised to exclude tension physiology.")
            elif name in ("Pneumonia", "Consolidation"):
                imp_lines.append(f"{i}. {disp_name.upper()} ({band.capitalize()}, score {s:.2f}), likely acute infective/inflammatory consolidation. Correlate with clinical exam, sputum, and labs.")
            elif name in ("Edema", "Cardiomegaly"):
                imp_lines.append(f"{i}. {disp_name.upper()} ({band.capitalize()}, score {s:.2f}) consistent with cardiopulmonary congestion / cardiomegaly.")
            elif name in ("Nodule", "Mass"):
                imp_lines.append(f"{i}. Thoracic {disp_name.upper()} ({band.capitalize()}, score {s:.2f}). Indeterminate etiology; requires guideline-directed evaluation.")
            else:
                imp_lines.append(f"{i}. {disp_name.capitalize()} ({band}, score {s:.2f}).")

        impression = "\n".join(imp_lines)
        if len(positives) == 1:
            impression += f"\nFindings most suggestive of {names}. Clinical correlation is recommended."
        else:
            impression += f"\nMulti-pathology study. Leading considerations: {names}. Correlate with symptoms, labs, and prior imaging."

    # Evidence-based Recommendations
    recs = []
    if "Pneumothorax" in pos_map:
        recs.append("1. Urgent bedside clinical assessment. If symptomatic or high-volume, consider immediate chest tube / pigtail thoracostomy decompression.")
    if "Pneumonia" in pos_map or "Consolidation" in pos_map or "Infiltration" in pos_map:
        recs.append(f"{len(recs)+1}. Initiate guideline-directed antimicrobial therapy per ATS/IDSA recommendations. Recommend interval follow-up chest radiograph in 4 to 6 weeks to document radiographic resolution.")
    if "Cardiomegaly" in pos_map or "Edema" in pos_map:
        recs.append(f"{len(recs)+1}. Assess volume status, serum NT-proBNP / troponin, and obtain 2D echocardiography per ACC/AHA guidelines to evaluate ventricular function.")
    if "Nodule" in pos_map or "Mass" in pos_map:
        recs.append(f"{len(recs)+1}. Recommend dedicated contrast-enhanced thin-slice chest CT per Fleischner Society 2017 guidelines for nodule/mass risk stratification.")
    if "Effusion" in pos_map:
        recs.append(f"{len(recs)+1}. Consider bedside thoracic ultrasound or lateral decubitus radiograph; evaluate for diagnostic thoracentesis if of new or undetermined etiology.")
    if not recs:
        recs.append("1. Routine clinical follow-up as clinically indicated. No urgent thoracic imaging or intervention required.")

    recommendations_text = "\n".join(recs)

    # Evidence-based Pharmacotherapy & Prescriptions
    prescriptions_text = _build_prescriptions(positives, patient)

    # Plain-Language Patient Summary
    patient_summary = _build_patient_summary(positives, p_age, p_sex, p_indication)

    # Patient demographics header
    patient_demographics = f"PATIENT: {str(p_age) + 'yo ' if p_age else ''}{p_sex} | SETTING: {p_setting}"
    if p_indication:
        patient_demographics += f"\nCLINICAL INDICATION: {p_indication}"
    if p_comorbidities:
        patient_demographics += f"\nPERTINENT MEDICAL HISTORY: {p_comorbidities}"

    header = "EXAMINATION: CHEST RADIOGRAPH (FRONTAL)"
    if image_id:
        header += f"\nACCESSION: {image_id}"

    draft = (
        f"{header}\n"
        f"{patient_demographics}\n"
        f"TECHNIQUE: Frontal chest radiograph. Automated multi-label DenseNet-121 analysis.\n\n"
        f"FINDINGS:\n{findings}\n\n"
        f"IMPRESSION:\n{impression}\n\n"
        f"RECOMMENDATIONS:\n{recommendations_text}\n\n"
        f"GUIDELINE PRESCRIPTIONS & PHARMACOTHERAPY:\n{prescriptions_text}\n\n"
        f"PATIENT-FRIENDLY SUMMARY:\n{patient_summary}\n\n"
        "DISCLAIMER: Research prototype. Not for clinical use. "
        "A licensed radiologist must review the images."
    )

    return {
        "findings": findings,
        "impression": impression,
        "positives": [{"pathology": n, "score": float(s), "severity": _severity(s)} for n, s in positives],
        "draft": draft,
        "source": "template",
        "compartments": compartments,
        "recommendations": recommendations_text,
        "prescriptions": prescriptions_text,
        "patient_summary": patient_summary,
        "critical_alert": critical_alert,
    }
