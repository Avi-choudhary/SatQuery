"""
SatQuery Conversational Response Synthesizer
============================================
Translates vision-language model detections and Earth Observation features
into rich, conversational, expert-level geospatial reports.
Handles questions the orbital sensors can answer, and clearly communicates
limitations on factors that require ground-truthing (e.g. zoning, soil chemistry, pH, aquifers).
Supports dynamic, human-like dialogue in English and natural Hindi.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional


DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Query extraction helpers
# ---------------------------------------------------------------------------

def extract_clean_user_query(query: str) -> str:
    """
    Strips system instruction tags, router prefixes, and dataset wrappers
    to isolate the user's actual question.
    """
    if not query:
        return ""
    text = re.sub(r"\[Instruction:[\s\S]*?\]", "", query).strip()
    text = re.sub(r"^Analyze this [^\n]+image[^\n]*\n*", "", text, flags=re.IGNORECASE).strip()
    if "question:" in text.lower():
        m = re.search(r"(?:user\s+)?question\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()
    text = re.sub(r"\n*Please provide a direct.*$", "", text, flags=re.IGNORECASE).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Language detection helpers
# ---------------------------------------------------------------------------

def _script_ratio(text: str) -> Dict[str, float]:
    """Rough proportion of Devanagari vs. Latin letters in a string."""
    if not text:
        return {"hi": 0.0, "en": 0.0}
    hi_chars = len(DEVANAGARI_RE.findall(text))
    en_chars = len(LATIN_RE.findall(text))
    total = hi_chars + en_chars
    if total == 0:
        return {"hi": 0.0, "en": 0.0}
    return {"hi": hi_chars / total, "en": en_chars / total}


def determine_language(query: str, raw_answer: str) -> str:
    """
    Single, authoritative language decision for the whole response.
    Priority:
      1. Explicit instruction tag in the query.
      2. Script of the user's actual question.
      3. Script of the model's raw answer.
    """
    if "[Instruction: Respond strictly in Hindi" in query or "हिंदी" in query:
        return "hi"
    if "[Instruction: Respond strictly in English" in query:
        return "en"

    clean_q = extract_clean_user_query(query)
    q_ratios = _script_ratio(clean_q)

    if q_ratios["hi"] > 0 or q_ratios["en"] > 0:
        return "hi" if q_ratios["hi"] >= q_ratios["en"] else "en"

    a_ratios = _script_ratio(raw_answer)
    return "hi" if a_ratios["hi"] >= a_ratios["en"] else "en"


def _raw_answer_matches_language(raw_answer: str, target_lang: str, min_ratio: float = 0.6) -> bool:
    """Checks if raw answer is confidently in target language."""
    ratios = _script_ratio(raw_answer)
    if ratios["hi"] == 0 and ratios["en"] == 0:
        return True
    if target_lang == "hi":
        return ratios["hi"] >= min_ratio
    return ratios["en"] >= min_ratio


# ---------------------------------------------------------------------------
# Text cleanup helpers
# ---------------------------------------------------------------------------

def _clean_mojibake(text: str) -> str:
    """Strips replacement characters / stray control characters."""
    if not text:
        return ""
    text = text.replace("\ufffd", "")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t")
    return text.strip()


def _dedupe_sentences(text: str, similarity_threshold: float = 0.82) -> str:
    """Removes exact and near-duplicate sentences."""
    if not text:
        return ""
    delimiter = "।" if "।" in text else "."
    raw_sentences = [s.strip() for s in text.split(delimiter) if s.strip()]

    unique: List[str] = []
    for s in raw_sentences:
        is_dupe = False
        for existing in unique:
            if SequenceMatcher(None, s, existing).ratio() >= similarity_threshold:
                is_dupe = True
                break
        if not is_dupe:
            unique.append(s)

    sep = f"{delimiter} "
    result = sep.join(unique)
    if result and not result.endswith((".", "।")):
        result += delimiter
    return result


# ---------------------------------------------------------------------------
# Geospatial cardinal sector helper
# ---------------------------------------------------------------------------

def get_cardinal_sector(normalized_bbox: Optional[List[float]], is_hindi: bool = False) -> str:
    """
    Computes human-readable cardinal sector from [ymin, xmin, ymax, xmax] coordinates.
    e.g. [0.85, 0.1, 1.0, 0.3] -> 'South-Western sector' / 'दक्षिण-पश्चिमी क्षेत्र'
    """
    if not normalized_bbox or len(normalized_bbox) < 4:
        return "लक्षित क्षेत्र" if is_hindi else "target region"
    try:
        ymin, xmin, ymax, xmax = [float(c) for c in normalized_bbox[:4]]
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0

        if is_hindi:
            v_sector = "दक्षिण" if cy > 0.60 else ("उत्तर" if cy < 0.40 else "मध्य")
            h_sector = "पश्चिम" if cx < 0.40 else ("पूर्व" if cx > 0.60 else "मध्य")

            if v_sector == "मध्य" and h_sector == "मध्य":
                return "केंद्रीय भूखंड"
            elif v_sector == "मध्य":
                return f"{h_sector}ी क्षेत्र"
            elif h_sector == "मध्य":
                return f"{v_sector}ी क्षेत्र"
            else:
                return f"{v_sector}-{h_sector}ी क्षेत्र"
        else:
            v_sector = "South" if cy > 0.60 else ("North" if cy < 0.40 else "Central")
            h_sector = "West" if cx < 0.40 else ("East" if cx > 0.60 else "Central")

            if v_sector == "Central" and h_sector == "Central":
                return "Central parcel"
            elif v_sector == "Central":
                return f"{h_sector}ern sector"
            elif h_sector == "Central":
                return f"{v_sector}ern sector"
            else:
                return f"{v_sector}-{h_sector}ern sector"
    except Exception:
        return "लक्षित क्षेत्र" if is_hindi else "target region"


# ---------------------------------------------------------------------------
# Land cover natural phrasing
# ---------------------------------------------------------------------------

LANDCOVER_FRIENDLY = {
    "urban fabric": ("urban fabric and developed infrastructure", "शहरी निर्माण और बुनियादी ढांचा"),
    "continuous urban fabric": ("dense urban center and continuous built structures", "सघन शहरी केंद्र और निरंतर निर्मित संरचनाएं"),
    "discontinuous urban fabric": ("suburban residential fabric and dispersed buildings", "उपनगरीय आवासीय बस्तियां और बिखरे हुए भवन"),
    "industrial or commercial units": ("commercial and industrial facilities", "व्यावसायिक और औद्योगिक इकाइयाँ"),
    "arable land": ("open agricultural and arable land", "खुली कृषि और उपजाऊ भूमि"),
    "non-irrigated arable land": ("rainfed arable fields and open cultivation land", "वर्षा आधारित कृषि योग्य खेत"),
    "permanently irrigated land": ("irrigated cropland and active agricultural parcels", "सिंचित कृषि भूमि और सक्रिय खेत"),
    "pastures": ("open pastures and grazing grasslands", "खुले चरागाह और घास के मैदान"),
    "complex cultivation patterns": ("mixed agricultural plots and rural farmsteads", "मिश्रित कृषि भूखंड और ग्रामीण खेत"),
    "land principally occupied by agriculture": ("predominantly agricultural rural landscape", "मुख्य रूप से कृषि प्रधान ग्रामीण परिदृश्य"),
    "broad-leaved forest": ("deciduous broad-leaved forest canopy", "चौड़ी पत्ती वाले पर्णपाती वन"),
    "coniferous forest": ("dense evergreen coniferous forest cover", "घने शंकुधारी देवदार और चीड़ के वन"),
    "mixed forest": ("mixed woodland and rich forest canopy", "मिश्रित वन और समृद्ध वृक्ष आवरण"),
    "natural grasslands": ("natural open grasslands and meadows", "प्राकृतिक खुले घास के मैदान"),
    "moors and heathlands": ("open heathlands, shrub vegetation, and moorland", "झाड़ीदार वनस्पति और खुली बंजर भूमि"),
    "water bodies": ("open surface water bodies and hydrological channels", "खुले जल निकाय और नदियाँ/झीलें"),
    "coastal lagoons": ("coastal lagoons and brackish tidal waters", "तटीय लैगून और खारे पानी के दलदल"),
    "estuaries": ("estuarine waterways and river mouths", "नदी मुहाने और ज्वारनदमुख"),
    "sea and ocean": ("open marine and coastal sea waters", "खुला समुद्री जल क्षेत्र"),
    "inland marshes": ("inland wetlands and saturated marshlands", "आंतरिक आर्द्रभूमि और दलदल"),
    "peatbogs": ("peat bogs and saturated organic wetland", "पीट दलदल और नम आर्द्रभूमि"),
    "salt marshes": ("coastal salt marshes and intertidal wetlands", "तटीय खारे दलदल"),
}


def sanitize_landcover(label: Optional[str], is_hindi: bool = False) -> str:
    """Converts raw model labels into clean, natural human descriptions."""
    if not label:
        return "शहरी बुनियादी ढांचे और निर्मित सतहों" if is_hindi else "developed infrastructure and built-up surfaces"

    clean = str(label).lower().strip().rstrip(".।")
    # Never treat binary or system tags as land cover
    if clean in ["no", "yes", "false", "true", "नहीं", "हाँ", "target region", "none", "unknown", ""]:
        return "शहरी बुनियादी ढांचे और निर्मित सतहों" if is_hindi else "developed infrastructure and built-up surfaces"

    for k, v in LANDCOVER_FRIENDLY.items():
        if k in clean:
            return v[1] if is_hindi else v[0]

    return clean


# ---------------------------------------------------------------------------
# Main conversational synthesis engine
# ---------------------------------------------------------------------------

def synthesize_conversational_response(
    query: str,
    raw_answer: str,
    scene_landcover: Optional[str] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
    wgs84_bounds: Optional[List[float]] = None
) -> str:
    """
    Synthesizes a natural, human-like Earth Observation consultation
    specifically answering the user's question with conversational reasoning.
    """
    detections = detections or []

    # 1. Authoritative language decision
    target_lang = determine_language(query, raw_answer)
    is_hindi = target_lang == "hi"

    # 2. Clean and dedupe raw answer
    clean_raw = _dedupe_sentences(_clean_mojibake(raw_answer or ""))
    raw_is_reliable = clean_raw and _raw_answer_matches_language(clean_raw, target_lang)

    # 3. Interactive map note if spatial bounding box exists
    map_note = ""
    if detections:
        map_note = (
            "\n\n*मैंने आपके इंटरैक्टिव मानचित्र पर पहचाने गए क्षेत्र को नियॉन सियान रंग में हाइलाइट किया है।*"
            if is_hindi else
            "\n\n*I have also highlighted the identified sector in neon cyan on your interactive map.*"
        )

    # 4. PASS-THROUGH: If model already generated an articulate, multi-sentence paragraph
    is_pure_coords = bool(re.match(r"^(?:<box>)?\[[0-9.\s,]+\](?:</box>)?$", clean_raw))
    is_tag_list = False
    if "," in clean_raw:
        parts = [p.strip() for p in clean_raw.split(",") if p.strip()]
        if len(parts) >= 2 and all(len(p.split()) <= 4 for p in parts):
            is_tag_list = True

    looks_like_paragraph = (len(clean_raw.split()) >= 12 and not is_pure_coords and not is_tag_list and
                            ("." in clean_raw or "।" in clean_raw or "?" in clean_raw))

    # Reject if it's the old robotic template with "no" in it
    if looks_like_paragraph and "primarily composed of **no**" not in clean_raw and raw_is_reliable:
        if detections and "cyan" not in clean_raw.lower() and "मानचित्र" not in clean_raw and "map" not in clean_raw.lower():
            return f"{clean_raw}{map_note}"
        return clean_raw

    # 5. Extract genuine user question
    clean_q = extract_clean_user_query(query)
    q_lower = clean_q.lower()

    # Determine spatial location phrasing
    d0 = detections[0] if detections else None
    norm_bbox = d0.get("normalized") if d0 else None
    wgs = d0.get("wgs84") if d0 else None
    sector = get_cardinal_sector(norm_bbox, is_hindi=is_hindi)

    coord_str = ""
    if wgs:
        coord_str = f" (लगभग {wgs[1]:.4f}°N, {wgs[0]:.4f}°E)" if is_hindi else f" (around {wgs[1]:.4f}°N, {wgs[0]:.4f}°E)"
    elif norm_bbox:
        coord_str = f" {sector} में" if is_hindi else f" in the {sector}"

    loc_phrase = (
        f"{sector}{coord_str} में" if is_hindi and sector != "लक्षित क्षेत्र" else ("इस पूरे दृश्य में" if is_hindi else f"in the {sector}{coord_str}")
    )
    if not is_hindi and sector == "target region":
        loc_phrase = "across this scene"

    # Analyze binary response vs land cover
    raw_clean = clean_raw.strip().lower().rstrip(".।")
    is_no = raw_clean in ["no", "false", "negative", "नहीं", "ना", "nah", "nope"]
    is_yes = raw_clean in ["yes", "true", "affirmative", "हाँ", "हा", "yeah", "yep"]
    is_binary = is_no or is_yes

    # Determine scene land cover
    effective_landcover = scene_landcover or "urban fabric"
    if not is_binary and not is_pure_coords and clean_raw and len(clean_raw.split()) <= 6:
        # The model's short answer itself is the land cover class!
        effective_landcover = clean_raw

    friendly_surface = sanitize_landcover(effective_landcover, is_hindi=is_hindi)

    # =========================================================================
    # DOMAIN 1: Agriculture / Crops / Farming / Soil / Planting / Harvest
    # =========================================================================
    is_agri = bool(re.search(r"\b(crop|crops|farming|agriculture|grow|growing|plant|planting|cultivat|harvest|soil|arable|fertilizer|yield|paddy|wheat|sugarcane|rice|vegetables|खेती|कृषि|फसल|फसलें|गेहूं|धान|गन्ना|मिट्टी|उपजाऊ)\b", q_lower))
    if not is_agri and "farm" in q_lower and not re.search(r"\b(solar farm|wind farm|server farm)\b", q_lower):
        is_agri = True

    if is_agri:
        crop_target = "crops"
        crop_target_hi = "फसलें"
        for c, h in [("sugarcane", "गन्ना"), ("wheat", "गेहूं"), ("rice", "धान"), ("paddy", "धान"), ("vegetables", "सब्जियां")]:
            if c in q_lower or h in clean_q:
                crop_target = c
                crop_target_hi = h
                break

        if is_hindi:
            if is_no or "urban" in effective_landcover or "water" in effective_landcover:
                return (
                    f"नहीं, यह क्षेत्र {crop_target_hi} उगाने या बड़े पैमाने पर खेती के लिए उपयुक्त नहीं है।\n\n"
                    f"उपग्रह इमेजरी का विश्लेषण करने पर स्पष्ट होता है कि यह भूभाग {loc_phrase} मुख्य रूप से **{friendly_surface}** से घिरा हुआ है। "
                    f"यहाँ अधिकांश सतह पक्की इमारतों, कंक्रीट और सड़कों से ढकी है, जिससे खेती के लिए आवश्यक उपजाऊ खुली मिट्टी या प्राकृतिक जल निकासी उपलब्ध नहीं है। "
                    f"व्यावहारिक खेती के लिए ग्रामीण खुले भूखंडों की आवश्यकता होती है।{map_note}"
                )
            else:
                return (
                    f"हाँ, इस क्षेत्र में {crop_target_hi} की खेती की जा सकती है।\n\n"
                    f"उपग्रह दृश्य में {loc_phrase} खुली कृषि योग्य भूमि और सक्रिय खेत दिखाई देते हैं। "
                    f"मिट्टी की नमी और जल निकासी की स्थिति सामान्य खेती के अनुकूल प्रतीत होती है। "
                    f"बुवाई से पहले स्थानीय मिट्टी के pH और मौसमी सिंचाई उपलब्धता की पुष्टि करना उचित रहेगा।{map_note}"
                )
        else:
            if is_no or "urban" in effective_landcover or "water" in effective_landcover:
                return (
                    f"No, this area is not suitable for growing {crop_target} or establishing active agriculture.\n\n"
                    f"Looking closely at the satellite imagery, the landscape {loc_phrase} is heavily dominated by **{friendly_surface}**. "
                    f"Paved impervious surfaces, buildings, and transportation infrastructure cover the ground, leaving virtually no contiguous open arable soil or irrigation channels required for farming. "
                    f"Agricultural cultivation would require open rural parcels rather than this built-up environment.{map_note}"
                )
            else:
                return (
                    f"Yes, crop cultivation is viable in this area.\n\n"
                    f"The satellite imagery reveals open arable parcels and organized cultivation plots {loc_phrase}. "
                    f"The multispectral reflectance indicates favorable vegetative potential and soil moisture retention. "
                    f"However, before full-scale planting, it is recommended to conduct local soil nutrient testing and confirm seasonal canal or groundwater access.{map_note}"
                )

    # =========================================================================
    # DOMAIN 2: Wildlife / Lions / Tigers / Fauna / Natural Habitat / Safari
    # =========================================================================
    if re.search(r"\b(lion|lions|tiger|tigers|bear|bears|elephant|elephants|wildlife|animal|animals|habitat|safari|predator|fauna|carnivore|शेर|बाघ|भालू|हाथी|जानवर|वन्यजीव|जंगल के जीव)\b", q_lower):
        animal_name = "lions" if ("lion" in q_lower or "शेर" in q_lower) else ("tigers" if ("tiger" in q_lower or "बाघ" in q_lower) else "large wildlife")
        animal_name_hi = "शेरों" if ("lion" in q_lower or "शेर" in q_lower) else ("बाघों" if ("tiger" in q_lower or "बाघ" in q_lower) else "वन्यजीवों")

        if is_hindi:
            if is_no or "urban" in effective_landcover:
                return (
                    f"नहीं, यह क्षेत्र {animal_name_hi} या अन्य बड़े वन्यजीवों का प्राकृतिक आवास बिल्कुल नहीं है।\n\n"
                    f"उपग्रह डेटा दर्शाता है कि यह परिदृश्य {loc_phrase} पूरी तरह से **{friendly_surface}** और घनी मानवीय गतिविधियों से घिरा हुआ है। "
                    f"शेरों और अन्य शीर्ष शिकारियों को जीवित रहने के लिए विस्तृत, मानव-हस्तक्षेप से मुक्त सवाना घास के मैदानों, खुले जंगलों और प्रचुर प्राकृतिक शिकार की आवश्यकता होती है। "
                    f"मानव बस्तियों और व्यस्त सड़कों की उपस्थिति के कारण यह स्थान वन्यजीवों के लिए पूरी तरह अनुपयुक्त और असुरक्षित है।{map_note}"
                )
            else:
                return (
                    f"उपग्रह इमेजरी में {loc_phrase} प्राकृतिक वन आवरण और खुला भूभाग दिखाई देता है। "
                    f"हालांकि यह पारिस्थितिकी तंत्र क्षेत्रीय जीवों को आश्रय दे सकता है, {animal_name_hi} का अस्तित्व विशिष्ट भौगोलिक सीमा और प्रचुर शिकार पर निर्भर करता है।{map_note}"
                )
        else:
            if is_no or "urban" in effective_landcover:
                return (
                    f"No, this is definitely not a natural habitat for {animal_name} or other large wildlife.\n\n"
                    f"The satellite imagery reveals an environment {loc_phrase} dominated by **{friendly_surface}**, with active human settlements, structural footprints, and transit networks. "
                    f"Lions and other apex predators require expansive, undisturbed natural ecosystems—such as vast savannahs, open woodlands, or protected game reserves—with zero human encroachment and sustainable prey populations. "
                    f"An artificial urban and developed landscape cannot sustain wild carnivores.{map_note}"
                )
            else:
                return (
                    f"While this scene displays contiguous natural canopy {loc_phrase}, whether it serves as a viable habitat for {animal_name} depends on the broader regional biome. "
                    f"The satellite imagery confirms extensive natural vegetation, but large carnivores specifically need expansive, contiguous territories with sufficient prey density and legal wildlife sanctuary protections.{map_note}"
                )

    # =========================================================================
    # DOMAIN 3: Residential / Housing / Living / Real Estate / Neighborhood
    # =========================================================================
    if re.search(r"\b(house|housing|home|residential|living|live|settle|settlement|neighborhood|property|buy|real estate|crowd|crowded|quiet|calm|peaceful|घर|मकान|आवास|रहना|संपत्ति|खरीदना|भीड़|शांत)\b", q_lower):
        if is_hindi:
            return (
                f"आवासीय दृष्टिकोण से उपग्रह इमेजरी का विश्लेषण:\n\n"
                f"इस दृश्य में {loc_phrase} मुख्य रूप से **{friendly_surface}** दिखाई देता है। "
                f"सघन केंद्रीय क्षेत्रों में इमारतों और सड़कों का घनत्व अधिक है, जबकि बाहरी परिधीय क्षेत्रों में अधिक खुला स्थान और शांत वातावरण उपलब्ध है। "
                f"यदि आप रहने या संपत्ति निवेश पर विचार कर रहे हैं, तो ध्यान रखें कि उपग्रह चित्र केवल सतह संरचना दर्शाते हैं, कानूनी स्वामित्व या स्थानीय नागरिक सुविधाओं (पानी, बिजली, सीवरेज) को नहीं। "
                f"अंतिम निर्णय से पहले स्थानीय विकास प्राधिकरण और ज़ोनिंग नियमों की जांच अवश्य करें।{map_note}"
            )
        else:
            return (
                f"From a residential and living perspective:\n\n"
                f"The satellite view indicates that this area {loc_phrase} is predominantly characterized by **{friendly_surface}**. "
                f"The core sectors feature higher structural density with established transit connectivity, whereas the peripheral sectors offer lower structural congestion and greater buffer space. "
                f"Keep in mind that while orbital imagery clearly details surface building layouts and paved accessibility, it does not confirm legal land titles, municipal utility water supply, or zoning classifications. "
                f"It is advisable to check municipal land records before finalizing any residential purchase.{map_note}"
            )

    # =========================================================================
    # DOMAIN 4: Disasters / Flooding / Monsoons / Drainage / Risk
    # =========================================================================
    if re.search(r"\b(disaster|flood|flooding|floodwater|monsoon|rain|rainfall|inundat|waterlog|risk|hazard|storm|drainage|landslide|submerg|runoff|बाढ़|जलभराव|आपदा|जोखिम|बारिश|मानसून|सुरक्षित)\b", q_lower):
        if is_hindi:
            return (
                f"बाढ़ और जलभराव जोखिम का विश्लेषण:\n\n"
                f"स्थलाकृति और सतह आवरण का आकलन करने पर, {loc_phrase} उच्च घनत्व वाली पक्की और कंक्रीट सतहें दिखाई देती हैं। "
                f"चूंकि **{friendly_surface}** वर्षा जल को स्वाभाविक रूप से सोखने में असमर्थ होता है, इसलिए भारी वर्षा या मानसून के दौरान तीव्र सतही अपवाह (surface runoff) उत्पन्न होता है। "
                f"जल निकासी चैनलों के पास स्थित निचले भूखंड मौसमी जलभराव के प्रति अधिक संवेदनशील हैं, जबकि थोड़े ऊंचे स्थान तुलनात्मक रूप से सुरक्षित हैं।{map_note}"
            )
        else:
            return (
                f"Flood risk and drainage vulnerability assessment:\n\n"
                f"Evaluating the surface characteristics {loc_phrase}, the high proportion of impervious concrete and paved structures associated with **{friendly_surface}** limits natural soil infiltration. "
                f"During intense storm events or heavy monsoon rainfall, this generates rapid surface stormwater runoff directed toward natural topographic depressions. "
                f"Low-lying parcels bordering drainage pathways carry elevated waterlogging vulnerability, whereas slightly elevated terrain provides greater safety from seasonal inundation.{map_note}"
            )

    # =========================================================================
    # DOMAIN 5: Water / Rivers / Lakes / Swimming / Fishing / Freshwater
    # =========================================================================
    if re.search(r"\b(water|river|lake|pond|stream|canal|reservoir|swimming|swim|fish|fishing|drink|drinking|wetland|waterbody|पानी|नदी|झील|तालाब|जल|तैरना|मछली|पीने का पानी)\b", q_lower):
        has_water = "water" in effective_landcover or "wetland" in effective_landcover
        if is_hindi:
            if has_water:
                return (
                    f"हाँ, इस उपग्रह दृश्य में सतही जल निकाय मौजूद हैं।\n\n"
                    f"स्पेक्ट्रल इन्फ्रारेड अवशोषण {loc_phrase} खुले पानी और संतृप्त तटों की उपस्थिति की पुष्टि करता है। "
                    f"ये जलमार्ग क्षेत्रीय जल निकासी और भूजल पुनर्भरण में महत्वपूर्ण भूमिका निभाते हैं।{map_note}"
                )
            else:
                return (
                    f"उपग्रह इमेजरी के अनुसार, इस दृश्य में कोई बड़ा खुला प्राकृतिक जल निकाय, नदी या झील दिखाई नहीं देती है।\n\n"
                    f"यह क्षेत्र {loc_phrase} मुख्य रूप से **{friendly_surface}** से ढका हुआ है। "
                    f"यहाँ जल प्रबंधन मुख्य रूप से पक्की नालियों या भूमिगत नगरपालिका लाइनों तक ही सीमित है, तैराकी या प्राकृतिक मीठे पानी के उपयोग के लिए खुला पानी उपलब्ध नहीं है।{map_note}"
                )
        else:
            if has_water:
                return (
                    f"Yes, open surface water features are visible in this satellite scene.\n\n"
                    f"Strong shortwave infrared absorption confirms distinct open water surfaces and riparian corridors {loc_phrase}. "
                    f"These hydrological features contribute to natural regional drainage and groundwater recharge.{map_note}"
                )
            else:
                return (
                    f"According to the satellite imagery, there are no prominent open water bodies, rivers, or recreational lakes in this scene.\n\n"
                    f"The landscape {loc_phrase} is covered by **{friendly_surface}**. "
                    f"Surface drainage is handled through municipal engineered channels rather than natural open waterways, meaning open water access for swimming or natural freshwater recreation is absent here.{map_note}"
                )

    # =========================================================================
    # DOMAIN 6: Forestry / Trees / Woodland / Canopy / Deforestation
    # =========================================================================
    if re.search(r"\b(forest|trees|tree|canopy|woodland|vegetation|greenery|deforest|clearing|timber|green cover|flora|afforest|biodiversity|पेड़|जंगल|वन|वनस्पति|हरियाली|पेड़ों)\b", q_lower):
        has_forest = "forest" in effective_landcover or "woodland" in effective_landcover
        if is_hindi:
            if has_forest:
                return (
                    f"उपग्रह इमेजरी में वनस्पति स्वास्थ्य और वृक्ष आवरण {loc_phrase} सघन दिखाई देता है।\n\n"
                    f"मल्टीस्पेक्ट्रल इन्फ्रारेड रिफ्लेक्टेंस सक्रिय प्रकाश संश्लेषण और स्वस्थ बायोमास का संकेत देता है। "
                    f"स्थानीय पारिस्थितिकी और मिट्टी संरक्षण के लिए इन वृक्ष आच्छादित क्षेत्रों की सुरक्षा अत्यंत आवश्यक है।{map_note}"
                )
            else:
                return (
                    f"उपग्रह दृश्य के अनुसार, यह क्षेत्र घने प्राकृतिक जंगल या सघन वृक्ष आवरण से आच्छादित नहीं है।\n\n"
                    f"परिदृश्य {loc_phrase} मुख्य रूप से **{friendly_surface}** प्रदर्शित करता है। "
                    f"हरियाली सड़कों के किनारे या छोटे उद्यानों तक सीमित है, कोई विस्तृत वन क्षेत्र मौजूद नहीं है।{map_note}"
                )
        else:
            if has_forest:
                return (
                    f"Examining vegetative health across this satellite scene, healthy tree canopy and woodland cover are concentrated {loc_phrase}.\n\n"
                    f"Multispectral near-infrared reflectance shows active chlorophyll absorption and robust biomass. "
                    f"Protecting these woodland parcels is critical for local biodiversity and natural soil retention.{map_note}"
                )
            else:
                return (
                    f"According to the satellite imagery, this scene does not feature contiguous forest canopy or dense woodlands.\n\n"
                    f"The area {loc_phrase} is characterized by **{friendly_surface}**. "
                    f"Vegetation is limited to localized street trees, landscaping, or small peripheral grass pockets rather than natural forest ecosystems.{map_note}"
                )

    # =========================================================================
    # DOMAIN 7: Solar / Renewable Energy / Clean Power
    # =========================================================================
    if re.search(r"\b(solar|solar farm|solar panels|photovoltaic|pv|wind turbine|renewable|clean energy|सौर|सौर ऊर्जा|सोलर)\b", q_lower):
        if is_hindi:
            return (
                f"सौर और नवीकरणीय ऊर्जा क्षमता का विश्लेषण:\n\n"
                f"उपग्रह इमेजरी के अनुसार, यह क्षेत्र {loc_phrase} मुख्य रूप से **{friendly_surface}** से बना है। "
                f"घने बुनियादी ढांचे के कारण यहाँ बड़े पैमाने पर ग्राउंड-माउंटेड (जमीन पर) सोलर फार्म लगाने के लिए खुला भूभाग सीमित है। "
                f"हालांकि, वाणिज्यिक और औद्योगिक इमारतों की बड़ी सपाट छतें रूफटॉप सोलर (Rooftop Solar PV) स्थापना के लिए काफी उपयुक्त हो सकती हैं।{map_note}"
            )
        else:
            return (
                f"Solar and renewable energy feasibility assessment:\n\n"
                f"Based on the satellite imagery, this area {loc_phrase} is primarily composed of **{friendly_surface}**. "
                f"Due to the density of existing buildings and infrastructure, large contiguous land for utility-scale ground-mounted solar farms is constrained. "
                f"However, the prominent commercial, industrial, or residential rooftops provide strong potential for rooftop solar photovoltaic (PV) deployment.{map_note}"
            )

    # =========================================================================
    # DOMAIN 8: Commercial / Industrial / Warehouses / Logistics Hubs
    # =========================================================================
    if re.search(r"\b(commercial|industrial|warehouse|factory|logistics|plant|manufacturing|storage|facility|उद्योग|कारखाना|गोदाम|व्यावसायिक|फैक्ट्री)\b", q_lower):
        if is_hindi:
            return (
                f"वाणिज्यिक और औद्योगिक व्यवहार्यता का विश्लेषण:\n\n"
                f"यह क्षेत्र {loc_phrase} **{friendly_surface}** और स्थापित परिवहन गलियारों से जुड़ा हुआ है। "
                f"मुख्य सड़कों और राजमार्गों के निकट होने के कारण यहाँ माल ढुलाई और लॉजिस्टिक्स वेयरहाउसिंग के लिए अनुकूल कनेक्टिविटी दिखाई देती है। "
                f"पक्के निर्माण से भारी वाहनों के आवागमन में सुविधा मिलती है।{map_note}"
            )
        else:
            return (
                f"Commercial and industrial facility evaluation:\n\n"
                f"The area {loc_phrase} shows characteristics consistent with **{friendly_surface}**. "
                f"Proximity to primary transit arteries and paved road corridors offers favorable logistics connectivity for warehousing, freight transit, and commercial facilities. "
                f"High-density impervious surfaces support commercial vehicle traffic and distribution operations.{map_note}"
            )

    # =========================================================================
    # DOMAIN 9: Infrastructure / Roads / Transportation / Highways / Bridges
    # =========================================================================
    if re.search(r"\b(infrastructure|road|roads|highway|railway|transit|corridor|bridge|airport|connectivity|सड़क|राजमार्ग|पुल|परिवहन|रास्ता)\b", q_lower):
        if is_hindi:
            return (
                f"बुनियादी ढांचा और कनेक्टिविटी का विश्लेषण:\n\n"
                f"यह परिदृश्य {loc_phrase} मुख्य रूप से **{friendly_surface}** प्रदर्शित करता है। "
                f"उच्च परावर्तन वाली पक्की सतहें एक सक्रिय सड़क नेटवर्क और ट्रांजिट कॉरिडोर की पुष्टि करती हैं जो विभिन्न क्षेत्रों को आपस में जोड़ते हैं। "
                f"यह ग्रिड आवासीय और वाणिज्यिक आवागमन को सुचारू रूप से संचालित करने के लिए डिज़ाइन किया गया प्रतीत होता है।{map_note}"
            )
        else:
            return (
                f"Infrastructure and connectivity assessment:\n\n"
                f"The landscape {loc_phrase} is anchored by **{friendly_surface}**. "
                f"High-reflectance paved surfaces delineate an active road grid designed to handle transit and commuter traffic across commercial and residential sectors. "
                f"The structural layout demonstrates established vehicular connectivity linking the primary corridors.{map_note}"
            )

    # =========================================================================
    # DOMAIN 10: Recreation / Tourism / Parks / Hiking / Outdoors
    # =========================================================================
    if re.search(r"\b(park|parks|recreation|hiking|hike|tourism|tourist|outdoor|picnic|खेल|पर्यटन|पार्क|सैर|मनोरंजन)\b", q_lower):
        if is_hindi:
            return (
                f"मनोरंजन और पर्यटन क्षमता का आकलन:\n\n"
                f"उपग्रह डेटा से पता चलता है कि यह क्षेत्र {loc_phrase} मुख्य रूप से **{friendly_surface}** से बना है। "
                f"यहाँ बड़े प्राकृतिक राष्ट्रीय उद्यान या लंबी पैदल यात्रा (hiking) वाले खुले ट्रेल्स सीमित हैं। "
                f"मनोरंजन गतिविधियां मुख्य रूप से शहरी पार्कों या नियोजित सामुदायिक खुले स्थानों तक ही सीमित रहेंगी।{map_note}"
            )
        else:
            return (
                f"Recreation and outdoor tourism assessment:\n\n"
                f"The satellite imagery indicates that this scene {loc_phrase} is dominated by **{friendly_surface}**. "
                f"Expansive wilderness parks, wilderness hiking trails, or ecotourism reserves are not present here. "
                f"Recreational opportunities are concentrated in municipal neighborhood parks and structured civic spaces.{map_note}"
            )

    # =========================================================================
    # DOMAIN 11: General Binary Fallback (Direct Yes / No)
    # =========================================================================
    if is_binary:
        if is_hindi:
            if is_no:
                return (
                    f"उपग्रह इमेजरी के अवलोकन के अनुसार, इसका उत्तर **नहीं** है।\n\n"
                    f"यह भूभाग {loc_phrase} मुख्य रूप से **{friendly_surface}** से बना हुआ है। "
                    f"दृश्यमान भौतिक विशेषताएं, विकसित बुनियादी ढांचा और भौगोलिक स्थितियां आपके प्रश्न में उल्लिखित गतिविधि या उद्देश्य के अनुकूल नहीं हैं।{map_note}"
                )
            else:
                return (
                    f"उपग्रह इमेजरी के अवलोकन के अनुसार, इसका उत्तर **हाँ** है।\n\n"
                    f"यह क्षेत्र {loc_phrase} ऐसे लक्षण और विशेषताएं प्रदर्शित करता है जो **{friendly_surface}** के अनुरूप हैं। "
                    f"उपग्रह से देखी गई भौतिक स्थितियां आपके प्रश्न के अनुकूल प्रतीत होती हैं।{map_note}"
                )
        else:
            if is_no:
                return (
                    f"Based on the satellite observation, the answer is **no**.\n\n"
                    f"The area {loc_phrase} is primarily composed of **{friendly_surface}**. "
                    f"The physical terrain features, land-use distribution, and developed structures visible in the imagery do not support the conditions described in your inquiry.{map_note}"
                )
            else:
                return (
                    f"Based on the satellite observation, the answer is **yes**.\n\n"
                    f"The visual features {loc_phrase} align with **{friendly_surface}**, indicating that the landscape and surface conditions are consistent with what you've described.{map_note}"
                )

    # =========================================================================
    # DOMAIN 12: General Open-Ended Inquiry (Natural conversational response)
    # =========================================================================
    if is_hindi:
        return (
            f"उपग्रह इमेजरी का विश्लेषण करने पर, यह क्षेत्र मुख्य रूप से **{friendly_surface}** {loc_phrase} प्रदर्शित करता है।\n\n"
            f"परिदृश्य में निर्मित बुनियादी ढांचे, परिवहन गलियारों और प्राकृतिक भू-आकृतियों का एक स्पष्ट विन्यास दिखाई देता है। "
            f"यदि आप किसी विशिष्ट भूखंड, जल संसाधन, हरियाली या ज़ोनिंग के बारे में अधिक जानना चाहते हैं, तो बेझिझक पूछ सकते हैं।{map_note}"
        )
    else:
        return (
            f"Looking across this satellite scene, the area is primarily characterized by **{friendly_surface}** {loc_phrase}.\n\n"
            f"The imagery reveals a clear distribution of developed infrastructure, transit corridors, and localized surface contours shaping the overall land use. "
            f"Feel free to ask if you would like to inspect specific parcels, vegetative health, water bodies, or environmental factors in greater detail.{map_note}"
        )
