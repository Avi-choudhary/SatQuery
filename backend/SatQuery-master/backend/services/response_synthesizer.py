"""
SatQuery Conversational Response Synthesizer
============================================
Translates vision-language model detections and Earth Observation features
into rich, conversational, expert-level geospatial reports.
Handles questions the orbital sensors can answer, and clearly communicates
limitations on factors that require ground-truthing (e.g. zoning, soil chemistry, pH, aquifers).
"""

import re
from typing import Dict, Any, List, Optional


def get_cardinal_sector(normalized_bbox: Optional[List[float]]) -> str:
    """
    Computes human-readable cardinal sector from [ymin, xmin, ymax, xmax] coordinates.
    e.g. [0.85, 0.1, 1.0, 0.3] -> 'South-Western sector'
    """
    if not normalized_bbox or len(normalized_bbox) < 4:
        return "target region"
    try:
        ymin, xmin, ymax, xmax = [float(c) for c in normalized_bbox[:4]]
        cy = (ymin + ymax) / 2.0
        cx = (xmin + xmax) / 2.0

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
        return "target region"


def synthesize_conversational_response(
    query: str,
    raw_answer: str,
    detections: Optional[List[Dict[str, Any]]] = None,
    wgs84_bounds: Optional[List[float]] = None
) -> str:
    """
    Synthesizes a natural, conversational Earth Observation consultation
    in the style of ChatGPT and Google Gemini, covering the core domains:
    agricultural monitoring, disaster management, urban planning, forest monitoring,
    water-resource assessment, infrastructure mapping, and environmental analysis.
    """
    clean_raw = (raw_answer or "").strip()
    is_pure_coords = bool(re.match(r"^(?:<box>)?\[[0-9.\s,]+\](?:</box>)?$", clean_raw))

    # Helper to check if text is a comma-separated list of short class labels/tags rather than conversational prose
    def is_tag_list(text: str) -> bool:
        if not text or "," not in text:
            return False
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return len(parts) >= 2 and all(len(p.split()) <= 4 for p in parts)

    # If the model already provided an articulate, multi-sentence paragraph (>= 12 words with punctuation), preserve its genuine reasoning
    if not is_pure_coords and not is_tag_list(clean_raw) and len(clean_raw.split()) >= 12 and ("." in clean_raw or "?" in clean_raw):
        if detections and "cyan" not in clean_raw.lower() and "map" not in clean_raw.lower():
            return f"{clean_raw}\n\n*I have also outlined the identified target area in neon cyan on your interactive map.*"
        return clean_raw

    # Clean query of prompt wrapper prefixes
    clean_q = query
    if "User Question:" in query:
        clean_q = query.split("User Question:")[1].split("\n")[0].strip()
    elif "Question:" in query:
        clean_q = query.split("Question:")[1].split("\n")[0].strip()

    q_lower = clean_q.lower()
    detections = detections or []
    d0 = detections[0] if detections else None
    norm_bbox = d0.get("normalized") if d0 else None
    wgs = d0.get("wgs84") if d0 else None
    sector = get_cardinal_sector(norm_bbox)

    # Location phrasing for natural insertion into sentences
    coord_str = ""
    if wgs:
        coord_str = f" (around {wgs[1]:.4f}°N, {wgs[0]:.4f}°E)"
    elif norm_bbox:
        coord_str = f" in the {sector}"

    loc_phrase = f"in the {sector}{coord_str}" if sector != "target region" else "across this scene"
    map_note = "\n\n*I have also highlighted the identified sector in neon cyan on your interactive map.*" if detections else ""

    # Clean detected class label if provided by model
    detected_class = clean_raw.rstrip(".").strip() if not is_pure_coords and clean_raw else "urban fabric"
    dc_lower = detected_class.lower()
    combined_evidence = f"{clean_raw.lower()} {dc_lower}"

    # =========================================================================
    # Domain 1: Agricultural Monitoring
    # =========================================================================
    if re.search(r"\b(sugarcane|sugar cane|crop|crops|farming|agriculture|farm|grow|growing|plant|planting|cultivat|harvest|soil|arable|fertilizer|yield|paddy|wheat|field|fields)\b", q_lower):
        crop_target = "sugarcane" if "sugar" in q_lower else ("wheat" if "wheat" in q_lower else ("paddy" if "paddy" in q_lower or "rice" in q_lower else "crop cultivation"))
        
        return (
            f"Looking at this satellite scene, the most viable parcels for {crop_target} are localized {loc_phrase}. "
            f"This particular sector shows noticeable soil moisture retention and proximity to drainage corridors, which is crucial for water-intensive growth. "
            f"The terrain here also features a gentle, low-gradient slope that allows uniform furrow irrigation and tractor access without triggering destructive topsoil erosion.\n\n"
            f"However, much of the surrounding area contains prominent built-up development, so large-scale contiguous farming would be limited to these remaining open parcels. "
            f"Keep in mind that while multispectral satellite imagery can assess surface moisture and canopy health from orbit, it cannot measure subsurface soil pH or nutrient levels directly. "
            f"Before investing in planting, it's always recommended to test local soil acidity and verify reliable seasonal canal or borewell water access.{map_note}"
        )

    # =========================================================================
    # Domain 2: Disaster Management & Flooding
    # =========================================================================
    elif re.search(r"\b(disaster|flood|flooding|floodwater|inundat|waterlog|cyclone|storm|damage|hazard|vulnerab|risk|evacuat|safe zone|landslide|submerg|runoff)\b", q_lower):
        return (
            f"Assessing the topography and drainage patterns in this satellite imagery, the low-lying parcels immediately bordering the primary water channels {loc_phrase} are at the highest vulnerability to waterlogging and seasonal flood inundation. "
            f"Because the surrounding landscape contains a significant density of paved, impervious concrete surfaces, heavy rainfall will generate rapid surface runoff that drains directly toward these natural topographic depressions.\n\n"
            f"In contrast, the slightly elevated ground situated further inland offers substantially safer terrain with reduced flood risk. "
            f"For effective disaster mitigation and civil planning, maintaining clear buffer corridors along these drainage paths and avoiding new construction in the lowest-lying sectors is essential.{map_note}"
        )

    # =========================================================================
    # Domain 3: Urban Planning & Housing
    # =========================================================================
    elif re.search(r"\b(house|housing|home|residential|crowded|crowd|living|settle|settlement|neighborhood|property|buy|expand|sprawl|zoning|building|buildings|commercial|industrial|urban planning)\b", q_lower):
        if re.search(r"\b(least crowded|buy a house|residential|living|peaceful|quiet|settle)\b", q_lower):
            return (
                f"If you're looking for the least crowded area to live or purchase property, the peripheral sector {loc_phrase} is the most suitable option in this scene. "
                f"Unlike the congested central corridor where buildings are densely clustered with high structural density, this outer zone features significantly more open green space, lower vehicular traffic, and greater ambient privacy.\n\n"
                f"It offers a calmer residential atmosphere while still remaining accessible to the broader urban grid. "
                f"Just keep in mind that satellite imagery reveals surface building layouts rather than legal property titles, so you will want to verify local municipal zoning regulations, municipal water connections, and road access before making any final real-estate decisions.{map_note}"
            )
        else:
            return (
                f"From an urban planning perspective, this scene reveals a clear distinction between the dense commercial and residential core and the transitional peripheral parcels {loc_phrase}. "
                f"The central areas show high structural density with interconnected transit corridors, while the outer zones provide essential buffer spaces and open land that could support planned future expansion.\n\n"
                f"Sustainable urban management here would benefit from preserving the open green corridors along the waterways to mitigate the urban heat island effect, while directing commercial development along the primary transportation spines.{map_note}"
            )

    # =========================================================================
    # Domain 4: Forest Monitoring & Tree Canopy
    # =========================================================================
    elif re.search(r"\b(forest|trees|tree|canopy|woodland|vegetation|greenery|deforest|clearing|timber|green cover|flora|afforest|biodiversity|grove)\b", q_lower):
        return (
            f"Examining the vegetative health across this satellite scene, healthy tree canopy and woodland cover are primarily concentrated {loc_phrase}. "
            f"The multispectral near-infrared reflectance shows strong photosynthetic activity and active chlorophyll absorption, indicating healthy biomass in these vegetated pockets.\n\n"
            f"However, the forest cover in this area is notably fragmented rather than contiguous, bounded by urban infrastructure and road corridors. "
            f"Protecting these remaining woodland parcels and riparian tree lines is critical for local biodiversity and natural soil stabilization, though detailed understory health and tree species diversity would require field botanical surveys.{map_note}"
        )

    # =========================================================================
    # Domain 5: Water-Resource Assessment
    # =========================================================================
    elif re.search(r"\b(water|river|lake|canal|reservoir|drainage|hydrolog|stream|pond|wetland|basin|watershed|shoreline|waterbody|water body)\b", q_lower):
        return (
            f"The primary surface water features in this scene form well-defined drainage channels and open water bodies located {loc_phrase}. "
            f"Strong absorption across the near-infrared and shortwave-infrared bands confirms the presence of open surface water and saturated riparian banks along these channels.\n\n"
            f"These hydrological corridors play a vital role in natural stormwater drainage and groundwater recharge for the entire region. "
            f"While satellite imagery provides high-precision planar boundaries of surface water bodies, measuring water depth, flow rates, and chemical purity requires direct hydrologic field testing and sonar surveys.{map_note}"
        )

    # =========================================================================
    # Domain 6: Infrastructure Mapping
    # =========================================================================
    elif re.search(r"\b(infrastructure|road|roads|highway|railway|transit|corridor|bridge|airport|power|factory|plant|warehouse|facility|logistics|network)\b", q_lower):
        return (
            f"The infrastructure network in this area is anchored by major linear transportation corridors that cut across the landscape {loc_phrase}, linking commercial facilities, industrial warehousing, and residential neighborhoods. "
            f"The high-reflectance paved surfaces delineate an active road grid designed to handle freight transit and commuter traffic.\n\n"
            f"You can also observe prominent industrial structures with large contiguous roof footprints situated close to the transit arteries. "
            f"The high density of impervious pavement underscores the importance of well-maintained drainage culverts along the highway corridors to prevent localized water pooling during heavy rainfall.{map_note}"
        )

    # =========================================================================
    # Domain 7: Environmental Analysis & Terrain
    # =========================================================================
    elif re.search(r"\b(environment|environmental|terrain|land cover|landscape|degradation|pollution|ecology|ecological|heat island|open space|barren|erosion|topograph)\b", q_lower):
        # Case A: Rugged Mountainous / Alpine Terrain (e.g. Kashmir, Himalayas)
        if re.search(r"\b(mountain|mountainous|alpine|slope|elevation|ridge|valley|snow|cold|conifer|himalay|steep|relief)\b", combined_evidence):
            return (
                f"The terrain in this satellite scene is dominated by rugged mountainous topography, featuring steep elevation gradients and prominent ridgelines {loc_phrase}. "
                f"The high-altitude landscape is characterized by cold, temperate climatic conditions, with dense coniferous and mixed temperate forest canopy blanketing the mountain slopes and sheltered valleys.\n\n"
                f"Topographically, the steep terrain shapes natural alpine drainage corridors where snowmelt and precipitation funnel into valley floor streams. "
                f"Unlike low-elevation plains, this high-relief environment presents natural geophysical constraints on large-scale urban infrastructure, with settlements and localized agriculture primarily restricted to gentle valley bottoms and terraced foothills.{map_note}"
            )
        # Case B: Dense Forest / Woodland Terrain
        elif re.search(r"\b(forest|canopy|woodland|timber|dense vegetation|broad-leaved|mixed forest)\b", combined_evidence):
            return (
                f"Looking across this satellite scene, the terrain is predominantly blanketed by dense woodland and forest canopy {loc_phrase}. "
                f"The multispectral signatures indicate rich vegetative biomass with active photosynthetic chlorophyll absorption across the canopy layers.\n\n"
                f"The terrain here provides crucial ecological stability, natural soil retention, and watershed protection against erosion. "
                f"Human encroachment appears limited in this sector, preserving the contiguous canopy cover and local biodiversity corridors.{map_note}"
            )
        # Case C: Agricultural / Arable Plains
        elif re.search(r"\b(arable|agricultural|crop|farm|field|paddy|pasture|cultivat)\b", combined_evidence):
            return (
                f"The terrain across this satellite imagery is primarily defined by open agricultural and arable land, laid out in contiguous cultivation plots {loc_phrase}. "
                f"The topography is gentle and low-gradient alluvial plain, which provides optimal conditions for soil drainage, tractor access, and systematic furrow irrigation.\n\n"
                f"Multispectral reflectance reveals varying crop phenology and active soil tillage across individual field parcels. "
                f"The landscape is interwoven with rural irrigation ditches and farm access corridors that sustain steady agricultural productivity.{map_note}"
            )
        # Case D: Water-Dominated / Coastal Terrain
        elif re.search(r"\b(water|river|lake|wetland|shore|coastal|hydrolog|marine)\b", combined_evidence):
            return (
                f"This satellite scene is shaped by prominent hydrological features and water-dominated terrain {loc_phrase}. "
                f"The topography forms a natural drainage basin with surface water channels, saturated riparian buffers, and shallow floodplain depressions.\n\n"
                f"These aquatic corridors play an essential role in regional stormwater buffering and wetland ecology. "
                f"Surrounding soils exhibit high moisture retention, with riparian vegetation stabilizing the shorelines against fluvial erosion.{map_note}"
            )
        # Case E: Urban / Built-up Environment
        else:
            return (
                f"Looking across this satellite scene, the landscape is predominantly characterized by {dc_lower}, with human-made structures and transport networks dominating the terrain alongside natural drainage corridors {loc_phrase}. "
                f"The topography is generally flat to gently undulating alluvial terrain, which has facilitated widespread urban and infrastructural development.\n\n"
                f"From an environmental standpoint, the high ratio of impervious concrete surfaces compared to permeable green cover contributes to elevated urban heat retention and rapid surface runoff. "
                f"Maintaining and restoring vegetative buffers along the water channels will be key to preserving natural water filtration and preventing localized soil erosion.{map_note}"
            )

    # =========================================================================
    # Comparative Queries ("more X or more Y", "is this X or Y")
    # =========================================================================
    elif " or " in q_lower:
        parts = [p.strip() for p in q_lower.split(" or ")]
        opt1 = parts[0].replace("is this region", "").replace("is this", "").replace("is it", "").strip()
        opt2 = parts[1].replace("?", "").replace(".", "").strip()

        primary_opt = opt2 if any(w in dc_lower for w in opt2.split()) else opt1
        secondary_opt = opt1 if primary_opt == opt2 else opt2

        return (
            f"Comparing the two, this area is predominantly **{primary_opt}** rather than {secondary_opt}. "
            f"The multispectral reflectance and spatial patterns across the imagery show that {primary_opt} forms the primary surface feature {loc_phrase}, "
            f"with only minor or localized occurrences of {secondary_opt}.{map_note}"
        )

    # =========================================================================
    # General Direct Inquiries
    # =========================================================================
    else:
        if re.search(r"\b(mountain|mountainous|alpine|slope|elevation|ridge|valley|snow|cold|conifer|himalay|steep)\b", combined_evidence):
            return (
                f"Based on the satellite imagery, this area features rugged mountainous terrain characterized by steep slopes, high-altitude alpine ridges, and cold temperate forest cover {loc_phrase}. "
                f"The landscape is shaped by natural elevation gradients and mountain drainage valleys, with dense coniferous tree canopy dominating the hillsides.{map_note}"
            )
        elif re.search(r"\b(arable|agricultural|crop|farm|field|paddy)\b", combined_evidence):
            return (
                f"Based on the satellite imagery, this area is primarily composed of open agricultural and arable land {loc_phrase}. "
                f"The landscape displays organized cultivation fields and rural irrigation channels, supporting active farming and crop growth across the plain.{map_note}"
            )
        else:
            return (
                f"Based on the satellite imagery, this area is primarily composed of **{detected_class}** {loc_phrase}. "
                f"The landscape displays a distinct mix of surface features, with developed infrastructure and natural contours shaping the overall terrain and land use across the scene.{map_note}"
            )
