"""Extract motorable OSM roads for the study bounding box from the Geofabrik South India .osm.pbf.
Usage: python3 00_extract_osm_roads.py
Output: data/raw/osm/roads_osm_extract.json (Overpass-like JSON consumed by 01_clip_inputs.py roads layer)
"""
import json
import osmium
from config import RAW

BBOX = (75.30, 11.38, 76.52, 12.90)   # lon_min, lat_min, lon_max, lat_max (both districts + margin)
KEEP = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
        "track", "living_street", "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"}

elements = []
fp = osmium.FileProcessor(str(RAW / "osm" / "southern-zone-latest.osm.pbf"), osmium.osm.NODE | osmium.osm.WAY) \
    .with_locations() \
    .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY)).with_filter(osmium.filter.KeyFilter("highway"))
for w in fp:
    if not w.is_way():
        continue
    hw = w.tags.get("highway")
    if hw not in KEEP:
        continue
    try:
        pts = [(n.lon, n.lat) for n in w.nodes]
    except osmium.InvalidLocationError:
        continue
    if not any(BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3] for x, y in pts):
        continue
    elements.append({"type": "way", "id": w.id, "tags": {"highway": hw},
                     "geometry": [{"lon": x, "lat": y} for x, y in pts]})
with open(RAW / "osm" / "roads_osm_extract.json", "w", encoding="utf-8") as f:
    json.dump({"source": "Geofabrik southern-zone-latest.osm.pbf", "elements": elements}, f)
print("ways kept:", len(elements))
