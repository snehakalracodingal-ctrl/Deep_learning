import streamlit as st
import folium
from streamlit_folium import st_folium
import geonamescache

st.title("World Explorer: Fly Anywhere with 2 Numbers!")
st.write("Your mission: land near a big city on **all 6 continents** and collect a passport stamp from each. "
         "Move the sliders, or just click on the map to drop your pin!")


gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
cities = [c for c in gc.get_cities().values() if c["population"] >= 1_000_000]
continents = {"AS": "Asia", "AF": "Africa", "EU": "Europe",
              "NA": "North America", "SA": "South America", "OC": "Oceania"}


if "stamps" not in st.session_state:
    st.session_state.stamps = {}                     
pin_start = st.session_state.get("pin_start", (0.0, 0.0))   # where the last map click put the pin


lat = st.slider("Latitude (moves you North / South)", -90.0, 90.0, pin_start[0])
lon = st.slider("Longitude (moves you East / West)", -180.0, 180.0, pin_start[1])


def gap(city):
    return (city["latitude"] - lat) ** 2 + (city["longitude"] - lon) ** 2

near = min(cities, key=gap)
country = countries[near["countrycode"]]
on_land = gap(near) ** 0.5 < 8                        

if on_land:
    st.success(f"You landed near {near['name']}, {country['name']}!")
    if st.button("Stamp my passport"):
        st.session_state.stamps[continents[country["continentcode"]]] = f"{near['name']}, {country['name']}"
else:
    st.warning("Splash! Open ocean or empty wilderness. Try different numbers!")


m = folium.Map(location=[20, 0], zoom_start=2, min_zoom=1, max_bounds=True, tiles=None)   # one world only, no repeats
folium.TileLayer("Esri.WorldImagery", no_wrap=True, bounds=[[-90, -180], [90, 180]]).add_to(m)
folium.PolyLine([[0, -180], [0, 180]], color="yellow", tooltip="Equator").add_to(m)
folium.PolyLine([[-90, 0], [90, 0]], color="cyan", tooltip="Prime Meridian").add_to(m)
pin_layer = folium.FeatureGroup()
folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="plane", prefix="fa")).add_to(pin_layer)
map_click = st_folium(m, height=420, feature_group_to_add=pin_layer, returned_objects=["last_clicked"])["last_clicked"]

if map_click and map_click != st.session_state.get("last_click"):
    st.session_state.last_click = map_click
    new_lat = round(map_click["lat"], 2)
    new_lon = round((map_click["lng"] + 180) % 360 - 180, 2)   # keep longitude between -180 and 180
    st.session_state.pin_start = (new_lat, new_lon)
    st.rerun()

st.progress(len(st.session_state.stamps) / 6, text=f"{len(st.session_state.stamps)} of 6 continents")
st.dataframe([{"Continent": k, "Landed near": v} for k, v in st.session_state.stamps.items()])
if len(st.session_state.stamps) == 6:
    st.balloons()
    st.success("You travelled the whole world using only numbers! Every dot on a map is just "
               "latitude + longitude. Next lesson: make those dots show how BIG each city is.")