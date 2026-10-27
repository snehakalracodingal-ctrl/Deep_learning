import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
import reverse_geocoder
import geonamescache

st.title("Where Am I? Be a Location Detective")
st.write("Your computer can find where you are right now as two numbers: **latitude** and **longitude**. "
         "We show the town or city you are in. Can you guess the nearest OTHER big city and the state or region?")


gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
cities = [c for c in gc.get_cities().values() if c["population"] >= 500_000]


st.subheader("Step 1: Find where you are")
st.write("Press the small target button below and allow your browser to share your location. No GPS? Use the two sliders instead.")
gps = streamlit_geolocation()
lat = st.slider("Latitude (moves you North / South)", -90.0, 90.0, 0.0)
lon = st.slider("Longitude (moves you East / West)", -180.0, 180.0, 0.0)
if gps["latitude"] is not None:
    lat, lon = gps["latitude"], gps["longitude"]           # the real location wins over the sliders
first, second = st.columns(2)
first.metric("Your latitude", f"{lat:.2f}")
second.metric("Your longitude", f"{lon:.2f}")
st.caption("This app does not save your location.")


def gap(city):
    return (city["latitude"] - lat) ** 2 + (city["longitude"] - lon) ** 2

nearby = sorted(cities, key=gap)[1:6]                       
around = [(lat + up, lon + side) for up in (-5, 0, 5) for side in (-5, 0, 5)]     # 9 spots: you and 8 spots around you
places = reverse_geocoder.search(around, mode=1)           
answer_city = places[4]["name"]                   
answer_country = countries.get(places[4]["cc"], {}).get("name", places[4]["cc"])
answer_state = places[4]["admin1"] or "No state here"
big_city = nearby[0]                               
if gps["latitude"] is not None or lat != 0 or lon != 0:    # after you press the button (or move a slider)
    st.success(f"You are in {answer_city}, {answer_country}.")

st.subheader("Step 2: Make your guesses")
city_guess = st.selectbox("Which OTHER big city is nearest to you?", sorted({city["name"] for city in nearby}), index=None, placeholder="Choose a city")
state_guess = st.selectbox("Which state or region are you in?", sorted({place["admin1"] or "No state here" for place in places}), index=None, placeholder="Choose a state or region")
if None in (city_guess, state_guess):
    st.stop()                                            


st.subheader("Step 3: The answers")
score = 0
for label, guess, answer in [("Nearest other big city", city_guess, big_city["name"]), ("State or region", state_guess, answer_state)]:
    if guess == answer:
        st.success(f"{label}: {answer}. Right!")
        score = score + 1
    else:
        st.error(f"{label}: it is {answer}, not {guess}.")
st.write(f"You got **{score} out of 2** right.")

# Step 6: you and your nearest city on the map
m = folium.Map(min_zoom=2, max_bounds=True, tiles=None)                     # one world only, no repeats
folium.TileLayer("Esri.WorldImagery", no_wrap=True, bounds=[[-90, -180], [90, 180]]).add_to(m)
folium.Marker([lat, lon], icon=folium.Icon(color="red", icon="user", prefix="fa"), tooltip="You are here").add_to(m)
folium.CircleMarker([big_city["latitude"], big_city["longitude"]], radius=8, color="yellow", fill=True,
                    tooltip=folium.Tooltip(big_city["name"], permanent=True)).add_to(m)
folium.PolyLine([[lat, lon], [big_city["latitude"], big_city["longitude"]]], color="yellow").add_to(m)
m.fit_bounds(m.get_bounds(), max_zoom=10)
st_folium(m, width=700, height=350, returned_objects=[])
