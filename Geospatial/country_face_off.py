import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geonamescache

st.title("Country Face-Off: Who Has More People in Big Cities?")
st.write("Pick two countries and study the maps of their 10 biggest cities. Dot **size** = how many people live in the city. "
         "Make a guess, then let the data decide!")


gc = geonamescache.GeonamesCache()
countries = gc.get_countries()
by_country = {}
for c in gc.get_cities().values():
    if c["population"] >= 100_000:
        by_country.setdefault(c["countrycode"], []).append(c)

top10 = {code: sorted(v, key=lambda c: c["population"], reverse=True)[:10]
         for code, v in by_country.items() if len(v) >= 10}
names = {countries[code]["name"]: code for code in top10}

left, right = st.columns(2)
name_a = left.selectbox("Country A", sorted(names), index=None, placeholder="Choose a country")
name_b = right.selectbox("Country B", sorted(names), index=None, placeholder="Choose a country")
if name_a is None or name_b is None or name_a == name_b:
    st.info("Choose two different countries to see their maps!")
    st.stop()                                           

def total_people(name):                                 # all the people in a country's 10 biggest cities
    return sum(c["population"] for c in top10[names[name]])

for box, name in zip(st.columns(2), [name_a, name_b]):
    m = folium.Map(tiles="Esri.WorldImagery")
    for c in top10[names[name]]:
        colour = "red" if c["population"] >= 5_000_000 else "orange" if c["population"] >= 1_000_000 else "yellow"
        folium.CircleMarker([c["latitude"], c["longitude"]], radius=max(4, c["population"] ** 0.5 / 250),
                            color=colour, fill=True, fill_opacity=0.7).add_to(m)
    m.fit_bounds(m.get_bounds())                    
    with box:
        st.subheader(name)
        st_folium(m, height=300, key=name, returned_objects=[], use_container_width=True)
st.caption("Each dot is one of the 10 biggest cities. Red = over 5 million people, orange = over 1 million, yellow = under 1 million.")

guess = st.radio("Your guess: which country has MORE people living in its 10 biggest cities?", [name_a, name_b], index=None)
if guess is None:
    st.stop()                                           
winner = max([name_a, name_b], key=total_people)
if guess == winner:
    st.success(f"You were right! {winner} has more people in its 10 biggest cities.")
else:
    st.error(f"Surprise! {winner} has more people in its 10 biggest cities.")


for box, name in zip(st.columns(2), [name_a, name_b]):
    top = top10[names[name]]
    box.metric(f"{name}: people in its 10 biggest cities", f"{total_people(name):,}")
    box.metric(f"{top[0]['name']} (its biggest city) is this % of those people", f"{100 * top[0]['population'] / total_people(name):.0f}%")
    box.write(f"**{name}: how many people live in each of its 10 biggest cities**")
    box.bar_chart(pd.DataFrame({"City": [c["name"] for c in top], "People living there": [c["population"] for c in top]}),
                  x="City", y="People living there", x_label="City", y_label="People living there")
st.info("Try new pairs, then look for a pattern. Next lesson: colour whole countries, not just dots!")