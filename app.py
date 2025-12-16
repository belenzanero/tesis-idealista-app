# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 14:03:18 2025

@author: bzane
"""

import os
import requests
import base64
import pandas as pd
import numpy as np
import folium
import polyline
import urllib.parse
import streamlit as st

# Opcional: para mostrar folium bien integrado
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False


# ------------------------------------------------------------
# CONFIG STREAMLIT
# ------------------------------------------------------------
st.set_page_config(page_title="Buscador de Alojamiento", layout="wide")


# ------------------------------------------------------------
# SESSION STATE (para que los resultados queden fijos)
# ------------------------------------------------------------
if "result_df_top" not in st.session_state:
    st.session_state["result_df_top"] = None
if "result_mapa" not in st.session_state:
    st.session_state["result_mapa"] = None
if "result_meta" not in st.session_state:
    st.session_state["result_meta"] = None


# ------------------------------------------------------------
# CLAVES API (st.secrets -> env -> fallback hardcode)
# ------------------------------------------------------------
def _get_secret(name: str, default: str = "") -> str:
    # st.secrets a veces no existe si no hay secrets.toml
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.getenv(name, default).strip()

# Si tenés secrets.toml, podés borrar estos defaults hardcodeados y dejar default=""
IDEALISTA_API_KEY = st.secrets["IDEALISTA_API_KEY"].strip()
IDEALISTA_API_SECRET = st.secrets["IDEALISTA_API_SECRET"].strip()
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"].strip()

if not GOOGLE_API_KEY:
    st.error("Falta GOOGLE_API_KEY. Configurala en .streamlit/secrets.toml o como variable de entorno.")
    st.stop()

if not IDEALISTA_API_KEY or not IDEALISTA_API_SECRET:
    st.error("Faltan IDEALISTA_API_KEY / IDEALISTA_API_SECRET.")
    st.stop()


# ------------------------------------------------------------
# DATOS DE CIUDADES Y UNIVERSIDADES
# ------------------------------------------------------------
ciudades = {
    "madrid": {
        "centro": (40.4203, -3.7058),  # Gran Vía
        "universidades": [
            {"nombre": "Universidad Autónoma de Madrid", "lat": 40.5443, "lon": -3.6969},
            {"nombre": "Universidad Complutense de Madrid", "lat": 40.4469, "lon": -3.7289},
            {"nombre": "Universidad Politécnica de Madrid", "lat": 40.4521, "lon": -3.7286},
            {"nombre": "Universidad de Alcalá", "lat": 40.5112, "lon": -3.3495},
            {"nombre": "Universidad Rey Juan Carlos - Móstoles", "lat": 40.3333, "lon": -3.8653},
            {"nombre": "Universidad de Castilla-La Mancha (Madrid)", "lat": 40.4078, "lon": -3.6964},
            {"nombre": "Universidad de Navarra (Campus Madrid)", "lat": 40.4322, "lon": -3.6853},
            {"nombre": "Universidad Pontificia Comillas", "lat": 40.4339, "lon": -3.7147},
            {"nombre": "Universidad San Pablo CEU", "lat": 40.4463, "lon": -3.6885},
            {"nombre": "Universidad Europea de Madrid", "lat": 40.3936, "lon": -3.9197},
            {"nombre": "Universidad Francisco de Vitoria", "lat": 40.4194, "lon": -3.8903},
            {"nombre": "Universidad Antonio de Nebrija", "lat": 40.4301, "lon": -3.7176},
            {"nombre": "Universidad Camilo José Cela", "lat": 40.3958, "lon": -3.6525},
            {"nombre": "Universidad Villanueva", "lat": 40.4485, "lon": -3.6895},
            {"nombre": "Universidad a Distancia de Madrid (UDIMA)", "lat": 40.5764, "lon": -3.6331},
        ]
    },
    "barcelona": {
        "centro": (41.3870, 2.1701),
        "universidades": [
            {"nombre": "Universidad de Barcelona (UB)", "lat": 41.3859, "lon": 2.1687},
            {"nombre": "Universidad Autónoma de Barcelona (UAB)", "lat": 41.5005, "lon": 2.1072},
            {"nombre": "Universidad Politécnica de Cataluña (UPC)", "lat": 41.3878, "lon": 2.1136},
            {"nombre": "Universidad Pompeu Fabra (UPF)", "lat": 41.3890, "lon": 2.1900},
            {"nombre": "Universitat Oberta de Catalunya (UOC)", "lat": 41.4036, "lon": 2.1946},
            {"nombre": "Universidad Ramon Llull (URL)", "lat": 41.3993, "lon": 2.1475},
            {"nombre": "Universidad Internacional de Cataluña (UIC)", "lat": 41.4011, "lon": 2.1418},
            {"nombre": "Universidad Abat Oliba CEU (UAO CEU)", "lat": 41.4094, "lon": 2.1375},
            {"nombre": "Universidad de Vic - Universidad Central de Cataluña (UVic-UCC)", "lat": 41.9303, "lon": 2.2542},
        ]
    },
    "valencia": {
        "centro": (39.4699, -0.3763),
        "universidades": [
            {"nombre": "Universitat de València (UV)", "lat": 39.4780, "lon": -0.3416},
            {"nombre": "Universitat Politècnica de València (UPV)", "lat": 39.4811, "lon": -0.3455},
            {"nombre": "Universidad Católica de Valencia San Vicente Mártir (UCV)", "lat": 39.4713, "lon": -0.3799},
            {"nombre": "Universidad Cardenal Herrera CEU (CEU UCH)", "lat": 39.4284, "lon": -0.3843},
            {"nombre": "Universidad Europea de Valencia", "lat": 39.4721, "lon": -0.3776},
            {"nombre": "Universidad Internacional de Valencia (VIU)", "lat": 39.4719, "lon": -0.3762},
            {"nombre": "Florida Universitària", "lat": 39.3962, "lon": -0.4494},
            {"nombre": "EDEM Escuela de Empresarios", "lat": 39.4553, "lon": -0.3196},
        ]
    }
}


# ------------------------------------------------------------
# IDEALISTA TOKEN
# ------------------------------------------------------------
def _request_token(base_url, api_key, api_secret):
    cred = f"{api_key}:{api_secret}"
    b64 = base64.b64encode(cred.encode()).decode()
    headers = {
        "Authorization": f"Basic {b64}",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Accept": "application/json",
    }
    data = {"grant_type": "client_credentials", "scope": "read"}
    url = f"{base_url}/oauth/token"
    try:
        r = requests.post(url, headers=headers, data=data, timeout=25)
    except Exception as e:
        return None, {"status": "network_error", "payload": str(e)}

    try:
        payload = r.json()
    except Exception:
        payload = {"raw": r.text[:800]}

    if r.status_code == 200 and payload.get("access_token"):
        return payload["access_token"], None
    return None, {"status": r.status_code, "payload": payload}

@st.cache_data(ttl=50 * 60, show_spinner=False)
@st.cache_data(ttl=50 * 60, show_spinner=False)
def get_token_cached(api_key, api_secret):
    token, err = _request_token("https://api.idealista.com", api_key, api_secret)
    if token:
        return token, "prod"
    raise RuntimeError(f"No se pudo obtener token Idealista en producción. Detalle: {err}")


# ------------------------------------------------------------
# GOOGLE APIs
# ------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def obtener_tiempo_google(orig, dest, modo, api_key):
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {"origins": orig, "destinations": dest, "mode": modo, "key": api_key}
    r = requests.get(url, params=params, timeout=25)
    data = r.json()
    el = data["rows"][0]["elements"][0]
    if el.get("status") != "OK":
        return float("inf")
    return el["duration"]["value"] / 60

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def obtener_ruta_directions(origen, destino, modo, api_key):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {"origin": origen, "destination": destino, "mode": modo, "key": api_key}
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        return []
    data = r.json()
    if not data.get("routes"):
        return []
    puntos_codificados = data["routes"][0]["overview_polyline"]["points"]
    return polyline.decode(puntos_codificados)

@st.cache_data(ttl=30 * 24 * 60 * 60, show_spinner=False)
def places_text_search_new(query, centro_ciudad, radio_m, api_key):
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.id",
    }
    body = {"textQuery": query}
    if centro_ciudad:
        body["locationBias"] = {
            "circle": {
                "center": {"latitude": centro_ciudad[0], "longitude": centro_ciudad[1]},
                "radius": radio_m,
            }
        }
    r = requests.post(url, json=body, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data.get("places"):
        return None
    return data["places"][0]

@st.cache_data(ttl=30 * 24 * 60 * 60, show_spinner=False)
def geocoding_fallback(nombre, ciudad, centro_ciudad, radio_m, api_key):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    candidatos = [
        f"{nombre}, {ciudad}, España",
        f"Universidad {nombre}, {ciudad}, España",
        f"{nombre} campus {ciudad}, España",
        f"{nombre}, España",
        nombre,
    ]
    for direccion in candidatos:
        params = {
            "address": direccion,
            "key": api_key,
            "language": "es",
            "region": "es",
            "components": "country:ES",
        }
        if centro_ciudad:
            params["location"] = f"{centro_ciudad[0]},{centro_ciudad[1]}"
            params["radius"] = str(radio_m)
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
        if data.get("status") == "OK" and data.get("results"):
            first = data["results"][0]
            loc = first["geometry"]["location"]
            return {
                "id": first.get("place_id"),
                "formattedAddress": first.get("formatted_address"),
                "location": {"latitude": loc["lat"], "longitude": loc["lng"]},
                "displayName": {"text": first.get("formatted_address", nombre)},
            }
    return None

def buscar_facultad(nombre, ciudad, centro_ciudad):
    radio_m = 30000
    queries = [
        f"{nombre} {ciudad} España",
        f"Universidad {nombre} {ciudad} España",
        f"{nombre} campus {ciudad} España",
        f"{nombre}, {ciudad}, España",
    ]
    for q in queries:
        p = places_text_search_new(q, centro_ciudad, radio_m, GOOGLE_API_KEY)
        if p:
            return p, "places_new"
    p = geocoding_fallback(nombre, ciudad, centro_ciudad, radio_m, GOOGLE_API_KEY)
    if p:
        return p, "geocoding"
    return None, None


# ------------------------------------------------------------
# OTROS HELPERS
# ------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

def generar_link_maps(lat_origen, lon_origen, lat_destino, lon_destino, modo):
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={lat_origen},{lon_origen}"
        f"&destination={lat_destino},{lon_destino}"
        f"&travelmode={modo}"
    )

def buscar_idealista(token, base_search, lat, lon, presupuesto, max_items=50, num_page=1):
    url = f"{base_search}/3.5/es/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "Accept": "application/json",
    }
    data = {
        "center": f"{lat},{lon}",
        "distance": "5000",
        "propertyType": "homes",
        "operation": "rent",
        "maxPrice": str(int(presupuesto)),
        "maxItems": str(max_items),
        "numPage": str(num_page),
    }
    r = requests.post(url, headers=headers, data=data, timeout=25)
    try:
        payload = r.json()
    except Exception:
        raise RuntimeError(f"Idealista no devolvió JSON. status={r.status_code}. Respuesta:\n{r.text[:800]}")
    if r.status_code != 200 or "elementList" not in payload:
        raise RuntimeError(f"Error Idealista. status={r.status_code}. payload={payload}")
    elementos = payload.get("elementList", [])
    return pd.DataFrame(elementos)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Buscador de Alojamiento para Estudiantes")

with st.sidebar:
    st.header("Inputs")

    ciudad_seleccionada = st.selectbox(
        "Ciudad",
        options=list(ciudades.keys()),
        format_func=lambda x: x.capitalize()
    )

    info_ciudad = ciudades[ciudad_seleccionada]
    universidades = info_ciudad["universidades"]
    centro_ciudad = info_ciudad["centro"]
    lat_centro, lon_centro = centro_ciudad
    nombre_centro = "Gran Vía" if ciudad_seleccionada == "madrid" else f"Centro de {ciudad_seleccionada.capitalize()}"

    modo_uni = st.radio(
        "Cómo indicar universidad",
        options=["Elegir de lista", "Escribir nombre (Google Maps)"],
        index=1
    )

    lat_uni = lon_uni = None
    nombre_uni = None
    place_id = None
    direccion_uni = None

    if modo_uni == "Elegir de lista":
        uni_nombre = st.selectbox("Universidad", [u["nombre"] for u in universidades])
        u = next(x for x in universidades if x["nombre"] == uni_nombre)
        lat_uni, lon_uni, nombre_uni = u["lat"], u["lon"], u["nombre"]
        direccion_uni = u["nombre"]
        # limpiamos sesión por si venía de una búsqueda
        for k in ["uni_lat", "uni_lon", "uni_nombre", "uni_direccion", "uni_place_id", "uni_fuente"]:
            st.session_state.pop(k, None)
    else:
        texto_facultad = st.text_input("Escribí el nombre de tu facultad", value="Universidad Rey Juan Carlos Vicálvaro")
        buscar = st.button("Buscar en Google Maps")
        if buscar and texto_facultad.strip():
            p, fuente = buscar_facultad(texto_facultad.strip(), ciudad_seleccionada.capitalize(), centro_ciudad)
            if not p:
                st.error("No pude encontrar coordenadas con Google.")
            else:
                lat_uni = p["location"]["latitude"]
                lon_uni = p["location"]["longitude"]
                place_id = p.get("id")
                nombre_uni = p.get("displayName", {}).get("text", texto_facultad)
                direccion_uni = p.get("formattedAddress", nombre_uni)

                st.session_state["uni_lat"] = lat_uni
                st.session_state["uni_lon"] = lon_uni
                st.session_state["uni_nombre"] = nombre_uni
                st.session_state["uni_direccion"] = direccion_uni
                st.session_state["uni_place_id"] = place_id
                st.session_state["uni_fuente"] = fuente

        if "uni_lat" in st.session_state:
            lat_uni = st.session_state["uni_lat"]
            lon_uni = st.session_state["uni_lon"]
            nombre_uni = st.session_state["uni_nombre"]
            direccion_uni = st.session_state["uni_direccion"]
            place_id = st.session_state.get("uni_place_id")

            st.success("Ubicación encontrada")
            st.write(direccion_uni)
            st.write(f"Coordenadas: {lat_uni}, {lon_uni}")
            if place_id:
                st.write(f"Link: https://www.google.com/maps/place/?q=place_id:{place_id}")

    st.divider()

    presupuesto = st.number_input("Precio máximo (€)", min_value=200, max_value=6000, value=1000, step=50)

    tipos_disponibles = ["chalet", "duplex", "flat", "penthouse", "studio"]
    tipo_propiedad = st.multiselect("Tipos de propiedad (Idealista)", tipos_disponibles, default=[])

    tamano_min = st.text_input("Tamaño mínimo en m² (opcional)", value="")
    rooms_min = st.selectbox("Mínimo habitaciones", [0, 1, 2, 3, 4], index=0)
    banos_min = st.selectbox("Mínimo baños", [0, 1, 2, 3], index=0)
    exterior = st.checkbox("Exterior", value=False)

    preferencia = st.radio(
        "Prioridad",
        options=["Cerca universidad", f"Cerca {nombre_centro}", "Equilibrio"],
        index=0
    )

    medio_transporte = st.selectbox("Medio de transporte", ["transit", "walking", "driving"], index=0)
    tiempo_max = st.slider("Tiempo máximo a la universidad (min)", min_value=10, max_value=120, value=60, step=5)

    ejecutar = st.button("Buscar alojamiento", type="primary")

    # opcional: limpiar resultados guardados
    if st.button("Limpiar resultados"):
        st.session_state["result_df_top"] = None
        st.session_state["result_mapa"] = None
        st.session_state["result_meta"] = None
        st.rerun()


# ------------------------------------------------------------
# EJECUCIÓN
# ------------------------------------------------------------
# si no apretaste buscar:
# - si NO hay resultados guardados -> mensaje
# - si HAY resultados guardados -> los mostramos abajo y NO cortamos la app
if not ejecutar:
    if st.session_state["result_df_top"] is None:
        st.info("Completá los inputs y tocá Buscar alojamiento.")
        st.stop()


# si apretaste buscar, calculamos y guardamos resultados
if ejecutar:
    if lat_uni is None or lon_uni is None:
        st.error("Primero definí la universidad (coordenadas).")
        st.stop()

    with st.spinner("Obteniendo token Idealista..."):
        try:
            token, entorno = get_token_cached(IDEALISTA_API_KEY, IDEALISTA_API_SECRET)
        except Exception as e:
            st.error(str(e))
            st.stop()

    base_search = "https://api-sandbox.idealista.com" if entorno == "sandbox" else "https://api.idealista.com"

    with st.spinner("Buscando propiedades en Idealista..."):
        try:
            df = buscar_idealista(token, base_search, lat_uni, lon_uni, presupuesto, max_items=50, num_page=1)
        except Exception as e:
            st.error(str(e))
            st.stop()

    if df.empty:
        st.warning("No se encontraron propiedades con esos criterios.")
        st.stop()

    # asegurar columnas
    for c in ["latitude", "longitude"]:
        if c not in df.columns:
            st.error(f"La API no devolvió la columna {c}. Columnas: {list(df.columns)}")
            st.stop()

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    defaults = {
        "rooms": 0,
        "bathrooms": 0,
        "size": 0,
        "exterior": False,
        "hasLift": False,
        "address": "No disponible",
        "url": "",
        "price": 0,
        "floor": "No disponible",
        "detailedType": "",
    }
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default
        df[c] = df[c].fillna(default)

    def _tipo_str(x):
        if isinstance(x, dict):
            return (x.get("subTypology") or x.get("typology") or "").lower()
        return str(x).lower()

    df["detailedType_str"] = df["detailedType"].apply(_tipo_str)

    # filtros
    df = df[(df["rooms"] >= rooms_min) & (df["bathrooms"] >= banos_min)]
    if tamano_min.strip():
        try:
            df = df[df["size"] >= float(tamano_min)]
        except ValueError:
            pass
    if tipo_propiedad:
        df = df[df["detailedType_str"].apply(lambda s: any(tp in s for tp in tipo_propiedad))]
    if exterior:
        df = df[df["exterior"] == True]

    if df.empty:
        st.warning("Después de aplicar filtros no quedó ninguna propiedad.")
        st.stop()

    with st.spinner("Calculando tiempos con Google..."):
        df["tiempo_uni"] = df.apply(
            lambda row: obtener_tiempo_google(
                f"{row['latitude']},{row['longitude']}",
                f"{lat_uni},{lon_uni}",
                medio_transporte,
                GOOGLE_API_KEY
            ),
            axis=1
        )

        df["tiempo_centro"] = df.apply(
            lambda row: obtener_tiempo_google(
                f"{row['latitude']},{row['longitude']}",
                f"{lat_centro},{lon_centro}",
                medio_transporte,
                GOOGLE_API_KEY
            ),
            axis=1
        )

    df["distancia_km_uni"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"], lat_uni, lon_uni), axis=1)
    df["distancia_km_centro"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"], lat_centro, lon_centro), axis=1)
    df["tiempo_uni_limpio"] = df["tiempo_uni"].replace(np.inf, np.nan)

    # ranking
    if preferencia == "Cerca universidad":
        df_filtrado = df[df["tiempo_uni_limpio"] <= tiempo_max].sort_values("tiempo_uni_limpio")
    elif preferencia == f"Cerca {nombre_centro}":
        df_filtrado = df.sort_values("tiempo_centro")
    else:
        df_filtrado = df[df["tiempo_uni_limpio"] <= tiempo_max].copy()
        df_filtrado["ponderado"] = 0.5 * df_filtrado["tiempo_uni_limpio"] + 0.5 * df_filtrado["tiempo_centro"]
        df_filtrado = df_filtrado.sort_values("ponderado")

    if df_filtrado.empty:
        df_top = df.sort_values("tiempo_uni_limpio").head(3).reset_index(drop=True)
        st.warning("Ninguna propiedad cumple el tiempo máximo. Mostrando las 3 más cercanas por tiempo a la uni.")
    else:
        df_top = df_filtrado.head(3).reset_index(drop=True)

    df_top["id"] = df_top.index + 1

    # armar mapa
    mapa = folium.Map(location=[lat_uni, lon_uni], zoom_start=13)
    folium.Marker([lat_uni, lon_uni], popup=nombre_uni, icon=folium.Icon(color="blue")).add_to(mapa)

    colores = ["red", "green", "purple"]

    for i, fila in df_top.iterrows():
        if preferencia in ["Cerca universidad", "Equilibrio"]:
            lat_destino, lon_destino = lat_uni, lon_uni
        else:
            lat_destino, lon_destino = lat_centro, lon_centro

        popup = f"#{int(fila['id'])} - €{fila['price']}<br>{fila.get('address', '')}<br>{fila['tiempo_uni']:.1f} min"
        folium.Marker(
            [fila["latitude"], fila["longitude"]],
            popup=popup,
            icon=folium.DivIcon(html=f"<div style='font-size: 10pt; color:{colores[i]}'>{int(fila['id'])}</div>")
        ).add_to(mapa)

        ruta = obtener_ruta_directions(
            f"{fila['latitude']},{fila['longitude']}",
            f"{lat_destino},{lon_destino}",
            medio_transporte,
            GOOGLE_API_KEY
        )
        if ruta:
            folium.PolyLine(locations=ruta, color=colores[i], weight=3, opacity=0.7).add_to(mapa)

    # GUARDAR RESULTADOS (esto es lo que evita que “desaparezcan”)
    st.session_state["result_df_top"] = df_top
    st.session_state["result_mapa"] = mapa
    st.session_state["result_meta"] = {
        "lat_uni": lat_uni,
        "lon_uni": lon_uni,
        "lat_centro": lat_centro,
        "lon_centro": lon_centro,
        "nombre_centro": nombre_centro,
        "medio_transporte": medio_transporte,
        "nombre_uni": nombre_uni,
    }


# ------------------------------------------------------------
# OUTPUTS (siempre mostramos los guardados)
# ------------------------------------------------------------
df_top_show = st.session_state["result_df_top"]
mapa_show = st.session_state["result_mapa"]
meta = st.session_state["result_meta"] or {}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Opciones sugeridas (top 3)")

    st.dataframe(
        df_top_show[["id", "price", "size", "rooms", "bathrooms", "address", "tiempo_uni", "tiempo_centro", "url"]],
        use_container_width=True
    )

    for _, fila in df_top_show.iterrows():
        st.write(f"Propiedad #{int(fila['id'])}")
        st.write(f"Precio: €{fila['price']} | Tamaño: {fila['size']} m2 | Hab: {fila['rooms']} | Baños: {fila['bathrooms']}")
        st.write(f"Dirección: {fila.get('address', 'No disponible')}")
        st.write(f"Tiempo a universidad: {fila['tiempo_uni']:.1f} min ({fila['distancia_km_uni']:.2f} km)")
        st.write(f"Tiempo a {meta.get('nombre_centro', 'Centro')}: {fila['tiempo_centro']:.1f} min ({fila['distancia_km_centro']:.2f} km)")

        link_maps_uni = generar_link_maps(
            fila["latitude"], fila["longitude"],
            meta.get("lat_uni", 0), meta.get("lon_uni", 0),
            meta.get("medio_transporte", "transit")
        )
        link_maps_centro = generar_link_maps(
            fila["latitude"], fila["longitude"],
            meta.get("lat_centro", 0), meta.get("lon_centro", 0),
            meta.get("medio_transporte", "transit")
        )

        if fila.get("url"):
            st.write(f"Idealista: {fila['url']}")
        st.write(f"Maps a la uni: {link_maps_uni}")
        st.write(f"Maps al centro: {link_maps_centro}")
        st.divider()

with col2:
    st.subheader("Mapa")

    if HAS_ST_FOLIUM:
        st_folium(mapa_show, width=700, height=520)
    else:
        st.components.v1.html(mapa_show._repr_html_(), height=520, scrolling=True)

