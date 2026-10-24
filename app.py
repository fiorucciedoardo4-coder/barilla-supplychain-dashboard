import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Funzioni di calcolo scritte nel file backend.py
from backend import (ARTICOLI, UNITA, genera_dati, calcola_previsioni,
                     calcola_metriche, scegli_metodo, dimensiona_scorte)

st.set_page_config(page_title="Previsione domanda e scorte - Barilla", layout="wide")


# Il dataset viene rigenerato solo quando cambiano i parametri della domanda
@st.cache_data
def carica_dati(variabilita, pendenza_trend, ampiezza_stag):
    return genera_dati(variabilita, pendenza_trend, ampiezza_stag)


# Esegue previsioni, metriche, scelta del metodo e scorte per un articolo.
# Restituisce un dizionario con tutti i risultati.
def analizza_articolo(serie, tipo, finestra_sma, alpha_ses, lead_time,
                      servizio, intervallo, margine):
    sma, ses = calcola_previsioni(serie, finestra_sma, alpha_ses)
    mae_sma, rmse_sma, perc_sma = calcola_metriche(serie, sma)
    mae_ses, rmse_ses, perc_ses = calcola_metriche(serie, ses)
    metodo = scegli_metodo(rmse_sma, rmse_ses)

    # le scorte si calcolano con la previsione del metodo migliore
    if metodo == "SMA":
        previsione = sma[-1]
        rmse = rmse_sma
    else:
        previsione = ses[-1]
        rmse = rmse_ses

    scorte = dimensiona_scorte(serie, tipo, previsione, rmse, lead_time,
                               servizio, intervallo, margine)

    return {
        "sma": sma, "ses": ses,
        "metriche_sma": (mae_sma, rmse_sma, perc_sma),
        "metriche_ses": (mae_ses, rmse_ses, perc_ses),
        "metodo": metodo, "previsione": previsione, "scorte": scorte,
    }


# Valori di partenza dei parametri (scenario base)
BASE = {
    "variabilita": 1.0, "pendenza_trend": 8, "ampiezza_stag": 250,
    "finestra_sma": 4, "alpha_ses": 0.30,
    "lead_time": 2, "servizio": 0.95, "intervallo": 2, "margine": 0.10,
}

# ---------------------------------------------------------
# SIDEBAR: parametri modificabili
# ---------------------------------------------------------
st.sidebar.title("Parametri")

articolo_scelto = st.sidebar.selectbox("Articolo", options=list(ARTICOLI.keys()))
tipo = ARTICOLI[articolo_scelto]

st.sidebar.subheader("Domanda")
variabilita = st.sidebar.slider("Variabilità (moltiplica il rumore)", 0.5, 3.0, BASE["variabilita"], 0.1)
pendenza_trend = st.sidebar.slider("Trend: crescita settimanale (" + UNITA + ")", 0, 20, BASE["pendenza_trend"])
ampiezza_stag = st.sidebar.slider("Stagionalità: ampiezza del picco (" + UNITA + ")", 0, 450, BASE["ampiezza_stag"], 10)

st.sidebar.subheader("Metodi di previsione")
finestra_sma = st.sidebar.slider("Finestra SMA (settimane)", 2, 12, BASE["finestra_sma"])
alpha_ses = st.sidebar.slider("Alpha SES", 0.05, 0.95, BASE["alpha_ses"], 0.05)

st.sidebar.subheader("Approvvigionamento")
lead_time = st.sidebar.slider("Lead time (settimane)", 1, 8, BASE["lead_time"])
servizio = st.sidebar.slider("Livello di servizio", 0.80, 0.99, BASE["servizio"], 0.01)
intervallo = st.sidebar.slider("Intervallo tra due ordini (settimane)", 1, 8, BASE["intervallo"])
margine = st.sidebar.slider("Capacità produttiva oltre la domanda media (%)", 0, 50, int(BASE["margine"] * 100), 5) / 100

# ---------------------------------------------------------
# CALCOLI
# ---------------------------------------------------------
df = carica_dati(variabilita, pendenza_trend, ampiezza_stag)
serie = df[articolo_scelto]
ris = analizza_articolo(serie, tipo, finestra_sma, alpha_ses, lead_time,
                        servizio, intervallo, margine)

# stesso articolo con i parametri dello scenario base, per il confronto
df_base = carica_dati(BASE["variabilita"], BASE["pendenza_trend"], BASE["ampiezza_stag"])
ris_base = analizza_articolo(df_base[articolo_scelto], tipo, BASE["finestra_sma"],
                             BASE["alpha_ses"], BASE["lead_time"], BASE["servizio"],
                             BASE["intervallo"], BASE["margine"])

st.title("Previsione della domanda e pianificazione delle scorte")
st.caption("Project work - dati settimanali generati per quattro articoli ispirati alla gamma Barilla. Unità di misura: " + UNITA + ".")

tab1, tab2, tab3 = st.tabs(["Previsioni", "Scorte", "Riepilogo articoli"])

# ---------------------------------------------------------
# TAB 1: previsioni e accuratezza
# ---------------------------------------------------------
with tab1:
    st.subheader(articolo_scelto)

    mae_sma, rmse_sma, perc_sma = ris["metriche_sma"]
    mae_ses, rmse_ses, perc_ses = ris["metriche_ses"]

    tabella = pd.DataFrame({
        "Metodo": ["SMA (media mobile)", "SES (attenuazione esponenziale)"],
        "MAE": [round(mae_sma, 1), round(mae_ses, 1)],
        "RMSE": [round(rmse_sma, 1), round(rmse_ses, 1)],
        "MAE % sulla domanda media": [round(perc_sma, 1), round(perc_ses, 1)],
    })
    st.dataframe(tabella, hide_index=True)

    st.write(f"Metodo scelto (RMSE più basso): **{ris['metodo']}**. "
             f"Previsione per la prossima settimana: {ris['previsione']:.0f} {UNITA}.")
    if tipo == "Erratico":
        st.write("Con la domanda erratica entrambi i metodi sbagliano molto: "
                 "l'errore è più grande della domanda media.")
    if tipo == "Trend":
        st.write("Con il trend entrambi i metodi restano in ritardo rispetto alla domanda reale (sottostima).")

    # una settimana in piu' per mostrare la previsione futura
    date_prev = list(df["Data"]) + [df["Data"].iloc[-1] + pd.Timedelta(weeks=1)]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Data"], serie, label="Domanda reale", color="black")
    ax.plot(date_prev, ris["sma"], label=f"SMA ({finestra_sma} settimane)", linestyle="--")
    ax.plot(date_prev, ris["ses"], label=f"SES (alpha = {alpha_ses:.2f})", linestyle=":")
    ax.set_xlabel("Settimana")
    ax.set_ylabel("Domanda (" + UNITA + ")")
    ax.set_title("Domanda storica e prevista")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: scorte
# ---------------------------------------------------------
with tab2:
    st.subheader(f"Scorte per {articolo_scelto}")
    scorte = ris["scorte"]
    scorte_base = ris_base["scorte"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pipeline", f"{scorte['Pipeline Stock']:.0f}")
    c2.metric("Sicurezza", f"{scorte['Safety Stock']:.0f}")
    c3.metric("Ciclica", f"{scorte['Cycle Stock']:.0f}")
    c4.metric("Anticipata", f"{scorte['Anticipation Stock']:.0f}")
    c5.metric("Totale", f"{scorte['Scorta Totale']:.0f}",
              delta=f"{scorte['Scorta Totale'] - scorte_base['Scorta Totale']:.0f} rispetto allo scenario base",
              delta_color="off")
    st.caption("Valori in " + UNITA + ".")

    # confronto scenario base / scenario attuale
    componenti = ["Pipeline Stock", "Safety Stock", "Cycle Stock", "Anticipation Stock"]
    valori_base = []
    valori_attuali = []
    for c in componenti:
        valori_base.append(scorte_base[c])
        valori_attuali.append(scorte[c])

    posizioni = range(len(componenti))
    larghezza = 0.4
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar([p - larghezza / 2 for p in posizioni], valori_base, larghezza, label="Scenario base")
    ax2.bar([p + larghezza / 2 for p in posizioni], valori_attuali, larghezza, label="Scenario attuale")
    ax2.set_xticks(list(posizioni))
    ax2.set_xticklabels(componenti)
    ax2.set_ylabel("Quantità (" + UNITA + ")")
    ax2.set_title("Composizione delle scorte")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
        "- **Pipeline** = previsione settimanale × lead time\n"
        "- **Sicurezza** = z × RMSE del metodo scelto × √lead time\n"
        "- **Ciclica** = lotto / 2, con lotto = previsione × intervallo tra due ordini\n"
        "- **Anticipata** = solo per l'articolo stagionale: domanda del profilo stagionale "
        "che supera la capacità produttiva, da produrre prima del picco"
    )

# ---------------------------------------------------------
# TAB 3: metodo migliore per ogni articolo
# ---------------------------------------------------------
with tab3:
    st.subheader("Metodo più adatto per ogni articolo")
    righe = []
    for nome in ARTICOLI:
        r = analizza_articolo(df[nome], ARTICOLI[nome], finestra_sma, alpha_ses,
                              lead_time, servizio, intervallo, margine)
        righe.append({
            "Articolo": nome,
            "RMSE SMA": round(r["metriche_sma"][1], 1),
            "RMSE SES": round(r["metriche_ses"][1], 1),
            "Metodo scelto": r["metodo"],
            "Scorta totale (" + UNITA + ")": round(r["scorte"]["Scorta Totale"]),
        })
    st.dataframe(pd.DataFrame(righe), hide_index=True)
    st.caption("Calcolato con i parametri impostati nella barra laterale.")
