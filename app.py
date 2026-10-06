import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Importazione della logica dal file di back-end
from backend import genera_dati, calcola_previsioni, calcola_metriche, dimensiona_scorte

# ---------------------------------------------------------
# Configurazione Pagina Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Barilla S.p.A. | Supply Chain & Demand Management",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializzazione dati stocastici con memorizzazione in cache
@st.cache_data
def get_dataset():
    return genera_dati()

df = get_dataset()

# ---------------------------------------------------------
# Palette Colori Corporate Barilla
# ---------------------------------------------------------
COLOR_REALE = "#0B2545"    # Deep Navy
COLOR_SMA = "#E89C35"      # Amber Grain
COLOR_SES = "#C8102E"      # Barilla Red

# ---------------------------------------------------------
# SIDEBAR: PARAMETRI E CONTROLLI INTERATTIVI
# ---------------------------------------------------------
st.sidebar.markdown("## 🔴 **BARILLA S.p.A.**")
st.sidebar.title("Parametri di Controllo")

articolo_scelto = st.sidebar.selectbox(
    "Seleziona Prodotto / Pattern",
    options=[
        "Spaghetti n.5 500g (Stazionario)",
        "Pasta di Legumi (Trend)",
        "Pesto alla Genovese (Stagionale)",
        "Panettone Pan di Stelle (Erratico)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Modelli Previsionali")
finestra_sma = st.sidebar.slider("Finestra SMA (settimane)", min_value=2, max_value=12, value=4)
alpha_ses = st.sidebar.slider("Alpha SES (reattività)", min_value=0.05, max_value=0.95, value=0.30, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Parametri Logistici")
lead_time = st.sidebar.slider("Lead Time Fornitore (settimane)", min_value=1, max_value=8, value=2)
servizio = st.sidebar.selectbox("Livello di Servizio", options=[0.90, 0.95, 0.99], index=1)

# ---------------------------------------------------------
# COMPUTAZIONE DINAMICA TRAMITE BACKEND
# ---------------------------------------------------------
serie_selezionata = df[articolo_scelto]
serie_sma, serie_ses = calcola_previsioni(serie_selezionata, finestra_sma, alpha_ses)
mae_sma, rmse_sma, mape_sma = calcola_metriche(serie_selezionata, serie_sma)
mae_ses, rmse_ses, mape_ses = calcola_metriche(serie_selezionata, serie_ses)
scorte = dimensiona_scorte(serie_selezionata, lead_time_settimane=lead_time, livello_servizio=servizio)

# ---------------------------------------------------------
# HEADER DASHBOARD
# ---------------------------------------------------------
st.title("Barilla S.p.A. — Supply Chain & Demand Management")
st.caption("Decision Support System (DSS) per il monitoraggio della domanda e l'ottimizzazione del capitale circolante")
st.markdown("---")

# ---------------------------------------------------------
# STRUTTURA A SCHEDE (TABS)
# ---------------------------------------------------------
tab_previsioni, tab_scorte, tab_dettagli = st.tabs([
    "📈 Analisi Previsionale & Accuratezza",
    "📦 Dimensionamento Funzionale Scorte",
    "ℹ️ Profilo Logistico Articolo"
])

# =========================================================
# TAB 1: PREVISIONI & ACCURATEZZA
# =========================================================
with tab_previsioni:
    st.markdown(f"### Valutazione Modelli: **{articolo_scelto}**")
    
    # Valutazione automatica del modello migliore
    if mape_ses < mape_sma:
        st.success(f"💡 **Modello Raccomandato: SES (Single Exponential Smoothing)** — Registra un errore percentuale medio (MAPE) inferiore di {abs(mape_sma - mape_ses):.1f}% rispetto a SMA.")
    else:
        st.success(f"💡 **Modello Raccomandato: SMA (Simple Moving Average)** — Garantisce una stima più solida smussando le fluttuazioni con un errore medio inferiore.")

    # KPI Metriche in Container con bordo
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="RMSE (Penalità Forti Scostamenti)", 
                value=f"{rmse_sma:.1f} (SMA)", 
                delta=f"SES: {rmse_ses:.1f}", 
                delta_color="inverse"
            )
        with col2:
            st.metric(
                label="MAE (Errore Medio Assoluto)", 
                value=f"{mae_sma:.1f} pallet", 
                delta=f"SES: {mae_ses:.1f} pallet", 
                delta_color="inverse"
            )
        with col3:
            st.metric(
                label="MAPE (Errore Percentuale)", 
                value=f"{mape_sma:.1f}%", 
                delta=f"SES: {mape_ses:.1f}%", 
                delta_color="inverse"
            )

    # Grafico Interattivo Serie Temporale
    fig_previsioni = go.Figure()
    fig_previsioni.add_trace(go.Scatter(
        x=df["Data"], y=serie_selezionata, 
        mode="lines", name="Domanda Reale", 
        line=dict(color=COLOR_REALE, width=2.5)
    ))
    fig_previsioni.add_trace(go.Scatter(
        x=df["Data"], y=serie_sma, 
        mode="lines", name=f"SMA ({finestra_sma}w)", 
        line=dict(color=COLOR_SMA, width=2, dash="dash")
    ))
    fig_previsioni.add_trace(go.Scatter(
        x=df["Data"], y=serie_ses, 
        mode="lines", name=f"SES (α={alpha_ses})", 
        line=dict(color=COLOR_SES, width=2, dash="dot")
    ))
    
    fig_previsioni.update_layout(
        title=f"Confronto Serie Storica vs Previsioni: {articolo_scelto}",
        xaxis_title="Orizzonte Temporale (Settimane)",
        yaxis_title="Volume Assorbito (Pallet)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_previsioni, use_container_width=True)

# =========================================================
# TAB 2: DIMENSIONAMENTO SCORTE
# =========================================================
with tab_scorte:
    st.markdown(f"### Scomposizione del Fabbisogno a Magazzino (Lead Time: **{lead_time}w** | SL: **{int(servizio*100)}%**)")

    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Pipeline Stock", f"{scorte['Pipeline Stock']:.0f} pal.")
        c2.metric("Safety Stock", f"{scorte['Safety Stock']:.0f} pal.")
        c3.metric("Cycle Stock", f"{scorte['Cycle Stock']:.0f} pal.")
        c4.metric("Anticipation Stock", f"{scorte['Anticipation Stock']:.0f} pal.")
        c5.metric("Giacenza Totale", f"{scorte['Scorta Totale']:.0f} pal.")

    # Grafico a Barre delle Scorte
    df_scorte_plot = pd.DataFrame({
        "Componente": ["Pipeline Stock", "Safety Stock", "Cycle Stock", "Anticipation Stock"],
        "Pallet": [
            scorte["Pipeline Stock"],
            scorte["Safety Stock"],
            scorte["Cycle Stock"],
            scorte["Anticipation Stock"]
        ]
    })

    fig_scorte = px.bar(
        df_scorte_plot,
        x="Componente",
        y="Pallet",
        text="Pallet",
        color="Componente",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        title="Ripartizione Funzionale delle Giacenze di Magazzino"
    )
    fig_scorte.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_scorte.update_layout(
        template="plotly_white",
        showlegend=False, 
        yaxis_title="Quantità (Pallet)",
        xaxis_title="",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_scorte, use_container_width=True)

# =========================================================
# TAB 3: SCHEDA DESCRITTIVA ARTICOLO
# =========================================================
with tab_dettagli:
    st.markdown(f"### Scheda Operativa: **{articolo_scelto}**")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        with st.container(border=True):
            st.markdown("#### Caratteristiche della Domanda")
            if "Stazionario" in articolo_scelto:
                st.write("Articolo a consumo continuativo ad altissima rotazione. Variabilità ridotta e assenza di stagionalità marcata; ideale per approcci di rifornimento standard a flusso teso.")
            elif "Trend" in articolo_scelto:
                st.write("Categoria salutistica/funzionale in costante penetrazione di mercato. Richiede algoritmi reattivi per evitare sottostime sistematiche delle giacenze.")
            elif "Stagionale" in articolo_scelto:
                st.write("Assorbimento concentrato nel periodo tardo-primaverile ed estivo. Rende fondamentale la pianificazione tempestiva dell'Anticipation Stock.")
            else:
                st.write("Specialità da ricorrenza o lotti speciali. Domanda caratterizzata da periodi prolungati di inattività seguiti da picchi improvvisi.")

    with col_info2:
        with st.container(border=True):
            st.markdown("#### Parametri Chiave di Calcolo")
            st.markdown(f"""
            * **Lead Time di Rifornimento:** `{lead_time}` settimane
            * **Livello di Servizio Target:** `{int(servizio*100)}%`
            * **Finestra Media Mobile (SMA):** `{finestra_sma}` periodi
            * **Fattore di Smorzamento (SES α):** `{alpha_ses}`
            """)