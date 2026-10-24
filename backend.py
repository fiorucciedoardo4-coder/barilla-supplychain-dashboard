import numpy as np
import pandas as pd
from statistics import NormalDist


# Nomi degli articoli e tipo di andamento della domanda.
# I nomi sono ispirati alla gamma Barilla, ma i dati sono generati
# (la traccia del project work lo consente).
ARTICOLI = {
    "Spaghetti n.5 (Stazionario)": "Stazionario",
    "Pasta Proteica (Trend)": "Trend",
    "Sughi pronti (Stagionale)": "Stagionale",
    "Torte e prodotti da forno (Erratico)": "Erratico",
}

# Unita' di misura usata in tutto il progetto
UNITA = "colli"

# Le prime settimane non hanno ancora una previsione SMA (serve una finestra
# piena), quindi gli errori si calcolano da qui in poi per entrambi i metodi.
# 12 e' la finestra SMA massima che si puo' scegliere nella dashboard.
SETTIMANE_ESCLUSE = 12


def genera_dati(variabilita=1.0, pendenza_trend=8, ampiezza_stag=250, settimane=156):
    # seed fisso: stessi parametri -> stessi dati (riproducibilita')
    np.random.seed(42)
    date = pd.date_range(start="2025-01-01", periods=settimane, freq="W")
    t = np.arange(settimane)

    # 1. Stazionario: domanda media costante + rumore
    rumore_staz = np.random.normal(0, 60 * variabilita, settimane)
    domanda_staz = np.maximum(0, 2500 + rumore_staz)

    # 2. Trend: la domanda cresce di "pendenza_trend" colli ogni settimana
    rumore_trend = np.random.normal(0, 25 * variabilita, settimane)
    domanda_trend = np.maximum(0, 300 + pendenza_trend * t + rumore_trend)

    # 3. Stagionale: sinusoide con periodo 52 settimane, picco a luglio
    rumore_stag = np.random.normal(0, 30 * variabilita, settimane)
    domanda_stag = np.maximum(
        0, 500 + ampiezza_stag * np.sin(2 * np.pi * (t - 15) / 52) + rumore_stag
    )

    # 4. Erratico: molte settimane a zero e ordini sporadici
    domanda_err = np.random.choice(
        [0, 150, 450, 900], size=settimane, p=[0.60, 0.25, 0.10, 0.05]
    )

    nomi = list(ARTICOLI.keys())
    df = pd.DataFrame({
        "Data": date,
        nomi[0]: domanda_staz.round(1),
        nomi[1]: domanda_trend.round(1),
        nomi[2]: domanda_stag.round(1),
        nomi[3]: domanda_err,
    })

    return df


# Media mobile semplice: come la sliding window vista a lezione, ma la
# finestra contiene solo le n settimane PRECEDENTI (non si puo' usare il futuro).
# La lista restituita ha un elemento in piu' della serie: l'ultimo valore
# e' la previsione per la settimana successiva all'ultima osservata.
def media_mobile(serie, n):
    valori = list(serie)
    previsioni = [np.nan] * (len(valori) + 1)
    for t in range(n, len(valori) + 1):
        somma = 0
        for j in range(t - n, t):
            somma += valori[j]
        previsioni[t] = somma / n
    return previsioni


# Attenuazione esponenziale semplice (SES):
# previsione(t) = alpha * domanda(t-1) + (1 - alpha) * previsione(t-1)
# Come valore iniziale si usa la prima domanda osservata.
def attenuazione_esponenziale(serie, alpha):
    valori = list(serie)
    previsioni = [np.nan] * (len(valori) + 1)
    previsioni[1] = valori[0]
    for t in range(2, len(valori) + 1):
        previsioni[t] = alpha * valori[t - 1] + (1 - alpha) * previsioni[t - 1]
    return previsioni


def calcola_previsioni(serie, finestra_sma=4, alpha_ses=0.3):
    sma = media_mobile(serie, finestra_sma)
    ses = attenuazione_esponenziale(serie, alpha_ses)
    return sma, ses


def calcola_metriche(reale, previsto):
    # stesse settimane per tutti i metodi, cosi' il confronto e' corretto
    y_true = np.array(reale)[SETTIMANE_ESCLUSE:]
    y_pred = np.array(previsto[:len(reale)])[SETTIMANE_ESCLUSE:]
    errori = y_true - y_pred

    mae = np.mean(np.abs(errori))
    rmse = np.sqrt(np.mean(errori ** 2))
    # Il MAPE non si usa: con l'articolo erratico la domanda vale spesso 0
    # e la divisione per zero lo rende privo di significato. Si rapporta
    # invece il MAE alla domanda media (errore % sulla domanda media).
    mae_perc = mae / np.mean(y_true) * 100

    return mae, rmse, mae_perc


# Criterio di scelta: il metodo con RMSE piu' basso.
# L'RMSE e' anche la stima della deviazione standard dell'errore di
# previsione usata per la scorta di sicurezza, quindi il metodo scelto
# e' quello che richiede meno scorta di sicurezza.
def scegli_metodo(rmse_sma, rmse_ses):
    if rmse_ses < rmse_sma:
        return "SES"
    else:
        return "SMA"


def dimensiona_scorte(serie, tipo, previsione, rmse, lead_time_settimane=2,
                      livello_servizio=0.95, intervallo_riordino=2,
                      margine_capacita=0.10):
    # previsione = domanda settimanale prevista (dal metodo scelto)
    # rmse = errore del metodo scelto

    # z esatto della normale standard (modulo statistics)
    z = NormalDist().inv_cdf(livello_servizio)

    # Pipeline Stock: merce ordinata e in viaggio durante il lead time
    pipeline = previsione * lead_time_settimane

    # Safety Stock: copre l'errore di previsione durante il lead time
    safety = z * rmse * np.sqrt(lead_time_settimane)

    # Cycle Stock: ordino ogni "intervallo_riordino" settimane la quantita'
    # Q = previsione * intervallo; in media a magazzino c'e' Q/2
    lotto = previsione * intervallo_riordino
    cycle = lotto / 2

    # Anticipation Stock: solo per l'articolo stagionale (stagionalita'
    # prevedibile). Gli stabilimenti producono al massimo la domanda media
    # aumentata del margine; nelle settimane di picco la domanda oltre la
    # capacita' deve essere prodotta prima e messa a scorta.
    # Per i picchi casuali (articolo erratico) serve la scorta di sicurezza.
    anticipation = 0
    if tipo == "Stagionale":
        valori = np.array(serie)
        # profilo stagionale: media di ogni settimana dell'anno sui 3 anni
        profilo = []
        for settimana in range(52):
            profilo.append(np.mean(valori[settimana::52]))
        capacita = np.mean(profilo) * (1 + margine_capacita)
        for domanda in profilo:
            if domanda > capacita:
                anticipation += domanda - capacita

    return {
        "Pipeline Stock": round(pipeline, 1),
        "Safety Stock": round(safety, 1),
        "Cycle Stock": round(cycle, 1),
        "Anticipation Stock": round(anticipation, 1),
        "Scorta Totale": round(pipeline + safety + cycle + anticipation, 1),
    }


# Prova del backend senza dashboard: python backend.py
if __name__ == "__main__":
    df_test = genera_dati()
    print("--- PRIME RIGHE DATASET ---")
    print(df_test.head(5))

    for nome in ARTICOLI:
        serie_test = df_test[nome]
        sma, ses = calcola_previsioni(serie_test)
        mae_sma, rmse_sma, perc_sma = calcola_metriche(serie_test, sma)
        mae_ses, rmse_ses, perc_ses = calcola_metriche(serie_test, ses)
        metodo = scegli_metodo(rmse_sma, rmse_ses)
        if metodo == "SMA":
            scorte = dimensiona_scorte(serie_test, ARTICOLI[nome], sma[-1], rmse_sma)
        else:
            scorte = dimensiona_scorte(serie_test, ARTICOLI[nome], ses[-1], rmse_ses)
        print(f"\n{nome}")
        print(f"  SMA -> MAE: {mae_sma:.1f} | RMSE: {rmse_sma:.1f} | MAE%: {perc_sma:.1f}")
        print(f"  SES -> MAE: {mae_ses:.1f} | RMSE: {rmse_ses:.1f} | MAE%: {perc_ses:.1f}")
        print(f"  Metodo scelto: {metodo} - Scorta totale: {scorte['Scorta Totale']} {UNITA}")
