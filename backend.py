import numpy as np
import pandas as pd


def genera_dati(settimane=104):
  np.random.seed(42)
  date = pd.date_range(start="2026-01-01", periods=settimane, freq="W")
  t = np.arange(settimane)

  # 1. Stazionario
  rumore_staz = np.random.normal(0, 60, settimane)
  domanda_staz = np.maximum(0, 1000 + rumore_staz)

  # 2. Trend
  rumore_trend = np.random.normal(0, 25, settimane)
  domanda_trend = np.maximum(0, 300 + (8 * t) + rumore_trend)

  # 3. Stagionale
  rumore_stag = np.random.normal(0, 30, settimane)
  domanda_stag = np.maximum(
      0, 500 + 250 * np.sin(2 * np.pi * (t-15) / 52) + rumore_stag
  )

  # 4. Erratico
  domanda_err = np.random.choice(
      [0, 150, 450, 900], size=settimane, p=[0.60, 0.25, 0.10, 0.05]
  )

  # Creazione DataFrame con i nomi reali dei prodotti Barilla
  df = pd.DataFrame({
        "Data": date,
        "Spaghetti n.5 500g (Stazionario)": domanda_staz.round(1),
        "Pasta di Legumi (Trend)": domanda_trend.round(1),
        "Pesto alla Genovese (Stagionale)": domanda_stag.round(1),
        "Panettone Pan di Stelle (Erratico)": domanda_err,
    })

  return df


def calcola_previsioni(serie, finestra_sma=4, alpha_ses=0.3):
  sma = serie.rolling(window=finestra_sma).mean().shift(1)
  ses = serie.ewm(alpha=alpha_ses, adjust=False).mean().shift(1)
  return sma, ses


def calcola_metriche(reale, previsto):
  maschera = ~previsto.isna()
  y_true = reale[maschera]
  y_pred = previsto[maschera]

  mae = np.mean(np.abs(y_true - y_pred))
  rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
  mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

  return mae, rmse, mape


def dimensiona_scorte(serie, lead_time_settimane=3, livello_servizio=0.95):
  domanda_media = serie.mean()
  dev_std_domanda = serie.std()

  z_dict = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
  z = z_dict.get(livello_servizio, 1.65)

  # 1. Pipeline Stock
  pipeline = domanda_media * lead_time_settimane

  # 2. Safety Stock
  safety = z * dev_std_domanda * np.sqrt(lead_time_settimane)

  # 3. Cycle Stock
  intervallo_ordini = 2
  cycle = (domanda_media * intervallo_ordini) / 2

  # 4. Anticipation Stock
  anticipation = max(
      0, (serie.tail(4).mean() - domanda_media) * lead_time_settimane
  )

  return {
      "Pipeline Stock": round(pipeline, 1),
      "Safety Stock": round(safety, 1),
      "Cycle Stock": round(cycle, 1),
      "Anticipation Stock": round(anticipation, 1),
      "Scorta Totale": round(pipeline + safety + cycle + anticipation, 1),
  }


# Questo blocco viene eseguito SOLO se lanci direttamente "python backend.py"
# Quando invece viene importato da app.py, non crea interferenze
if __name__ == "__main__":
  df_test = genera_dati()
  print("--- PRIME RIGHE DATASET ---")
  print(df_test.head(5))

  serie_test = df_test["Spaghetti n.5 500g (Stazionario)"]
  sma, ses = calcola_previsioni(serie_test)
  mae, rmse, mape = calcola_metriche(serie_test, sma)
  print(f"\nTest Metriche SMA -> MAE: {mae:.2f} | RMSE: {rmse:.2f}")

  scorte_test = dimensiona_scorte(serie_test)
  print(f"Test Scorta Totale: {scorte_test['Scorta Totale']} pezzi")