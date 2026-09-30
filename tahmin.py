"""
Gunluk futbol tahmini - Caff1982/football-predictions fikrinin (form EMA'lari +
bahis oranlari -> sinir agi) GitHub Actions'ta kendi kendine calisan hali.

Veri: football-data.co.uk (gecmis sezonlar + yaklasan maclar ve oranlari)
Cikti: docs/tahminler.json (site bunu okur), docs/arsiv.csv (gecmis tahminler)
"""
import io, json, os, time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LIGLER = {
    "E0": "İngiltere Premier Lig", "E1": "İngiltere Championship", "E2": "İngiltere League One",
    "E3": "İngiltere League Two", "EC": "İngiltere Conference", "SC0": "İskoçya Premiership",
    "SC1": "İskoçya Championship", "D1": "Almanya Bundesliga", "D2": "Almanya 2. Bundesliga",
    "I1": "İtalya Serie A", "I2": "İtalya Serie B", "SP1": "İspanya La Liga",
    "SP2": "İspanya Segunda", "F1": "Fransa Ligue 1", "F2": "Fransa Ligue 2",
    "N1": "Hollanda Eredivisie", "B1": "Belçika Pro Lig", "P1": "Portekiz Liga",
    "T1": "Türkiye Süper Lig", "G1": "Yunanistan Süper Lig",
}
SEZON_SAYISI = 4          # egitim icin son kac sezon
EMA_SPAN = 10             # form penceresi (mac)
DEGER_ESIGI = 0.05        # model olasiligi x oran - 1 >= %5 ise "deger" isareti
BASE = "https://www.football-data.co.uk"
DOCS = "docs"
TR = ZoneInfo("Europe/Istanbul")
UK = ZoneInfo("Europe/London")
UA = {"User-Agent": "Mozilla/5.0 (gunluk-tahmin; GitHub Actions)"}


def csv_indir(url):
    for deneme in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 200 and len(r.content) > 100:
                return pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
            return None
        except Exception as e:
            print("indirme hatasi", url, e)
            time.sleep(3)
    return None


def sezon_kodlari(bugun):
    y = bugun.year % 100 if bugun.month >= 7 else (bugun.year - 1) % 100
    return [f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}" for i in range(SEZON_SAYISI)]


def gecmisi_yukle(bugun):
    parcalar = []
    for sezon in sezon_kodlari(bugun):
        for div in LIGLER:
            df = csv_indir(f"{BASE}/mmz4281/{sezon}/{div}.csv")
            if df is not None and "HomeTeam" in df.columns:
                df["Div"] = div
                parcalar.append(df)
    gecmis = pd.concat(parcalar, ignore_index=True)
    return temizle(gecmis)


def temizle(df):
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    for c in ["HomeTeam", "AwayTeam"]:
        df[c] = df[c].astype(str).str.strip()
    return df


def oran_kolonlari(df):
    """Ortalama oran yoksa bet365'e dus."""
    def sec(*adaylar):
        out = pd.Series(np.nan, index=df.index)
        for a in adaylar:
            if a in df.columns:
                out = out.fillna(pd.to_numeric(df[a], errors="coerce").where(lambda s: s > 1))
        return out
    df["oH"], df["oD"], df["oA"] = sec("AvgH", "BbAvH", "B365H"), sec("AvgD", "BbAvD", "B365D"), sec("AvgA", "BbAvA", "B365A")
    df["oU"], df["oAlt"] = sec("Avg>2.5", "BbAv>2.5", "B365>2.5"), sec("Avg<2.5", "BbAv<2.5", "B365<2.5")
    ters = 1 / df[["oH", "oD", "oA"]]
    toplam = ters.sum(axis=1)
    df["pH"], df["pD"], df["pA"] = ters["oH"] / toplam, ters["oD"] / toplam, ters["oA"] / toplam
    ou = 1 / df["oU"] + 1 / df["oAlt"]
    df["pUst"] = (1 / df["oU"]) / ou
    return df


ISTAT = [("FTHG", "FTAG", "gol"), ("HS", "AS", "sut"), ("HST", "AST", "isabet"), ("HC", "AC", "korner")]


def form_ozellikleri(gecmis):
    """Her takim icin mac oncesi EMA (sadece o maca kadar oynanan maclarla)."""
    g = gecmis.sort_values("Date").reset_index(drop=True)
    g["mid"] = np.arange(len(g))
    satirlar = []
    for taraf, rakip, ev in [("Home", "Away", 1), ("Away", "Home", 0)]:
        d = pd.DataFrame({"mid": g["mid"], "Date": g["Date"], "Takim": g[f"{taraf}Team"], "ev": ev})
        for h, a, ad in ISTAT:
            benim, onun = (h, a) if ev else (a, h)
            d[f"{ad}_at"] = pd.to_numeric(g.get(benim), errors="coerce")
            d[f"{ad}_ye"] = pd.to_numeric(g.get(onun), errors="coerce")
        d["puan"] = np.select([d["gol_at"] > d["gol_ye"], d["gol_at"] == d["gol_ye"]], [3, 1], 0)
        satirlar.append(d)
    uzun = pd.concat(satirlar).sort_values(["Date", "mid"]).reset_index(drop=True)
    kol = [c for c in uzun.columns if c.endswith(("_at", "_ye"))] + ["puan"]
    grp = uzun.groupby("Takim")
    ema = grp[kol].transform(lambda s: s.ewm(span=EMA_SPAN, ignore_na=True).mean())
    onceki = ema.groupby(uzun["Takim"]).shift(1)          # mac ONCESI form
    uzun["mac_sayisi"] = grp.cumcount()
    on = pd.concat([uzun[["mid", "Takim", "ev", "mac_sayisi"]], onceki.add_prefix("f_")], axis=1)
    son = pd.concat([uzun[["Takim"]], ema.add_prefix("f_")], axis=1).groupby("Takim").last()
    son["mac_sayisi"] = uzun.groupby("Takim").size()
    return g, on, son, kol


def ozellik_tablosu(ev, dep, oranlar):
    X = pd.DataFrame(index=oranlar.index)
    for c in ["pH", "pD", "pA", "pUst"]:
        p = oranlar[c].clip(0.01, 0.99).values
        X[c] = np.log(p / (1 - p))
    for c in ev.columns:
        if c.startswith("f_"):
            X["ev_" + c] = ev[c].values
            X["dep_" + c] = dep[c].values
            X["fark_" + c] = ev[c].values - dep[c].values
    return X


def model_kur():
    # Orijinal repo Keras sinir agi kullaniyordu; testte asiri ogrenip bahisciden kotu
    # olasilik verdi. Duzenli (regularized) lojistik regresyon daha kalibre ve saglam.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def main():
    simdi = datetime.now(TR)
    os.makedirs(DOCS, exist_ok=True)

    gecmis = gecmisi_yukle(simdi)
    oynanan = gecmis.dropna(subset=["FTHG", "FTAG"]).copy()
    oynanan = oran_kolonlari(oynanan)
    print("oynanmis mac:", len(oynanan))

    g, on, son, kol = form_ozellikleri(oynanan)
    g = oran_kolonlari(g)
    ev = on[on.ev == 1].set_index("mid").loc[g["mid"]]
    dep = on[on.ev == 0].set_index("mid").loc[g["mid"]]
    X = ozellik_tablosu(ev, dep, g)
    y1x2 = np.select([g.FTHG > g.FTAG, g.FTHG == g.FTAG], [0, 1], 2)   # 0=1, 1=X, 2=2
    yust = ((g.FTHG + g.FTAG) > 2).astype(int).values
    gecerli = ((ev.mac_sayisi.values >= 5) & (dep.mac_sayisi.values >= 5) & g[["pH", "pD", "pA"]].notna().all(axis=1).values)
    X, y1x2, yust, gv = X[gecerli], y1x2[gecerli], yust[gecerli], g[gecerli]

    # Zaman sirali dogrulama: son %15'i test, bahisciyle karsilastir
    kes = int(len(X) * 0.85)
    m1 = model_kur().fit(X.iloc[:kes], y1x2[:kes])
    p = m1.predict_proba(X.iloc[kes:])
    bahis = gv.iloc[kes:][["pH", "pD", "pA"]].values
    m2 = model_kur().fit(X.iloc[:kes], yust[:kes])
    pu = m2.predict_proba(X.iloc[kes:])[:, 1]
    bu = gv.iloc[kes:]["pUst"].fillna(0.5).values
    dogrulama = {
        "test_mac": int(len(X) - kes),
        "model_1x2_isabet": round(accuracy_score(y1x2[kes:], p.argmax(1)), 3),
        "bahisci_1x2_isabet": round(accuracy_score(y1x2[kes:], bahis.argmax(1)), 3),
        "model_1x2_logloss": round(log_loss(y1x2[kes:], p, labels=[0, 1, 2]), 4),
        "bahisci_1x2_logloss": round(log_loss(y1x2[kes:], bahis, labels=[0, 1, 2]), 4),
        "model_ust_isabet": round(accuracy_score(yust[kes:], (pu > 0.5).astype(int)), 3),
        "bahisci_ust_isabet": round(accuracy_score(yust[kes:], (bu > 0.5).astype(int)), 3),
        "model_ust_logloss": round(log_loss(yust[kes:], pu, labels=[0, 1]), 4),
        "bahisci_ust_logloss": round(log_loss(yust[kes:], bu, labels=[0, 1]), 4),
    }
    # Model bahisciyi olasilik kalitesinde (logloss) gecemiyorsa "deger" isaretleri gosterilmez
    dogrulama["deger_aktif"] = bool(dogrulama["model_1x2_logloss"] < dogrulama["bahisci_1x2_logloss"])
    dogrulama["deger_aktif_ust"] = bool(dogrulama["model_ust_logloss"] < dogrulama["bahisci_ust_logloss"])
    print(dogrulama)

    # Tum veriyle yeniden egit
    m1 = model_kur().fit(X, y1x2)
    m2 = model_kur().fit(X, yust)

    # Yaklasan maclar
    fik = csv_indir(f"{BASE}/fixtures.csv")
    maclar = []
    if fik is not None and "HomeTeam" in fik.columns:
        fik = temizle(fik)
        fik = fik[fik["Div"].isin(LIGLER)].copy()
        fik = oran_kolonlari(fik)
        fik = fik[fik["Date"].dt.date >= simdi.date()].reset_index(drop=True)
        if len(fik):
            bos = pd.Series(np.nan, index=[c for c in son.columns])
            evf = pd.DataFrame([son.loc[t] if t in son.index else bos for t in fik.HomeTeam]).reset_index(drop=True)
            depf = pd.DataFrame([son.loc[t] if t in son.index else bos for t in fik.AwayTeam]).reset_index(drop=True)
            Xf = ozellik_tablosu(evf, depf, fik)[X.columns]
            P = m1.predict_proba(Xf)
            PU = m2.predict_proba(Xf)[:, 1]
            for i, r in fik.iterrows():
                pr = {"1": float(P[i, 0]), "X": float(P[i, 1]), "2": float(P[i, 2])}
                saat = str(r.get("Time", "") or "")
                try:
                    uk = datetime.strptime(f"{r.Date.date()} {saat}", "%Y-%m-%d %H:%M").replace(tzinfo=UK)
                    tr = uk.astimezone(TR)
                    tarih, saat_tr = tr.strftime("%Y-%m-%d"), tr.strftime("%H:%M")
                except Exception:
                    tarih, saat_tr = str(r.Date.date()), ""
                oran = {"1": r.oH, "X": r.oD, "2": r.oA, "U": r.oU, "A": r.oAlt}
                olas = {**pr, "U": float(PU[i]), "A": float(1 - PU[i])}
                az_veri = bool(not (evf.loc[i, "mac_sayisi"] >= 5)) or bool(not (depf.loc[i, "mac_sayisi"] >= 5))
                degerler = []
                secenek = (["1", "X", "2"] if dogrulama["deger_aktif"] else []) + (["U", "A"] if dogrulama["deger_aktif_ust"] else [])
                for k in ([] if az_veri else secenek):
                    o = oran[k]
                    if pd.notna(o) and o > 1:
                        ev_ = olas[k] * o - 1
                        if ev_ >= DEGER_ESIGI:
                            degerler.append({"secim": k, "oran": round(float(o), 2), "beklenen": round(ev_, 3)})
                maclar.append({
                    "lig": LIGLER[r.Div], "div": r.Div, "tarih": tarih, "saat": saat_tr,
                    "tarih_uk": r.Date.strftime("%Y-%m-%d"),
                    "ev": r.HomeTeam, "dep": r.AwayTeam,
                    "olasilik": {k: round(v, 3) for k, v in olas.items()},
                    "tahmin": max(pr, key=pr.get),
                    "ust_alt": "Ü2.5" if PU[i] >= 0.5 else "A2.5",
                    "oran": {k: (round(float(v), 2) if pd.notna(v) else None) for k, v in oran.items()},
                    "deger": degerler, "az_veri": az_veri,
                })
    maclar.sort(key=lambda m: (m["tarih"], m["saat"], m["lig"]))

    arsiv_ozet = arsivi_guncelle(maclar, gecmis)

    json.dump({
        "guncelleme": simdi.strftime("%Y-%m-%d %H:%M"),
        "dogrulama": dogrulama, "arsiv": arsiv_ozet, "maclar": maclar,
    }, open(f"{DOCS}/tahminler.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("yaklasan mac:", len(maclar))


def arsivi_guncelle(maclar, gecmis):
    """Tahminleri sakla, sonuclananlari notla (kagit ustu takip)."""
    yol = f"{DOCS}/arsiv.csv"
    arsiv = pd.read_csv(yol, dtype=str) if os.path.exists(yol) else pd.DataFrame()
    yeni = pd.DataFrame([{
        "div": m["div"], "tarih_uk": m["tarih_uk"], "ev": m["ev"], "dep": m["dep"], "tarih": m["tarih"],
        "tahmin": m["tahmin"], "ust_alt": m["ust_alt"],
        "p1": m["olasilik"]["1"], "pX": m["olasilik"]["X"], "p2": m["olasilik"]["2"], "pU": m["olasilik"]["U"],
        "deger": ";".join(f'{d["secim"]}@{d["oran"]}' for d in m["deger"]),
    } for m in maclar]).astype(str)
    anahtar = ["div", "tarih_uk", "ev", "dep"]
    if len(arsiv):
        # Sonucu belli olmayan eski kayitlar yeni tahminle guncellenir
        arsiv = arsiv[~arsiv.set_index(anahtar).index.isin(yeni.set_index(anahtar).index)] if len(yeni) else arsiv
    arsiv = pd.concat([arsiv, yeni], ignore_index=True)
    arsiv.to_csv(yol, index=False)

    sonuc = gecmis.dropna(subset=["FTHG", "FTAG"]).copy()
    sonuc["tarih_uk"] = sonuc["Date"].dt.strftime("%Y-%m-%d")
    sonuc = sonuc.rename(columns={"Div": "div", "HomeTeam": "ev", "AwayTeam": "dep"})
    sonuc = oran_kolonlari(sonuc)
    b = arsiv.merge(sonuc[anahtar + ["FTHG", "FTAG", "oH", "oD", "oA", "oU", "oAlt"]], on=anahtar, how="inner")
    if not len(b):
        return {"notlanan": 0}
    fthg, ftag = b.FTHG.astype(float), b.FTAG.astype(float)
    gercek = np.select([fthg > ftag, fthg == ftag], ["1", "X"], "2")
    ust = np.where(fthg + ftag > 2, "Ü2.5", "A2.5")
    kazanc, adet = 0.0, 0
    for i, r in b.iterrows():
        if not isinstance(r["deger"], str) or r["deger"] in ("", "nan"):
            continue
        for parca in r["deger"].split(";"):
            secim, o = parca.split("@")
            o = float(o)
            tuttu = (secim in "1X2" and secim == gercek[i]) or (secim == "U" and ust[i] == "Ü2.5") or (secim == "A" and ust[i] == "A2.5")
            kazanc += (o - 1) if tuttu else -1
            adet += 1
    return {
        "notlanan": int(len(b)),
        "isabet_1x2": round(float((b.tahmin.values == gercek).mean()), 3),
        "isabet_ust_alt": round(float((b.ust_alt.values == ust).mean()), 3),
        "deger_bahis": adet, "deger_kar_birim": round(kazanc, 2),
        "deger_roi": round(kazanc / adet, 3) if adet else None,
    }


if __name__ == "__main__":
    main()
