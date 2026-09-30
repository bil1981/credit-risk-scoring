import os
import sqlite3
import pandas as pd


def build_unified_database_from_csv(
    train_csv="./src/application_test.csv",
    preds_csv="./src/submission_kernel02.csv",
    fi_csv="./src/feature_importance.csv",
    db_path="credit_risk.db",
    sample_size=None,  # Optionnel : ex. 10000 si vous souhaitez charger un échantillon rapide
):
    conn = sqlite3.connect(db_path)
    print("⏳ Lecture de application_train.csv...")

    # 1. Chargement des données clients initiales
    if os.path.exists(train_csv):
        if sample_size:
            df_client = pd.read_csv(train_csv, nrows=sample_size)
        else:
            df_client = pd.read_csv(train_csv)
        print(f"✔️ Données clients chargées : {df_client.shape[0]} lignes.")
    else:
        print(f"❌ Fichier introuvable : {train_csv}")
        return

    # 2. Fusion avec submission_kernel02.csv si disponible
    if os.path.exists(preds_csv):
        df_preds = pd.read_csv(preds_csv)
        df_preds["DECISION"] = df_preds["TARGET"].apply(
            lambda x: "REFUSED" if x >= 0.10 else "APPROVED"
        )

        # On remplace la TARGET initiale (0/1) par la probabilité prédite par LightGBM si elle existe
        if "TARGET" in df_client.columns:
            df_client = df_client.drop(columns=["TARGET"])

        df_merged = df_client.merge(df_preds, on="SK_ID_CURR", how="inner")

        # Si le fichier submission contenait les SK_ID_CURR du test, on garde la jointure externe si besoin
        if df_merged.empty:
            print(
                "⚠️ Pas de correspondance exacte sur SK_ID_CURR (train vs submission). Fusion outer appliquée."
            )
            df_merged = df_client.merge(df_preds, on="SK_ID_CURR", how="left").fillna(
                {"DECISION": "N/A"}
            )
    else:
        df_merged = df_client.copy()
        df_merged["DECISION"] = "N/A"

    # 3. Écriture dans SQLite
    df_merged.to_sql("client_features", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sk_id ON client_features(SK_ID_CURR);")
    print(
        f"Table 'client_features' enregistrée ({df_merged.shape[1]} colonnes, {len(df_merged)} lignes)."
    )

    # 4. Ingestion de feature_importance.csv
    if os.path.exists(fi_csv):
        df_fi = pd.read_csv(fi_csv)
        if "fold" in df_fi.columns:
            df_fi = (
                df_fi.groupby("feature")["importance"]
                .mean()
                .reset_index()
                .sort_values(by="importance", ascending=False)
            )
        df_fi.to_sql("feature_importance", conn, if_exists="replace", index=False)
        print("Table 'feature_importance' enregistrée.")

    conn.close()


if __name__ == "__main__":
    build_unified_database_from_csv()
