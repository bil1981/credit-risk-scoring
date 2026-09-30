import sqlite3
from pathlib import Path
import pandas as pd
import pytest


def get_database_path():
    for path in (
        Path("data/processed/credit_risk.db"),
        Path("data/proceed/credit_risk.db"),
    ):
        if path.exists():
            return path
    return None


@pytest.fixture
def db_connection():
    path = get_database_path()
    if path is None:
        pytest.skip("credit_risk.db introuvable")

    conn = sqlite3.connect(path)

    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn, name):
    return conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()


def get_table_columns(conn, name):
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}


def test_predictions_table_exists(db_connection):
    assert table_exists(db_connection, "predictions") is not None


def test_client_features_table_exists(db_connection):
    assert table_exists(db_connection, "client_features") is not None


def test_feature_importance_table_exists(db_connection):
    assert table_exists(db_connection, "feature_importance") is not None


def test_predictions_has_required_columns(db_connection):
    assert {"SK_ID_CURR", "TARGET", "DECISION"} <= get_table_columns(
        db_connection, "predictions"
    )


def test_client_features_has_sk_id_curr(db_connection):
    assert "SK_ID_CURR" in get_table_columns(db_connection, "client_features")


def test_feature_importance_has_required_columns(db_connection):
    assert {"feature", "importance"} <= get_table_columns(
        db_connection, "feature_importance"
    )


# ============================================================
# AFFICHAGE TEMPORAIRE POUR CAPTURE D'ÉCRAN
# ============================================================
def display_database_info():
    path = get_database_path()

    if path is None:
        print("❌ credit_risk.db introuvable")
        return

    conn = sqlite3.connect(path)

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """,
        conn,
    )

    print("\n===== LISTE DES TABLES =====")
    print(tables.to_string(index=False))

    for table_name in tables["name"]:
        print("\n" + "=" * 60)
        print(f"TABLE : {table_name}")
        print("=" * 60)

        cols_df = pd.read_sql_query(f'PRAGMA table_info("{table_name}")', conn)
        print(cols_df[["name", "type", "notnull", "pk"]].to_string(index=False))

    conn.close()


if __name__ == "__main__":
    display_database_info()
