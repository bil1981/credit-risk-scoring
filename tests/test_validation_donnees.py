# ----------------------------
# L'intérêt principal de de ces tests est d'effectuer des tests unitaires ciblés et rapides sur la validation de données, indépendamment de l'API HTTP.
# ------------------------

import pandas as pd
import pytest
import pandera.pandas as pa

from validation.schemas import validate_credit_data


def test_valid_credit_data():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "AMT_INCOME_TOTAL": [150000.0],
            "AMT_CREDIT": [500000.0],
            "AMT_ANNUITY": [25000.0],
        }
    )

    validated_df = validate_credit_data(df)

    assert len(validated_df) == 1


# Test d'une valeur négative
def test_negative_credit():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [100001],
            "AMT_INCOME_TOTAL": [150000.0],
            "AMT_CREDIT": [-500000.0],
            "AMT_ANNUITY": [25000.0],
        }
    )

    with pytest.raises(pa.errors.SchemaError):
        validate_credit_data(df)


"""
pytest pour les tests de fonctions, a la demande
Pandera/pydantic pour la validation des données, ca tourne tt le temps. il apparait dans les logs de l'API, et si ca ne passe pas, ca bloque la requete.
logging : creation de logger, lors de lancement du code
"""
