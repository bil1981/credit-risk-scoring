from pandera.pandas import Column, DataFrameSchema, Check


credit_schema = DataFrameSchema(
    {
        "SK_ID_CURR": Column(int, nullable=False, coerce=True, checks=Check.ge(1)),
        "AMT_INCOME_TOTAL": Column(
            float, nullable=True, required=False, coerce=True, checks=Check.ge(0)
        ),
        "AMT_CREDIT": Column(
            float, nullable=True, required=False, coerce=True, checks=Check.ge(0)
        ),
        "AMT_ANNUITY": Column(
            float, nullable=True, required=False, coerce=True, checks=Check.ge(0)
        ),
    },
    strict=False,
)


def validate_credit_data(df):
    """
    Valide les données de crédit avec Pandera.
    """
    return credit_schema.validate(df)
