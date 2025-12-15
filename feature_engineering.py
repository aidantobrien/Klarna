import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # New customer flag 
    df['is_new_customer'] = (df['days_since_first_loan'] < 0).astype(int)

    # Failed payment acceleration
    df['failed_ratio_3m_1y'] = np.where(
        df[['num_failed_payments_3m','num_failed_payments_1y']].isna().any(axis=1),
        np.nan,
        (df['num_failed_payments_3m'] + 1) / (df['num_failed_payments_1y'] + 1)
    )

    # Confirmed payment acceleration
    df['confirmed_ratio_3m_6m'] = np.where(
        df[['num_confirmed_payments_3m','num_confirmed_payments_6m']].isna().any(axis=1),
        np.nan,
        (df['num_confirmed_payments_3m'] + 1) / (df['num_confirmed_payments_6m'] + 1)
    )

    # Repayment speed / ratios
    df['repayment_3m_ratio'] = np.where(
        df[['amount_repaid_3m','amount_repaid_6m']].isna().any(axis=1),
        np.nan,
        (df['amount_repaid_3m'] + 1) / (df['amount_repaid_6m'] + 1)
    )

    # Exposure dynamics
    df['exposure_ratio_7d_14d'] = np.where(
        df[['new_exposure_7d','new_exposure_14d']].isna().any(axis=1),
        np.nan,
        (df['new_exposure_7d'] + 1) / (df['new_exposure_14d'] + 1)
    )

    # Loan size vs historical average
    df['avg_historical_loan'] = np.where(
        df[['existing_klarna_debt','num_active_loans']].isna().any(axis=1),
        np.nan,
        df['existing_klarna_debt'] / (df['num_active_loans'] + 1)
    )

    df['loan_vs_avg'] = np.where(
        df[['loan_amount','avg_historical_loan']].isna().any(axis=1),
        np.nan,
        (df['loan_amount'] + 1) / (df['avg_historical_loan'] + 1)
    )

    # Debt-to-loan ratio
    df['loan_to_existing_debt'] = np.where(
        df[['loan_amount','existing_klarna_debt']].isna().any(axis=1),
        np.nan,
        (df['loan_amount'] + 1) / (df['existing_klarna_debt'] + 1)
    )

    # Convert merchant columns to category
    df[['merchant_group','merchant_category']] = df[['merchant_group','merchant_category']].astype('category')

    return df
