import os
import numpy as np
import pandas as pd
def generate_banking_dataset(filepath, num_records=10000, seed=42):
    np.random.seed(seed)
    
    # Surnames for realistic data representation (French, Spanish, German mix)
    surnames = [
        "Martin", "Bernard", "Thomas", "Petit", "Robert", "Richard", "Durand", "Dubois", "Moreau", "Laurent",
        "Garcia", "Rodriguez", "Gonzalez", "Fernandez", "Lopez", "Martinez", "Sanchez", "Perez", "Gomez", "Martin",
        "Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann"
    ]
    
    # 1. Customer ID (Unique, starting at 15565701)
    customer_ids = np.arange(15565701, 15565701 + num_records)
    
    # 2. Surname
    customer_surnames = np.random.choice(surnames, size=num_records)
    
    # 3. CreditScore (Mean 650, Std 96, clamped between 350 and 850)
    credit_scores = np.random.normal(loc=650, scale=96, size=num_records).astype(int)
    credit_scores = np.clip(credit_scores, 350, 850)
    
    # 4. Geography (France ~50%, Germany ~25%, Spain ~25%)
    geographies = np.random.choice(["France", "Germany", "Spain"], size=num_records, p=[0.50, 0.25, 0.25])
    
    # 5. Gender (Male ~54%, Female ~46%)
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.54, 0.46])
    
    # 6. Age (Skewed distribution, mean ~38.9, min 18, max 92)
    # Using lognormal or gamma to create a realistic age distribution for banking customers
    ages = (np.random.gamma(shape=9, scale=4, size=num_records) + 4).astype(int)
    ages = np.clip(ages, 18, 92)
    
    # 7. Tenure (0 to 10 years, fairly uniform)
    tenures = np.random.randint(0, 11, size=num_records)
    
    # 8. Balance
    # In France and Spain, many customers (~36%) have zero balance. In Germany, almost everyone has a balance.
    balances = np.zeros(num_records)
    for i in range(num_records):
        geo = geographies[i]
        if geo == "Germany":
            # Very few zero balances in Germany
            has_balance = np.random.choice([True, False], p=[0.95, 0.05])
        else:
            # More zero balances in France/Spain
            has_balance = np.random.choice([True, False], p=[0.60, 0.40])
            
        if has_balance:
            balances[i] = np.random.normal(loc=120000, scale=30000)
            
    # Clamp balance to >= 0
    balances = np.clip(balances, 0, None).round(2)
    
    # 9. NumOfProducts (1 to 4)
    # Distribution: 1: ~50.8%, 2: ~45.9%, 3: ~2.7%, 4: ~0.6%
    num_products = np.random.choice([1, 2, 3, 4], size=num_records, p=[0.508, 0.459, 0.027, 0.006])
    
    # 10. HasCrCard (0 or 1, ~70.5% have credit card)
    has_cr_card = np.random.choice([1, 0], size=num_records, p=[0.705, 0.295])
    
    # 11. IsActiveMember (0 or 1, ~51.5% active)
    is_active_member = np.random.choice([1, 0], size=num_records, p=[0.515, 0.485])
    
    # 12. EstimatedSalary (Uniformly distributed between 10,000 and 200,000)
    estimated_salaries = np.random.uniform(low=10000, high=200000, size=num_records).round(2)
    
    # 13. Exited (Churn indicator)
    # Calculate churn log-odds with realistic business dependencies:
    # - Older customers churn more (dramatically so between 45 and 60)
    # - German customers churn at a much higher rate (~32%) than French/Spanish (~16%)
    # - Female customers churn slightly more (~25% vs ~16%)
    # - Active members are much more loyal
    # - Customers with 2 products are highly loyal (lowest churn)
    # - Customers with 1 product have moderate churn
    # - Customers with 3 or 4 products have extremely high churn (product overload / service issues)
    # - Higher balances have a minor positive effect on churn (wealthy customers leaving)
    # - CreditScore has a negative effect (creditworthy customers are slightly more loyal)
    
    exited = np.zeros(num_records, dtype=int)
    for i in range(num_records):
        # Base log odds
        log_odds = -2.3
        
        # Age effect (exponential increase for older ages, peak in late 50s)
        age = ages[i]
        if age < 30:
            log_odds += -0.5
        elif 30 <= age < 45:
            log_odds += 0.0
        elif 45 <= age < 60:
            log_odds += 1.4  # High risk age bracket
        else:
            log_odds += 0.8  # Senior age bracket (retired, stable but some consolidate)
            
        # Geographic effect
        geo = geographies[i]
        if geo == "Germany":
            log_odds += 0.85
        elif geo == "Spain":
            log_odds += -0.05
            
        # Gender effect
        gender = genders[i]
        if gender == "Female":
            log_odds += 0.45
            
        # Activity effect
        active = is_active_member[i]
        if active == 1:
            log_odds += -0.95
            
        # Products effect (V-shaped or critical penalty)
        prod = num_products[i]
        if prod == 1:
            log_odds += 0.3
        elif prod == 2:
            log_odds += -1.2  # Sweet spot! Highly loyal
        elif prod == 3:
            log_odds += 2.2  # Extremely high churn
        elif prod == 4:
            log_odds += 4.5  # Almost guaranteed churn
            
        # Balance & Salary effect
        bal = balances[i]
        sal = estimated_salaries[i]
        if bal > 0:
            # Positive relationship: high balance customers are slightly more prone to churn (attractive competitor offers)
            log_odds += 0.15 * (bal / 100000)
            
        # Credit Score effect
        score = credit_scores[i]
        log_odds += -0.2 * ((score - 600) / 100)
        
        # Calculate probability
        prob = 1 / (1 + np.exp(-log_odds))
        
        # Determine exit (with a bit of noise)
        exited[i] = np.random.choice([1, 0], p=[prob, 1 - prob])
        
    df = pd.DataFrame({
        "CustomerId": customer_ids,
        "Surname": customer_surnames,
        "CreditScore": credit_scores,
        "Geography": geographies,
        "Gender": genders,
        "Age": ages,
        "Tenure": tenures,
        "Balance": balances,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salaries,
        "Exited": exited
    })
    
    # Save to CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset generated successfully with {len(df)} records at: {filepath}")
    print(f"Overall Churn Rate: {df['Exited'].mean() * 100:.2f}%")
    print(f"Churn by Geography:\n{df.groupby('Geography')['Exited'].mean() * 100}")
    print(f"Churn by Products:\n{df.groupby('NumOfProducts')['Exited'].mean() * 100}")
if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "churn_modelling.csv")
    generate_banking_dataset(output_path)
