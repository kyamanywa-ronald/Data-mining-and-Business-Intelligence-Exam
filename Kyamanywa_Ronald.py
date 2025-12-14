# Import pandas for data manipulation and analysis
import pandas as pd  

# Import TransactionEncoder to convert transactions into one-hot encoded format
from mlxtend.preprocessing import TransactionEncoder  

# Import Apriori algorithm and association rule generator
from mlxtend.frequent_patterns import apriori, association_rules  

# PART A: DATA PREPARATION

print("PART A: DATA PREPARATION\n")

# Define transaction dataset where each list represents items bought together
transactions = [
    ['Bread', 'Milk', 'Eggs'],          
    ['Bread', 'Butter'],               
    ['Milk', 'Diapers', 'Beer'],        
    ['Bread', 'Milk', 'Butter'],       
    ['Milk', 'Diapers', 'Bread'],      
    ['Beer', 'Diapers'],                
    ['Bread', 'Milk', 'Eggs', 'Butter'],
    ['Eggs', 'Milk'],                  
    ['Bread', 'Diapers', 'Beer'],      
    ['Milk', 'Butter']                  
]

# Display the transaction dataset in tabular form
df = pd.DataFrame({
    'Transaction_ID': range(1, 11),
    'Items': [', '.join(items) for items in transactions]
})

print("Transaction Dataset")
print(df.to_string(index=False))
print()

# ONE-HOT ENCODING

print("One-Hot Encoded Dataset")

# Convert transaction data into binary (True/False) format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

# Create DataFrame from encoded data
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(df_encoded)
print()

# PART B: APRIORI ALGORITHM

print("PART B: APRIORI ALGORITHM")
print("Minimum Support = 0.2")
print("Minimum Confidence = 0.5\n")

# Apply Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(
    df_encoded,
    min_support=0.2,
    use_colnames=True
)

print("Frequent Itemsets Found:")
print(frequent_itemsets)
print()

# ASSOCIATION RULE GENERATION

# Generate association rules based on confidence
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.5
)

print("Association Rules Generated:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print()

