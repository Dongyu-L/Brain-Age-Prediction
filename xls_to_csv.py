import pandas as pd

infile = r"E:\Brain_Age_Prediction\IXI.xls" 
outfile = r"E:\Brain_Age_Prediction\IXI.csv"

df = pd.read_excel(infile, sheet_name=0) 
df.to_csv(outfile, index=False, encoding="utf-8-sig")
print("saved:", outfile)
