import numpy as np
import pandas as pd
from scipy import stats as sc
import math as mt
from sklearn.linear_model import LinearRegression

def takeInput():
    print("Choose Alternative Hypothisis")
    B1 = float(input("Enter the value of B1: "))
    a=int(input(f"1: B1 != {B1}\n2: B1 < {B1}\n3: B1 > {B1}\nSelect: "))
    H1=[f"B1 != {B1}", f"B1 < {B1}", f"B1 > {B1}"]
    alpha = float(input("Enter significance level (e.g., 0.05): "))
    return a, alpha, H1, B1

def linearRegModel(df):
    Y=df.iloc[:,0].values;
    X=df.iloc[:,1].values.reshape(-1,1)
    model=LinearRegression().fit(X,Y);
    y_pred=model.predict(X);
    b0=model.intercept_
    b1=model.coef_[0]
    print(f"\nIntercept (b0): {b0:.4f}")
    print(f"Slope (b1): {b1:.4f}")
    return b0,b1,y_pred
    
def calcCriticalValue(a,n,alpha):
    if a == 1:
        return sc.t.ppf(1 - alpha/2, n)
    elif a == 2 or a == 3:
        return sc.t.ppf(1 - alpha, n)

def showFullTestResults(a, T, T_Crit,H1, alpha):
    print("\nComplete Test...\n\nHypothesis:")
    print("H0: B1 = 0")
    print(f"H1: {H1[a-1]}\n")
    print("Level of Significance: ", alpha)
    print(f"\nCalculated T: {T:.4f}")
    print(f"Critical T: {T_Crit:.4f}\n")
    
    if a == 1:
        if abs(T) > T_Crit:
            print(f"Reject H0: |T| = {abs(T):.4f} > T_Crit = {T_Crit:.4f}")
        else:
            print(f"Fail to Reject H0: |T| = {abs(T):.4f} <= T_Crit = {T_Crit:.4f}")
    elif a == 2:
        if T < -T_Crit:
            print(f"Reject H0: T = {T:.4f} < -T_Crit = {-T_Crit:.4f}")
        else:
            print(f"Fail to Reject H0: T = {T:.4f} >= -T_Crit = {-T_Crit:.4f}")
    elif a == 3:
        if T > T_Crit:
            print(f"Reject H0: T = {T:.4f} > T_Crit = {T_Crit:.4f}")
        else:
            print(f"Fail to Reject H0: T = {T:.4f} <= T_Crit = {T_Crit:.4f}")
    print("\n")
    return

def hypothesisTest(df,a,alpha,H1,B1):
    Y=df.iloc[:,0].values;
    X=df.iloc[:,1].values;
    n=len(Y)-2
    b0,b1,y_pred=linearRegModel(df);
    SSE=np.sum((Y-y_pred)**2)
    s=mt.sqrt(SSE/n)
    Sxx=np.sum((X-(np.mean(X)))**2)
    T=(b1-B1)/(s/mt.sqrt(Sxx))
    T_Crit = calcCriticalValue(a,n,alpha)
    showFullTestResults(a, T, T_Crit,H1, alpha)
    return

def runProcess():
    fName = input("Enter the filename (e.g., a.txt): ")
    df = pd.read_csv(fName,header=None)
    a, alpha, H1, B1 = takeInput()
    hypothesisTest(df,a,alpha,H1,B1)

runProcess()