import numpy as np
import pandas as pd
from scipy import stats as sc
from sklearn.linear_model import LinearRegression

def linearRegModel(df):
    Y=df.iloc[:,0].values
    X=df.iloc[:,1:].values
    model = LinearRegression()
    model.fit(X, Y)
    b0 = model.intercept_
    bi = model.coef_
    y_pred = model.predict(X)
    return b0,bi,y_pred

def calcFullModel(df):
    Y=df.iloc[:,0].values
    X=df.iloc[:,1:].values
    b0,bis,y_pred=linearRegModel(df);
    SSR_full = np.sum((np.mean(Y) - y_pred)**2)
    SSE_full = np.sum((Y - y_pred)**2)
    S_full=SSE_full/(len(Y)-X.shape[1]-1)
    return SSR_full, S_full

def calcReducedModel(cols,df,SSR_full):
    SSR_red={}
    SSR_Cond={}
    for col in cols:
        df1=df.drop(col,axis=1)
        b0,bis,y_pred=linearRegModel(df1);
        Y=df1.iloc[:,0].values
        X=df1.iloc[:,1:].values
        SSR_red[col] = np.sum((y_pred - np.mean(Y))**2)
        SSR_Cond[col] = (SSR_full - SSR_red[col]);
           
    minCol = min(SSR_Cond, key=SSR_Cond.get)
    SSR_min = SSR_Cond[minCol]
    return SSR_red, minCol, SSR_min

def HypothesisTest(df):
    SSR_full, S_full=calcFullModel(df)
    cols=df.columns[1:]
    SSR_red, minCol, SSR_min=calcReducedModel(cols,df,SSR_full)
    print(f"\nSelect Variable {minCol}\n")
    F_Calc = SSR_min / S_full
    v2 = len(df) - len(cols) - 1
    F_Crit = sc.f.ppf(0.95, 1, v2)
    if F_Calc > F_Crit:
        print(f"F_Calc: {F_Calc:.4f} is greater than F_Crit: {F_Crit:.4f}.")
        print(f"The variable {minCol} is significant. Keeping it in the model.\n")
        return 0, minCol
    else:
        print(f"F_Calc: {F_Calc:.4f} is not greater than F_Crit: {F_Crit:.4f}.")
        print(f"The variable {minCol} is not significant. Removing it from the model.")
        return 1, minCol

def showOutput(df):
    b0,bis,y_pred=linearRegModel(df);
    cols=df.columns[1:]
    print(f"b0: {b0:.4f}")
    for col,i in zip(cols,bis):
        print(f"b{col}: {i:.4f}")
    return

def processRun():
    fileName = input("Enter the file name: ")
    df= pd.read_csv(fileName,header=None);
    cols=df.columns[1:]
    len1= len(cols)
    df2=df;
    for i in range(len1):
        if i == len1-1:
            print("Only one variable left. Ending the process.")
            print(f"Final model includes the variable: {df2.columns[1]}\n")
            break
        print(f"\nStage {i+1}:")
        result, minCol = HypothesisTest(df2)
        if result == 1:
            df2 = df2.drop(minCol, axis=1)
        else:
            print("No more variables to remove. Ending the process.")
            print(f"Final model includes the variables: {df2.columns[1:].tolist()}\n")
            break
    print("Full Model Coefficients\n");
    showOutput(df)
    print("\nReduced Model Coefficients\n")
    showOutput(df2)
    return

processRun()