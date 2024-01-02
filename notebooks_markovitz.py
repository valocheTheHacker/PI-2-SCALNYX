# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar,minimize, LinearConstraint
import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Label, Entry, Button, END
import csv


### main

def main():  ## Code
    ## import values
    #max equity=496
    #max rows=524
    nEquity=496
    nDays=524
    #data import
    returns=pd.read_csv("Cleaned returns with dividends.csv",sep=";",decimal=",")
    returns.drop('Dates', axis=1, inplace=True) #drop date column

    returns=returns.iloc[: nDays, :nEquity]
    
    #pass it through numpy to use matrix
    returns=pd.DataFrame.to_numpy(returns)

    #calculate the average return per stock
    averageRet=np.mean(returns,axis=0) #-> unecessary ?
    covariance=np.cov(np.transpose(returns))

    ## Importation des secteurs
    secteurs=pd.read_csv("Sectors.csv",sep=";") #unecessary ?
    
    
    root = Tk()
    root.title("Markovitz Portfolio Optimization")
    #root.geometry("500x300")
    
    #display labels
    sector_title = Label(root, text="Sectors").grid(row=0, column=0, pady=5)
    lowerb_title = Label(root, text="lowerb").grid(row=0, column=1)
    upperb_title = Label(root, text="upperb").grid(row=0, column=2)
    
    #display and modify constraints of sectors
    sectorList=['Communication Services','Consumer Discretionary','Consumer Staples','Energy','Financials','Health Care','Industrials','Information Technology','Materials','Real Estate','Utilities']
    lowerb=[0,0,0,0,0,0,0,0,0,0,0] #these are default values
    upperb=[1,1,1,1,1,1,1,1,1,1,1]
    lb = []
    ub = []
    i = 1
    for x in sectorList:
        Label(root, text=x).grid(row=i, column=0, sticky="w")
        lb.append(Entry(root))
        lb[i-1].grid(row=i, column=1)
        lb[i-1].insert(END, '0')
        ub.append(Entry(root))
        ub[i-1].grid(row=i, column=2)
        ub[i-1].insert(END, '1')
        i = i + 1
        
    # Label(root, text="Days in the past").grid(row=12, column=0, sticky="w")
    # saisi = Entry(root)
    # saisi.grid(row=12,column=1)
    # saisi.insert(END,'524')
    
    
    
    Button(root, text="Update constraints", command=lambda:update_constraints(lb,ub,lowerb,upperb)).grid(row = len(sectorList)+1, column=0, pady = 20)
    Button(root, text="Volatility Optimization", command=lambda:markovitz_optimization(lowerb,upperb,sectorList, covariance, averageRet, secteurs,nEquity, root)).grid(row = len(sectorList)+1, column=1, pady = 20)
    Button(root, text="Sharp Optimization", command=lambda:sharp_maximization(lowerb,upperb,sectorList, covariance, averageRet, secteurs,nEquity, root)).grid(row = len(sectorList)+1, column=2, pady = 50)
    Button(root, text="Quit", command=root.destroy).grid(row = len(sectorList)+1, column=3, pady = 20)
    root.mainloop()
    

def markovitz_optimization(lowerb,upperb,sectorList,covariance,averageRet,secteurs,nEquity,root):
    
    if(CheckValidConstraints(lowerb, upperb)):
        ##volatility minimization
        constraints=AllConstraints(lowerb, upperb, sectorList, covariance,secteurs,nEquity)
        bounds = [(0, 1) for i in range(covariance.shape[0])]
    
        res = minimize(
            objective_function,
            x0=np.random.random(covariance.shape[0]),
            args=(covariance),
            constraints=constraints,
            bounds=bounds,
        )
        weights=list()
        for i in res.x:
            weights.append(round(i,3))
    
        volatility=res.fun
        returns=averageRet@weights
        print("results of volatility minimization :")
        print("poids :",weights)
        print("somme des poids :",sum(weights))
        print("volatility : ", volatility)
        print("return : ", returns)
        print("sharp ratio : ", returns/volatility)
        #plt.scatter(volatility, returns,c='red', s=30) # red dot
        
        #writing the solution in the csv file
        Weights=pd.DataFrame(data=list(secteurs["EQUITY"]),columns=["Equity"])
        Weights['Weights']=weights
        Weights.to_csv('Optimization Solution.csv',sep=';',index=False)

    
        # Generate weigth sector pie chart
        secteurs_liste=secteurs["GICS Sector Name"].to_list()
    
        weights_sum = sum(weights)
        sector_weights = [0 for i in range(len(sectorList))]
        for i in range(len(secteurs_liste)):
            if(secteurs_liste[i] in sectorList):
                sector_weights[sectorList.index(secteurs_liste[i])] = sector_weights[sectorList.index(secteurs_liste[i])] + weights[i]
    
        for i in range(len(sector_weights)):
            sector_weights[i] = sector_weights[i] / weights_sum
        
        pie_sector_weights = []
        pie_sector_names = []
        for i in range(len(sector_weights)):
            if sector_weights[i] != 0.0:
                pie_sector_weights.append(sector_weights[i])
                pie_sector_names.append(sectorList[i])
        
        fig = Figure()
        plot1 = fig.add_subplot(111) 
        fig.text(x = 0, y = 1, s = "Optimized Portfolio")
        plot1.pie(pie_sector_weights,labels=pie_sector_names,autopct='%1.1f%%')
        canvas = FigureCanvasTkAgg(fig, master = root)   
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=3, rowspan=len(sectorList)+1, padx=5)
        
        Label(root, text=f"volatility: {volatility}  |  return: {returns}  | sharp ratio: {returns/volatility}").grid(row = len(sectorList)+1, column=3)
        
    else:
        print("constraints are not coherent")

def objective_function(x, covariance):
    return np.sqrt(x.T@(covariance@x))  

def update_constraints(lb,ub,lowerb,upperb):
    i = 1
    print("aaaaaaaaaaaaaaa")
    for x in range(len(lb)):
        lowerb[i-1] = float(lb[i-1].get())
        upperb[i-1] = float(ub[i-1].get())
        print(lowerb[i-1], "  ", upperb[i-1])
        i = i + 1
    print()
    
def IndSec(L,covariance,secteurs,nEquity):
    mat=np.zeros(covariance.shape[0])
    n=secteurs["EQUITY"][:nEquity].count() #vu que pour le moment on ne prend que les 4 premières actions.
    for j in range(n):
        if secteurs["GICS Sector Name"][j]==L:
            mat[j]=1
    return mat

def AllConstraints(lowerb,upperb,sectorList,covariance,secteurs,nEquity): 
    allConstraints=list()
    constraint = LinearConstraint(np.ones(covariance.shape[0]), lb=1, ub=1) #contrainte obligatoire: somme des coefs = 1
    allConstraints.append(constraint)
    #cCommServices = LinearConstraint(IndSec(sectorList[0]), lb=lowerb[0], ub=upperb[0])
    #cConsDis =LinearConstraint(IndSec(sectorList[0]), lb=lowerb[0], ub=upperb[0])
    for i in range(len(sectorList)):
        constraint=LinearConstraint(IndSec(sectorList[i],covariance,secteurs,nEquity), lb=lowerb[i], ub=upperb[i])
        allConstraints.append(constraint)
    return allConstraints

def CheckValidConstraints(lowerb,upperb):
    for i in range(len(lowerb)):
        if lowerb[i]<0 or lowerb[i]>1 or (lowerb[i]>upperb[i]):
            return False
        if upperb[i]>1 or upperb[i]<0:
            return False
    if sum(upperb)<1 or sum(lowerb)>1:
        return False
    return True

def quit_fun(root):
    root.quit()
    root.destroy()

def sharp_maximization(lowerb,upperb,sectorList,covariance,averageRet,secteurs,nEquity,root):
    if(CheckValidConstraints(lowerb, upperb)):
         print()
         constraint=AllConstraints(lowerb, upperb, sectorList, covariance,secteurs,nEquity)
         bounds = [(0, 1) for i in range(covariance.shape[0])]
        
        
         def objective_function2(x, covariance,averageRet):
             return (np.sqrt(x.T@(covariance@x))/(averageRet@x))
        
         res = minimize(
             objective_function2,
             x0=10 * np.random.random(covariance.shape[0]),
             args=(covariance,averageRet),
             constraints=constraint,
             bounds=bounds,
         )
         
         weights=list()
         for i in res.x:
             weights.append(round(i,3))
        
         print("results of sharp ratio maximization:")
         print("poids :",weights)
         volatility=np.sqrt(np.transpose(weights)@(covariance@weights))
         returns=averageRet@weights
         print("volatility : ", volatility)
         print("return : ", returns)
         print("sharp ratio : ", returns/volatility)
         #plt.scatter(np.sqrt(np.transpose(weights)@(covariance@weights)), averageRet@weights,c='green', s=30) # red dot

         #writing the solution in the csv file
         Weights=pd.DataFrame(data=list(secteurs["EQUITY"]),columns=["Equity"])
         Weights['Weights']=weights
         Weights.to_csv('Optimization Solution.csv',sep=';',index=False)

         
         
         # Generate weigth sector pie chart
         secteurs_liste=secteurs["GICS Sector Name"].to_list()
    
         weights_sum = sum(weights)
         sector_weights = [0 for i in range(len(sectorList))]
         for i in range(len(secteurs_liste)):
             if(secteurs_liste[i] in sectorList):
                 sector_weights[sectorList.index(secteurs_liste[i])] = sector_weights[sectorList.index(secteurs_liste[i])] + weights[i]
    
         for i in range(len(sector_weights)):
             sector_weights[i] = sector_weights[i] / weights_sum
         
         pie_sector_weights = []
         pie_sector_names = []
         for i in range(len(sector_weights)):
             if sector_weights[i] != 0.0:
                 pie_sector_weights.append(sector_weights[i])
                 pie_sector_names.append(sectorList[i])
         
    
         fig = Figure()
         plot1 = fig.add_subplot(111) 
         fig.text(x = 10, y = 10, s = "Optimized Portfolio")
         plot1.pie(pie_sector_weights,labels=pie_sector_names,autopct='%1.1f%%')
         canvas = FigureCanvasTkAgg(fig, master = root)   
         canvas.draw()
         canvas.get_tk_widget().grid(row=0, column=3, rowspan=len(sectorList)+1, padx=5)
         
         Label(root, text=f"volatility: {volatility}  |  return: {returns}  | sharp ratio: {returns/volatility}").grid(row = len(sectorList)+1, column=3)
    else:
        print("constraints are not coherent")


def objective_function2(x, covariance,averageRet):
    return (np.sqrt(x.T@(covariance@x))/(averageRet@x))

    

def generate_efficient_border():
    # GENERATE THE BORDER ##############################
    '''
    ##
    #simulate the portfolios to represent the efficient border
    nSimulation=1000
    List_Ret = list()
    List_Vol = list()
    List_SR = list()
    for i in range(nSimulation):
        weights=np.random.random(covariance.shape[0])
        weights=weights/sum(weights) #normalisation pour une somme de coef=1
        ret=averageRet@weights
        vol=np.sqrt(weights.T@(covariance@weights))
        List_Ret.append(ret)
        List_Vol.append(vol)
        List_SR.append(ret / vol)
    #plotting the results
    plt.figure()
    plt.scatter(List_Vol, List_Ret,s=10, c=List_SR, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.show()
    ''' 
    ''' mat=np.ones(covariance.shape[0])
    constraint = LinearConstraint(mat, lb=1, ub=1)
    bounds = [(0, 1) for k in range(covariance.shape[0])]
    for i in np.linspace(start=0.002, stop=0.014, num=30):
        constraint2 = LinearConstraint(averageRet, lb=i, ub=i)
        res = minimize(
            objective_function,
            x0=10 * np.random.random(covariance.shape[0]),
            args=(covariance),
            constraints=(constraint,constraint2),
            bounds=bounds,
        )
        weights=list()
        for j in res.x:
            weights.append(round(j,3))
        plt.scatter(res.fun, averageRet@weights,c='red',marker="o", s=2) # red dot
        #print(i) #affiche où on se situe sur le vecteur linspace / ordonée.
    plt.show()
    #fig.savefig('frontiereEfficiente.png') '''
    
main()