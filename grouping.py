import pandas as pd
class_list = pd.read_excel("C:\\Users\\CIMB11\\classes.xlsx")
li_2wheeler = []
li_4wheeler = []
li_Animal = []
li_Food = []
li_Gadget = []
li_Instrument = []
li_Misc = []
li_Nature = []
li_Sports = []
for index, row in class_list.iterrows():
   #print row['c1'], row['c2']
    if row['Category'] == "2 wheeler":
        li_2wheeler.append(row['Sample'])
    elif row['Category'] == "4 wheeler":
        li_4wheeler.append(row['Sample'])
    elif row['Category'] == "Animal":
        li_Animal.append(row['Sample'])
    elif row['Category'] == "Food":
        li_Food.append(row['Sample'])
    elif row['Category'] == "Gadget":
        li_Gadget.append(row['Sample'])
    elif row['Category'] == "Instrument":
        li_Instrument.append(row['Sample'])
    elif row['Category'] == "Misc":
        li_Misc.append(row['Sample'])
    elif row['Category'] == "Nature":
        li_Nature.append(row['Sample'])
    else:
        li_Sports.append(row['Sample'])

print(li_Sports)