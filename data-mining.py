import pandas as pd
from sklearn.impute import SimpleImputer
#############################################################################################
df = pd.read_csv('train.csv', usecols=["PassengerId","HomePlanet","Cabin","Destination","Age","VIP","Name"])
#############################################################################################

imputer = SimpleImputer(strategy='most_frequent')
imputermean = SimpleImputer(strategy='mean')

age_treshold = 60
df = df[df['Age'] <= age_treshold]

df[["PassengerId","HomePlanet","Cabin","Destination","Age","VIP"]] = imputer.fit_transform(df[["PassengerId","HomePlanet","Cabin","Destination","Age","VIP"]])

df.drop_duplicates(inplace=True)



# this one made the ages strange: 
# dfnew = df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())

#############################################################################################



#this one makes a new category in the data, IDK waht's going on here
    #df = df.get_dummies(df, columns=['HomePlanet'], prefix=['HomePlanet'])

print(df.info())
df.to_csv('/home', index=False, header=True)
