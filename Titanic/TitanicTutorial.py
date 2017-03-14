import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import ensemble
from sklearn.model_selection import KFold



test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")
train_data.rename(columns={'SibSp':'SiblingsSpouses',
                            'Parch':'ParentsChildren'}, inplace = True)
test_data.rename(columns={'SibSp':'SiblingsSpouses',
                            'Parch':'ParentsChildren'}, inplace = True)
test_columns = test_data.columns
train_columns = train_data.columns


# We are going to approximate the ages by using median values. 
# We divide the dataset into male and female, and also look at the name title. 
# 
# Someone with Master in the name is likely to be 18 years or younger. 
# 
# Someone with Miss is either younger than 18, or older but un-married. 
# For those older, we look for females who are not traveling
# with parents or children. 
# It's an approximation, but it should work for our dataset.
# 
# We could have also just taken the median per sex and approximate it that way.


temp = train_data[pd.notnull(train_data['Age'])]
maletemp = temp[temp['Sex'] == 'male']
femaletemp = temp[temp['Sex'] == 'female']


youngmisstemp = femaletemp[femaletemp['Name'].str.contains('Miss')]

youngmisstemp = youngmisstemp[youngmisstemp['ParentsChildren'] > 0]

oldmisstemp = femaletemp[femaletemp['Name'].str.contains('Miss')]

oldmisstemp = oldmisstemp[oldmisstemp['ParentsChildren'] == 0]


mastermedian = maletemp[maletemp['Name']
                .str.contains('Master')]['Age'].median()

mistermedian = maletemp[maletemp['Name'].str.contains('Mr.')]['Age'].median()

mrsmedian = femaletemp[femaletemp['Name'].str.contains('Mrs.')]['Age'].median()

youngmissmedian = youngmisstemp['Age'].median()

oldmissmedian = oldmisstemp['Age'].median()


mastermask = (train_data['Name'].str.contains('Master')) \
            & (train_data['Sex'] == 'male') & (np.isnan(train_data['Age']))

mrmask = (train_data['Name'].str.contains('Mr.')) \
            & (train_data['Sex'] == 'male') & (np.isnan(train_data['Age']))

mrsmask = (train_data['Name'].str.contains('Mrs.')) \
            & (train_data['Sex'] == 'female') & (np.isnan(train_data['Age']))

youngmissmask = (train_data['Name'].str.contains('Miss')) \
        & (train_data['Sex'] == 'female') & (train_data['ParentsChildren']>0) \
        & (np.isnan(train_data['Age']))

oldmissmask = (train_data['Name'].str.contains('Miss')) \
        & (train_data['Sex'] == 'female') & (train_data['ParentsChildren']==0)\
        & (np.isnan(train_data['Age']))


train_data.loc[mastermask, 'Age'] = mastermedian
train_data.loc[mrmask, 'Age'] = mistermedian
train_data.loc[mrsmask, 'Age'] = mrsmedian
train_data.loc[youngmissmask, 'Age'] =youngmissmedian
train_data.loc[oldmissmask, 'Age'] = oldmissmedian


# Looks like we have one entry left; a Dr. Arthur Jackson Brewe.
# We can manually set his age.

train_data.loc[train_data['PassengerId'] == 767, 'Age'] = mistermedian



# So now, our training data has no NaN age values.
# Let's now look at the test set, and do a similar approximation for
# the missing ages there. 
# As another approximation, we will use the same median values as the ones we
# used for the training set.

temp = test_data

mastermask = (temp['Name'].str.contains('Master')) & (temp['Sex'] == 'male') \
                & (np.isnan(temp['Age']))

mrmask = (temp['Name'].str.contains('Mr.')) & (temp['Sex'] == 'male') \
                & (np.isnan(temp['Age']))

mrsmask = (temp['Name'].str.contains('Mrs.')) & (temp['Sex'] == 'female') \
                & (np.isnan(temp['Age']))

youngmissmask = (temp['Name'].str.contains('Miss')) \
                & (temp['Sex'] == 'female') & (temp['ParentsChildren']>0) \
                & (np.isnan(temp['Age']))

oldmissmask = (temp['Name'].str.contains('Miss')) & (temp['Sex'] == 'female') \
                & (temp['ParentsChildren']==0) & (np.isnan(temp['Age']))

oldmissmask2 = (temp['Name'].str.contains('Ms.')) & (temp['Sex'] == 'female') \
                & (temp['ParentsChildren']==0) & (np.isnan(temp['Age']))

temp.loc[mastermask, 'Age'] = mastermedian
temp.loc[mrmask, 'Age'] = mistermedian
temp.loc[mrsmask, 'Age'] = mrsmedian
temp.loc[youngmissmask, 'Age'] = youngmissmedian
temp.loc[oldmissmask, 'Age'] = oldmissmedian
temp.loc[oldmissmask2, 'Age'] = oldmissmedian


test_data_clean = temp


# So we will be using the following features:
# 
# Sex, Pclass, Age, Siblings/Spouses, Parents/Children, Title
#


train_data_features = pd.concat([train_data['Age'], 
                                train_data['Sex'], 
                                train_data['Pclass'], 
                                train_data['SiblingsSpouses'], 
                                train_data['ParentsChildren'], 
                                train_data['Name']], axis=1)
     
train_data_features['SiblingsSpouses'] = \
    train_data_features['SiblingsSpouses'].apply(lambda x: 1 if x > 0 else 0)

train_data_features['ParentsChildren'] = \
    train_data_features['ParentsChildren'].apply(lambda x: 1 if x > 0 else 0)

train_data_features['Sex'] = \
        train_data_features['Sex'].apply(lambda x: 1 if x =='female' else 0)

train_data_features['Name'] = train_data_features['Name'] \
            .apply(lambda x: 0 if ('Mr.' in x or 'Master' in x or 'Mrs.' in x
                                        or 'Miss' in x or 'Ms.' in x) else 1)

adaboostclassifier = ensemble.AdaBoostClassifier()

splits = 10
kf = KFold(n_splits = splits, shuffle = True)
accuracy = 0
for train_fold, cv_fold in kf.split(train_data_features):    
    adaboostclassifier.fit(train_data_features.loc[train_fold],
                            train_data.loc[train_fold,'Survived'])

    y_true = train_data.loc[cv_fold, 'Survived']

    accuracy = accuracy + adaboostclassifier.score \
                                (train_data_features.loc[cv_fold], y_true)
    
accuracy = accuracy/10
print ("Adaboost Decision tree accuracy: ", accuracy)


# Let's try to now work with Adaboost and vary it's parameters.

splits = 10
kf = KFold(n_splits = 10, shuffle = True)
accuracy = 0
max_accuracy = 0
best_estimators = 0
total_estimators = [10,20,30,40,50,60,70,80,90,100]
for estimators in total_estimators:
    adaboostclassifier = ensemble.AdaBoostClassifier(n_estimators=estimators)
    for train_fold, cv_fold in kf.split(train_data_features):    
        adaboostclassifier.fit(train_data_features.loc[train_fold],
                                train_data.loc[train_fold,'Survived'])

        y_true = train_data.loc[cv_fold, 'Survived']
        accuracy = accuracy + adaboostclassifier.score \
                                    (train_data_features.loc[cv_fold], y_true)    
    accuracy = accuracy/10
    if (accuracy > max_accuracy):
        max_accuracy = accuracy
        best_estimators = estimators

print ("Adaboost Decision tree max accuracy:", max_accuracy, "at",
                                            best_estimators, "estimators.")


test_data_clean_features = pd.concat([test_data_clean['Age'], 
                        test_data_clean['Sex'], test_data_clean['Pclass'], 
                        test_data_clean['SiblingsSpouses'], 
                        test_data_clean['ParentsChildren'], 
                        test_data_clean['Name']], axis=1)
                        
PID = test_data_clean['PassengerId']

test_data_clean_features['SiblingsSpouses'] = \
test_data_clean_features['SiblingsSpouses'].apply(lambda x: 1 if x > 0 else 0)

test_data_clean_features['ParentsChildren'] = \
test_data_clean_features['ParentsChildren'].apply(lambda x: 1 if x > 0 else 0)

test_data_clean_features['Sex'] = test_data_clean_features['Sex'].apply \
                                        (lambda x: 1 if x =='female' else 0)

test_data_clean_features['Name'] = test_data_clean_features['Name'].apply \
(lambda x: 0 if ('Mr.' in x or 'Master' in x or 'Mrs.' in x or 'Miss' in x
                                                        or 'Ms.' in x) else 1)

adaboostclassifier = ensemble.AdaBoostClassifier(n_estimators=70)
adaboostclassifier.fit(train_data_features, train_data['Survived'])
test_predictions = adaboostclassifier.predict(test_data_clean_features)
submission = pd.DataFrame({"PassengerId" : PID, "Survived" : test_predictions})

submission.to_csv("submission.csv", index=False)
