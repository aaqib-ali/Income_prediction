import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# from this project
import utils.common as common


def check_target_distribution(df_clean):

        current_ax = plt.gca()
        sns.set_style('whitegrid')

        sns.countplot(x='income', hue='income', data=df_clean, ax=current_ax)

        plt.title('Income less <=50K VS Income greater >50K')
        plt.xlabel('Income')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.legend()

        plt.show()
        #common.save_fig('Income less than =50K VS Income greater than 50K', os.path.
                        #join("./results", "Catagorical_features"))


def dist_numeric(df,logy=False):

    # prepare directory
    common.my_mkdir(os.path.join("./results", "Numeric_features"))
    numeric_features = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']

    # plot numeric quantities
    for c in numeric_features:
        print(c)
        try:
            plt.figure(c)
            sns.distplot(df[c], bins=20)
            plt.title(c)
            plt.xlabel(c)
            plt.ylabel("Counts")

            if logy:
                try:
                    plt.yscale('log')
                except:
                    pass
            common.save_fig(c,os.path.join("./results", "Numeric_features"))

        except:
            pass


def count_catagorical(df, logy=False):

    common.my_mkdir(os.path.join("./results", "Cat_features"))

    cat_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race',
                    'sex', 'native_country']

    # plot numeric quantities
    for c in cat_features:
        print(c)

        try:
            plt.figure(c)
            sns.countplot(df[c], hue='income', data=df)
            plt.title(c)
            plt.xlabel(c)
            plt.ylabel("Counts")

            if logy:
                try:
                    plt.yscale('log')
                except:
                    pass
            common.save_fig(c,os.path.join("./results", "Cat_features"))

        except:
            pass


def explore_marital_status(df_clean):

    marital = df_clean['marital_status'].value_counts()

    plt.style.use('default')
    plt.figure(figsize=(10, 10))
    plt.pie(marital.values, labels=marital.index,
            startangle=50, autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Marital Status distribution',fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.axis('equal')
    #plt.show()
    common.save_fig('Marital Status distribution', os.path.join("./results", "Cat_features"))

def explore_relationship(df_clean):

    relationship = df_clean['relationship'].value_counts()

    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    plt.pie(relationship.values, labels=relationship.index,
            startangle=50, autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Relationship distribution',fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.axis('equal')
    #plt.show()
    common.save_fig('Relationship distribution', os.path.join("./results", "Cat_features"))


def explore_race(df_clean):

    race = df_clean['race'].value_counts()
    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    plt.pie(race.values, labels=race.index,
            startangle=50, autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Race distribution',fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.axis('equal')
    #plt.show()
    common.save_fig('Race distribution', os.path.join("./results", "Cat_features"))


def explore_occupation(df_clean):

    occupation = df_clean['occupation'].value_counts()
    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    plt.pie(occupation.values, labels=occupation.index,
            startangle=50, autopct='%1.1f%%')
    centre_circle = plt.Circle((0, 0), 0.7, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Occupation distribution',fontdict={'fontsize': 30, 'fontweight': 'bold'})
    plt.axis('equal')
    plt.show()
    common.save_fig('Occupation distribution', os.path.join("./results", "Cat_features"))




